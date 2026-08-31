from types import SimpleNamespace

import numpy as np
import pytest
import astropy.units as u
from astropy.time import Time

from ssapy_toolkit.compute.faceted_magnitude import (
    facet_scattering_area,
    faceted_light_curve,
    faceted_reflection,
    line_of_sight_blocked,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
EPOCH = Time("2026-06-09T06:00:00")
GEO_RADIUS = 4.2164e7


def _facet(normal=(1.0, 0.0, 0.0), area=4.0, reflectivity=0.5):
    return SimpleNamespace(normal_body=normal, area=area, diffuse_reflectivity=reflectivity)


def test_head_on_facet_matches_closed_form():
    area = facet_scattering_area([_facet()], IDENTITY, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert area == pytest.approx(0.5 * 4.0 / np.pi)


def test_cosine_factors_apply_to_both_paths():
    source = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0])
    area = facet_scattering_area([_facet()], IDENTITY, source, (1.0, 0.0, 0.0))
    assert area == pytest.approx(0.5 * 4.0 * np.cos(np.pi / 3) / np.pi)


@pytest.mark.parametrize(
    "source,observer",
    [((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))],
)
def test_unlit_or_hidden_facets_contribute_nothing(source, observer):
    assert facet_scattering_area([_facet()], IDENTITY, source, observer) == 0.0


def test_attitude_rotates_the_facet_out_of_view():
    half = 0.5 * np.pi / 2.0
    quaternion = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # +90 deg about Z
    assert facet_scattering_area(
        [_facet()], quaternion, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    ) == pytest.approx(0.0, abs=1e-12)


def test_band_keyed_reflectivity_selects_per_band_values():
    facet = _facet(reflectivity={"V": 0.05, "SWIR": 0.30})
    visible = facet_scattering_area(
        [facet], IDENTITY, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), band_name="V"
    )
    infrared = facet_scattering_area(
        [facet], IDENTITY, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), band_name="SWIR"
    )
    assert infrared / visible == pytest.approx(6.0)


def test_missing_band_entry_is_rejected_with_available_bands():
    facet = _facet(reflectivity={"V": 0.05})
    with pytest.raises(ValueError, match="no entry for band 'R'"):
        facet_scattering_area([facet], IDENTITY, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), band_name="R")


@pytest.mark.parametrize("reflectivity", [-0.1, 1.5, np.nan])
def test_invalid_reflectivity_is_rejected(reflectivity):
    with pytest.raises(ValueError, match="diffuse_reflectivity"):
        facet_scattering_area([_facet(reflectivity=reflectivity)], IDENTITY, (1, 0, 0), (1, 0, 0))


def test_missing_reflectivity_is_rejected():
    facet = SimpleNamespace(normal_body=(1.0, 0.0, 0.0), area=1.0)
    with pytest.raises(ValueError, match="diffuse_reflectivity"):
        facet_scattering_area([facet], IDENTITY, (1, 0, 0), (1, 0, 0))


def test_empty_facet_set_is_rejected():
    with pytest.raises(ValueError, match="facets must not be empty"):
        facet_scattering_area([], IDENTITY, (1, 0, 0), (1, 0, 0))


def test_earth_occults_opposite_sides_but_not_the_same_side():
    near = np.array([GEO_RADIUS, 0.0, 0.0])
    far = np.array([-GEO_RADIUS, 0.0, 0.0])
    assert line_of_sight_blocked(near, far)
    assert not line_of_sight_blocked(near, np.array([GEO_RADIUS, 1.0e7, 0.0]))


def test_surface_observer_does_not_occult_itself():
    """A geoid observer sits below the equatorial radius at nonzero latitude."""

    site = np.array([0.0, 0.0, 6.3568e6])  # polar radius, inside R_EARTH
    target = np.array([0.0, 0.0, GEO_RADIUS])
    assert not line_of_sight_blocked(target, site)


def test_occulted_geometry_reports_no_flux():
    facets = [_facet(normal=(1.0, 0.0, 0.0)), _facet(normal=(-1.0, 0.0, 0.0))]
    behind = np.array([-GEO_RADIUS, 0.0, 0.0])
    result = faceted_reflection(
        behind, IDENTITY, facets, observer=np.array([GEO_RADIUS, 0.0, 0.0]), time=EPOCH, band="V"
    )
    assert result["occulted"]
    assert not np.isfinite(result["ab_mag_observed"])


def test_doubling_facet_area_brightens_by_the_expected_amount():
    facets_small = [_facet(normal=direction) for direction in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    facets_large = [
        _facet(normal=direction, area=8.0) for direction in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    position = np.array([GEO_RADIUS, 0.0, 0.0])
    observer = np.array([1.5 * GEO_RADIUS, 1.0e7, 1.0e7])
    common = {"observer": observer, "time": EPOCH, "band": "V"}
    small = faceted_reflection(position, IDENTITY, facets_small, **common)
    large = faceted_reflection(position, IDENTITY, facets_large, **common)
    difference = small["ab_mag_exoatmospheric"] - large["ab_mag_exoatmospheric"]
    assert difference == pytest.approx(2.5 * np.log10(2.0), rel=1e-9)


def test_light_curve_shapes_and_band_reuse_match_single_calls():
    facets = [_facet(normal=direction) for direction in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    samples = 4
    angles = np.linspace(0.0, 0.5, samples)
    positions = np.column_stack(
        [GEO_RADIUS * np.cos(angles), GEO_RADIUS * np.sin(angles), np.zeros(samples)]
    )
    quaternions = np.tile(IDENTITY, (samples, 1))
    times = EPOCH + np.arange(samples) * 60.0 * u.s
    observer = positions * 1.6 + np.array([0.0, 0.0, 5.0e6])

    curves = faceted_light_curve(
        positions, quaternions, facets, times, bands=("V", "SWIR"), observer=observer
    )
    assert set(curves) == {"V", "SWIR"}
    for entry in curves.values():
        assert entry["ab_mag_observed"].shape == (samples,)
        assert entry["occulted"].dtype == bool

    for index in range(samples):
        reference = faceted_reflection(
            positions[index], quaternions[index], facets,
            observer=observer[index], time=times[index], band="SWIR",
        )
        assert curves["SWIR"]["ab_mag_observed"][index] == pytest.approx(
            reference["ab_mag_observed"], rel=1e-12
        )


def test_light_curve_validates_shapes():
    facets = [_facet()]
    times = EPOCH + np.arange(3) * 60.0 * u.s
    with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
        faceted_light_curve(np.zeros((3, 3)), np.zeros((3, 3)), facets, times)
    with pytest.raises(ValueError, match="equal length"):
        faceted_light_curve(np.zeros((3, 3)), np.zeros((2, 4)), facets, times)
