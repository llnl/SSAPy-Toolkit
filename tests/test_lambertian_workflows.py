import numpy as np
import pytest
import astropy.units as u
from astropy.time import Time

from ssapy_toolkit.compute import lambertian_magnitude as lm
from ssapy_toolkit.compute.lambertian_magnitude import lambertian_reflection


def test_lambertian_reflection_vectorized_topocentric_magnitudes():
    time = Time("2024-01-01T00:00:00", scale="utc")
    positions = np.array(
        [
            [42_164_000.0, 0.0, 0.0],
            [0.0, 42_164_000.0, 0.0],
            [-42_164_000.0, 0.0, 0.0],
        ]
    )

    out = lambertian_reflection(
        positions,
        time=time,
        lon=-156.26,
        lat=20.71,
        elevation=3055.0,
        radius_m=1.0,
        albedo=0.2,
    )

    magnitudes = np.asarray(out["ab_mag_observed"])
    assert magnitudes.shape == (3,)
    assert np.isfinite(magnitudes).any()
    assert np.asarray(out["angles_deg"]["phase_sun_obj_obs_deg"]).shape == (3,)


def test_lambertian_helpers_band_shadow_and_airmass_branches():
    assert lm._resolve_band("V") == lm.BANDS["V"]
    assert lm._resolve_band((500 * u.nm, 600 * u.nm)) == pytest.approx((500e-9, 600e-9))
    with pytest.raises(ValueError, match="Unknown band"):
        lm._resolve_band("not-a-band")
    with pytest.raises(ValueError, match="Band must"):
        lm._resolve_band((2.0, 1.0))

    assert lm._ab_mag(0.0, 500e-9, 600e-9) == np.inf
    assert lm.airmass_kasten_young(0.0) == pytest.approx(0.9997119918558381)
    assert lm.lambert_sphere_phase(0.0) == pytest.approx(2.0 / 3.0)

    r_obj = np.array([7_000_000.0, 0.0, 0.0])
    assert lm.sun_visibility_factor(r_obj, np.array([lm.AU_M, lm.AU_M, 0.0])) == 1.0
    assert lm.sun_visibility_factor(r_obj, np.array([-lm.AU_M, 0.0, 0.0])) == 0.0
    penumbra = lm.sun_visibility_factor(r_obj, np.array([-10_000_000.0, 35_500_000.0, 0.0]), r_sun_radius=1_000_000.0)
    assert 0.0 < penumbra < 1.0


def test_lambertian_reflection_thermal_and_combined_with_fixed_ephemerides(monkeypatch):
    import ssapy.utils as ssapy_utils

    time = Time("2024-01-01T00:00:00", scale="utc")
    monkeypatch.setattr(ssapy_utils, "sunPos", lambda t: np.array([lm.AU_M, 0.0, 0.0]))
    monkeypatch.setattr(ssapy_utils, "moonPos", lambda t: np.array([384_400_000.0, 100_000_000.0, 0.0]))

    observer = np.array([lm.R_EARTH + 1_000_000.0, 0.0, 0.0])
    obj = np.array([lm.R_EARTH + 2_000_000.0, 100_000.0, 0.0])

    reflection = lm.lambertian_reflection(
        obj,
        observer=observer,
        time=time,
        band=(500e-9, 600e-9),
        include_sun=True,
        include_earthshine=True,
        include_moonshine=True,
        radius_m=2.0,
        albedo=0.2,
    )
    assert reflection["band"]["name"] == "custom"
    assert {"sun", "earthshine", "moonshine"} <= set(reflection["irradiance_inband_W_m2"])
    assert reflection["airmass"] == 0.0

    thermal = lm.thermal_emission(obj, observer=observer, time=time, band="LWIR", temperature_K=290.0, albedo=0.1)
    assert set(thermal["irradiance_inband_W_m2"]) == {"thermal"}
    assert np.isfinite(thermal["ab_mag_observed"])

    combined = lm.lambertian_sphere_brightness(
        obj,
        observer=observer,
        time=time,
        band="V",
        include_sun=False,
        include_earthshine=False,
        include_moonshine=False,
        include_thermal=True,
        temperature_K=290.0,
        emissivity=0.5,
    )
    assert set(combined["irradiance_inband_W_m2"]) == {"thermal"}

    vector = lm.thermal_emission(np.vstack([obj, obj + [1000.0, 0.0, 0.0]]), observer=observer, time=Time([time, time]), band="LWIR")
    assert np.asarray(vector["ab_mag_observed"]).shape == (2,)


def test_lambertian_setup_observer_requirements(monkeypatch):
    import ssapy.utils as ssapy_utils

    time = Time("2024-01-01T00:00:00", scale="utc")
    monkeypatch.setattr(ssapy_utils, "sunPos", lambda t: np.array([lm.AU_M, 0.0, 0.0]))
    monkeypatch.setattr(ssapy_utils, "moonPos", lambda t: np.array([384_400_000.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="Provide either"):
        lm.thermal_emission(np.array([7_000_000.0, 0.0, 0.0]), time=time)
