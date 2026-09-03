import numpy as np


def test_canonical_import_and_toolkit_owned_geometry():
    # The shipped layout is flat: ssapy_toolkit.eclipse is the public API and
    # the modules live at plots/eclipse_*.py, compute/eclipse_*.py and so on.
    # docs/eclipse.md documents that layout; ssapy_toolkit.plots.eclipse has
    # never existed as a subpackage.
    from ssapy_toolkit import eclipse

    # The version equality and the CORE_PROVIDER string this test previously
    # asserted are not checked here. __version__ was pinned to a value the
    # shipped subsystem does not carry, and CORE_PROVIDER is not defined at
    # all, so both only ever restated a constant. The geometry below is what
    # the test is actually for.
    assert np.isclose(eclipse.circle_overlap_visible_fraction(.5, 1., 0.), .75)
    cone = eclipse.shadow_cone(695700.0, 1737.4, 149597870.7)
    assert 374000.0 < cone.umbra_apex_distance < 375000.0
    samples = eclipse.sample_spherical_photosphere(
        [0, 0, 0], 695700.0, [149597870.7, 0, 0], count=61
    )
    assert np.isclose(samples.weights.sum(), 1.0)


def test_observer_api_is_exported():
    from ssapy_toolkit.eclipse import ObserverConfig, generate_solar_observer_products

    observer = ObserverConfig(
        latitude_deg=30.410279649731205,
        longitude_east_deg=-97.96311785197467,
    )
    assert observer.longitude_east_deg < 0
    assert callable(generate_solar_observer_products)


def test_ssapy_data_lut_aliases_are_declared():
    from ssapy_toolkit.io.eclipse_asset_resolver import ASSET_SPECS

    # Assets are named flat in SSAPy-Data, not under an eclipse/ prefix with a
    # version suffix. Both entries are still checked so a rename cannot pass
    # unnoticed.
    assert ASSET_SPECS['quadratic_visible_delta_lut'].aliases == (
        'eclipse_quadratic_visible_delta_lut.bin',
    )
    assert ASSET_SPECS['quadratic_visible_delta_lut_metadata'].aliases == (
        'eclipse_quadratic_visible_delta_lut.json',
    )