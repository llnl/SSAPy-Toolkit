import numpy as np

from ssapy_toolkit.orbital_mechanics import all_orbit_quantities as canonical_module


def test_all_orbit_quantities_safe_float():
    assert np.isnan(canonical_module._safe_float(None))
    assert np.isnan(canonical_module._safe_float("not-a-number"))
    assert canonical_module._safe_float("3.5") == 3.5
