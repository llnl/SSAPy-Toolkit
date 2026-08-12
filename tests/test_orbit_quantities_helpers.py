import numpy as np

from ssapy_toolkit.orbital_mechanics import all_orbit_quanities as legacy_module
from ssapy_toolkit.orbital_mechanics import all_orbit_quantities as canonical_module


def test_all_orbit_quantities_alias_and_safe_float():
    assert canonical_module.all_orbital_quantities is legacy_module.all_orbital_quantities
    assert np.isnan(legacy_module._safe_float(None))
    assert np.isnan(legacy_module._safe_float("not-a-number"))
    assert legacy_module._safe_float("3.5") == 3.5
