import numpy as np
import pytest

from ssapy_toolkit.engines import rescale_burn as exported_rescale_burn
from ssapy_toolkit.engines.rescale_burn import rescale_burn


def test_rescale_burn_modes_and_package_export():
    assert exported_rescale_burn is rescale_burn

    a, t, dv, thrust, impulse = rescale_burn(a0=0.2, m0=100.0, t0=10.0, m=200.0, t=5.0)
    assert np.isclose(a, 0.1)
    assert np.isclose(t, 5.0)
    assert np.isclose(dv, 0.5)
    assert np.isclose(thrust, 20.0)
    assert np.isclose(impulse, 100.0)

    a_imp, t_imp, dv_imp, thrust_imp, impulse_imp = rescale_burn(
        a0=np.array([0.2, 0.4]),
        m0=100.0,
        t0=10.0,
        m=200.0,
        t=5.0,
        mode="constant_impulse",
    )
    np.testing.assert_allclose(a_imp, [0.2, 0.4])
    np.testing.assert_allclose(t_imp, 5.0)
    np.testing.assert_allclose(dv_imp, [1.0, 2.0])
    np.testing.assert_allclose(thrust_imp, [40.0, 80.0])
    np.testing.assert_allclose(impulse_imp, [200.0, 400.0])

    with pytest.raises(ValueError, match="mode"):
        rescale_burn(0.2, 100.0, 10.0, mode="bad")
