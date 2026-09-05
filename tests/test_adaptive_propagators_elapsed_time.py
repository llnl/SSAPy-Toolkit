"""The remaining adaptive propagators integrate in elapsed time too.

`propagate_orbit_state`, its STM variant, `propagate_6dof_variational` and
`propagate_6dof_extended` all handed SciPy an absolute span and an absolute
``t_eval``, so they carried the same ``min_step = 10 * ulp(t)`` floor that
`propagate_6dof` did.
"""

import numpy as np
import pytest

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.propagators_orbit.high_accuracy import (
    propagate_orbit_state,
    propagate_orbit_state_with_stm,
)

EPOCHS = pytest.mark.parametrize(
    "epoch", [0.0, 1.0e8, 1.4e9, 3.8e9], ids=["t0", "gps1983", "gps2024", "gps2100"]
)

R0 = [7.0e6, 0.0, 0.0]
V0 = [0.0, float(np.sqrt(EARTH_MU / 7.0e6)), 0.0]


def _thrust_cut_out(epoch):
    """A step-discontinuous acceleration: on for 0.5 s, then off."""

    def acceleration(r, v, t):
        return np.array([1.0, 0.0, 0.0]) if (t - epoch) < 0.5 else np.zeros(3)

    return acceleration


@EPOCHS
def test_step_discontinuity_resolves_at_any_epoch(epoch):
    """Raised RuntimeError from GPS 1e8 upward before this change."""
    result = propagate_orbit_state(
        r0=R0, v0=V0, t0=epoch, times=epoch + np.array([0.0, 1.0, 2.0]),
        mu=EARTH_MU, acceleration=_thrust_cut_out(epoch), rtol=1e-10, atol=1e-12,
    )
    assert result.r.shape == (3, 3)
    assert np.all(np.isfinite(result.r))


@EPOCHS
def test_orbit_state_is_bit_identical_across_epochs(epoch):
    def run(reference):
        # Integer-second offsets are exactly representable at GPS magnitude.
        return propagate_orbit_state(
            r0=R0, v0=V0, t0=reference,
            times=reference + np.arange(0.0, 5880.0, 60.0), mu=EARTH_MU,
        )

    base = run(0.0)
    shifted = run(epoch)
    np.testing.assert_array_equal(shifted.r, base.r)
    np.testing.assert_array_equal(shifted.v, base.v)
    np.testing.assert_array_equal(shifted.t - epoch, base.t)


@EPOCHS
def test_stm_propagation_is_bit_identical_across_epochs(epoch):
    def run(reference):
        return propagate_orbit_state_with_stm(
            r0=R0, v0=V0, t0=reference,
            times=reference + np.arange(0.0, 1800.0, 60.0), mu=EARTH_MU,
        )

    base = run(0.0)
    shifted = run(epoch)
    np.testing.assert_array_equal(shifted.r, base.r)
    np.testing.assert_array_equal(shifted.stm, base.stm)
    np.testing.assert_array_equal(shifted.t - epoch, base.t)


@EPOCHS
def test_acceleration_models_still_receive_absolute_epochs(epoch):
    seen: list[float] = []

    def probe(r, v, t):
        seen.append(float(t))
        return np.zeros(3)

    propagate_orbit_state(
        r0=R0, v0=V0, t0=epoch, times=epoch + np.array([0.0, 10.0]),
        mu=EARTH_MU, acceleration=probe,
    )
    assert seen
    assert min(seen) >= epoch - 1.0e-6
    assert max(seen) <= epoch + 10.0 + 1.0e-6
