"""Fixed-epoch truth cases for the SSATK 6-DoF propagator.

The first case uses SSAPy's ``Orbit.at`` Keplerian propagator as a reference
for the shared point-mass model.  The absolute epoch is GPS seconds, matching
SSAPy's documented float-time convention and SSATK's relative ODE dynamics.
"""

import numpy as np
from ssapy import Orbit

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.propagators_6dof import propagate_6dof_high_accuracy


def test_two_body_fixed_epoch_matches_ssapy_keplerian_reference():
    radius = 7_000_000.0
    speed = np.sqrt(EARTH_MU / radius)
    epoch_gps = 1_400_000_000.0
    period = 2.0 * np.pi * np.sqrt(radius**3 / EARTH_MU)
    times = epoch_gps + np.array([0.0, 600.0, period, 2.0 * period])
    r0 = np.array([radius, 0.0, 0.0])
    v0 = np.array([0.0, speed, 0.0])

    reference = Orbit(r0, v0, epoch_gps, mu=EARTH_MU).at(times)
    trajectory = propagate_6dof_high_accuracy(
        r0=r0,
        v0=v0,
        t0=epoch_gps,
        times=times,
        inertia=np.eye(3),
        mu=EARTH_MU,
    )

    np.testing.assert_allclose(trajectory.r, reference.r, rtol=0.0, atol=1.0e-2)
    np.testing.assert_allclose(trajectory.v, reference.v, rtol=0.0, atol=1.0e-5)
