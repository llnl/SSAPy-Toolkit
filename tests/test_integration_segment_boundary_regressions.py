import numpy as np

from ssapy_toolkit.propagators_6dof.high_accuracy import propagate_spacecraft_segments
from ssapy_toolkit.propagators_6dof.sixdof import Spacecraft


def test_ulp_close_segment_start_is_snapped_without_mutating_input():
    epoch = 1.4e9
    times = np.array([np.nextafter(epoch, -np.inf), epoch + 1])
    trajectory = propagate_spacecraft_segments(
        Spacecraft(r=[7e6, 0, 0], v=[0, 7500, 0], t=epoch, inertia=np.eye(3)),
        [{"times": times}],
        mu=0,
    )
    assert trajectory.t[0] == epoch
    assert times[0] < epoch
