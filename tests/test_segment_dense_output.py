"""Dense output must survive segmented propagation.

``_combine_trajectories`` built its ``SixDOFTrajectory`` without ``solution``,
so ``propagate_spacecraft_segments`` returned ``solution=None`` on every
segmented run even when each segment was propagated with
``dense_output=True``.
"""

import numpy as np
import pytest

from ssapy_toolkit.propagators_6dof.high_accuracy import (
    ImpulseManeuver,
    propagate_spacecraft_segments,
)
from ssapy_toolkit.propagators_6dof.sixdof import (
    Spacecraft,
    _piecewise_solution_sequence,
)

# GPS seconds, 2024-05-13.
GPS_EPOCH = 1_400_000_000.0

R0 = np.array([7.0e6, 0.0, 0.0])
V0 = np.array([0.0, 7.5e3, 0.0])


def _drifting_spacecraft(epoch=GPS_EPOCH):
    return Spacecraft(r=R0, v=V0, t=epoch, inertia=np.eye(3))


def _three_segments(epoch=GPS_EPOCH, dense=(True, True, True)):
    return [
        {
            "times": [epoch + start, epoch + start + 0.5, epoch + start + 1.0],
            "mu": 0.0,
            "dense_output": flag,
        }
        for start, flag in zip((0.0, 1.0, 2.0), dense)
    ]


def test_segmented_propagation_preserves_dense_output():
    trajectory = propagate_spacecraft_segments(
        _drifting_spacecraft(), _three_segments()
    )
    assert trajectory.solution is not None


def test_dense_output_spans_every_segment_boundary():
    """One query crossing both interior boundaries, deliberately unsorted."""
    trajectory = propagate_spacecraft_segments(
        _drifting_spacecraft(), _three_segments()
    )

    offsets = np.array([2.75, 0.25, 1.50, 0.90, 2.10, 1.25])
    state = trajectory.solution(GPS_EPOCH + offsets)

    assert state.shape == (state.shape[0], offsets.size)
    expected = R0[None, :] + offsets[:, None] * V0[None, :]
    # mu=0, so the reference is exact straight-line drift. The position
    # tolerance is set by float64 epoch resolution, not by the dispatcher:
    # one ULP at GPS 1.4e9 is 2.4e-7 s, which at 7.5 km/s is 1.8e-3 m of
    # representable granularity. Measured max error here is 7.2e-4 m.
    np.testing.assert_allclose(state[:3].T, expected, rtol=0.0, atol=1.0e-2)
    # Velocity is constant under mu=0 and comes back exact.
    np.testing.assert_allclose(
        state[3:6].T, np.tile(V0, (offsets.size, 1)), rtol=0.0, atol=1.0e-9
    )


def test_dense_output_accepts_a_scalar_epoch():
    trajectory = propagate_spacecraft_segments(
        _drifting_spacecraft(), _three_segments()
    )
    state = np.asarray(trajectory.solution(GPS_EPOCH + 2.4))
    assert state.ndim == 1
    np.testing.assert_allclose(state[:3], R0 + 2.4 * V0, rtol=0.0, atol=1.0e-2)


def test_boundary_epoch_resolves_to_the_earlier_segment():
    """Pins the ``side="left"`` convention against an impulsive discontinuity.

    ``_piecewise_solution`` sends the split epoch to the earlier segment, so
    the boundary must read as the pre-impulse velocity.
    """
    dv = np.array([0.0, 0.0, 25.0])
    trajectory = propagate_spacecraft_segments(
        _drifting_spacecraft(),
        [
            {"times": [GPS_EPOCH, GPS_EPOCH + 1.0], "mu": 0.0, "dense_output": True},
            {
                "times": [GPS_EPOCH + 1.0, GPS_EPOCH + 2.0],
                "mu": 0.0,
                "dense_output": True,
                "impulses": ImpulseManeuver(dv, frame="inertial"),
            },
        ],
    )

    at_boundary = np.asarray(trajectory.solution(GPS_EPOCH + 1.0))
    just_after = np.asarray(trajectory.solution(GPS_EPOCH + 1.0 + 1.0e-3))

    np.testing.assert_allclose(at_boundary[3:6], V0, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(just_after[3:6], V0 + dv, rtol=0.0, atol=1.0e-9)


def test_dense_output_is_dropped_when_any_segment_lacks_it():
    trajectory = propagate_spacecraft_segments(
        _drifting_spacecraft(), _three_segments(dense=(True, False, True))
    )
    assert trajectory.solution is None


def test_single_segment_returns_the_underlying_solution():
    trajectory = propagate_spacecraft_segments(
        _drifting_spacecraft(),
        [{"times": [GPS_EPOCH, GPS_EPOCH + 1.0], "mu": 0.0, "dense_output": True}],
    )
    assert trajectory.solution is not None
    state = np.asarray(trajectory.solution(GPS_EPOCH + 0.5))
    np.testing.assert_allclose(state[:3], R0 + 0.5 * V0, rtol=0.0, atol=1.0e-2)


def test_dispatcher_rejects_a_mismatched_breakpoint_count():
    identity = [lambda t: np.atleast_2d(np.asarray(t, dtype=float))] * 3
    with pytest.raises(ValueError, match="one epoch per interior segment"):
        _piecewise_solution_sequence(identity, [1.0])


def test_dispatcher_returns_none_on_unsorted_breakpoints():
    identity = [lambda t: np.atleast_2d(np.asarray(t, dtype=float))] * 3
    assert _piecewise_solution_sequence(identity, [2.0, 1.0]) is None


def test_dispatcher_is_flat_not_nested():
    """N segments must cost one dispatch level, not N-1 nested closures."""
    depths = []

    def probe(index):
        def evaluate(t):
            depths.append(index)
            return np.atleast_2d(np.asarray(t, dtype=float))

        return evaluate

    count = 200
    solution = _piecewise_solution_sequence(
        [probe(index) for index in range(count)],
        np.arange(1.0, float(count)),
    )
    # One query inside the last segment must touch exactly one segment.
    solution(np.array([count - 0.5]))
    assert depths == [count - 1]
