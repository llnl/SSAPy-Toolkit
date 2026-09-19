"""Terminal-endpoint epoch regressions at absolute GPS epochs.

When a terminal event root lands within floating-point resolution of a
requested sample, the endpoint must collapse onto the stopping epoch and
state rather than being appended as a separate sample. An exact ``==``
comparison treats a root SciPy resolved as well as it can as a distinct
epoch, which at GPS 1.4e9 leaves two samples 2.4e-7 s apart and a near-zero
step in anything that differences the epoch series.
"""

import numpy as np
import pytest

from ssapy_toolkit.propagators_6dof import propagate_6dof

BASE = dict(
    r0=[7.0e6, 0.0, 0.0],
    v0=[0.0, 7546.0, 0.0],
    q0=[1.0, 0.0, 0.0, 0.0],
    omega0=[0.0, 0.0, 0.1],
    mu=0.0,
    inertia=np.diag([4.0, 5.0, 6.0]),
)


def _terminal_at(root):
    def event(t, y):
        return t - root

    event.terminal = True
    return event


@pytest.mark.parametrize("epoch", [0.0, 1.4e9, 3.8e9], ids=["t0", "gps2024", "gps2100"])
def test_root_one_ulp_past_a_sample_collapses_onto_the_stopping_epoch(epoch):
    interior = epoch + 5.0
    root = np.nextafter(interior, np.inf)
    trajectory = propagate_6dof(
        times=np.array([epoch, interior, epoch + 10.0]), t0=epoch,
        events=(_terminal_at(root),), rtol=1e-12, atol=1e-14, **BASE)

    times = np.asarray(trajectory.t, dtype=float)
    assert trajectory.status == 1
    # Three requested samples, termination at the second: two epochs survive.
    assert times.size == 2
    assert float(times[1]) - float(times[0]) > 1.0
    # The retained endpoint is the stopping epoch, not the requested sample.
    assert times[-1] == pytest.approx(root, abs=np.spacing(interior))
    assert np.all(np.diff(times) > 1.0e-6)


@pytest.mark.parametrize("epoch", [0.0, 1.4e9, 3.8e9], ids=["t0", "gps2024", "gps2100"])
def test_root_exactly_on_a_sample_is_not_duplicated(epoch):
    interior = epoch + 5.0
    trajectory = propagate_6dof(
        times=np.array([epoch, interior, epoch + 10.0]), t0=epoch,
        events=(_terminal_at(interior),), rtol=1e-12, atol=1e-14, **BASE)

    times = np.asarray(trajectory.t, dtype=float)
    assert times.size == 2
    assert times[-1] == pytest.approx(interior, abs=1.0e-6)


@pytest.mark.parametrize("epoch", [0.0, 1.4e9], ids=["t0", "gps2024"])
def test_root_between_samples_is_appended(epoch):
    root = epoch + 5.0
    trajectory = propagate_6dof(
        times=np.array([epoch, epoch + 2.0, epoch + 10.0]), t0=epoch,
        events=(_terminal_at(root),), rtol=1e-12, atol=1e-14, **BASE)

    times = np.asarray(trajectory.t, dtype=float)
    # Samples at +0 and +2 survive, and the stopping epoch is appended.
    assert times.size == 3
    assert times[-1] == pytest.approx(root, abs=1.0e-6)
    assert np.all(np.diff(times) > 1.0e-6)


def test_earlier_nonterminal_event_does_not_become_the_endpoint():
    def early(t, y):
        return t - 2.0

    early.terminal = False

    trajectory = propagate_6dof(
        times=np.linspace(0.0, 20.0, 21), t0=0.0,
        events=(early, _terminal_at(5.0)), rtol=1e-11, atol=1e-13, **BASE)

    assert trajectory.status == 1
    assert float(trajectory.t[-1]) == pytest.approx(5.0, abs=1.0e-6)


def test_requested_sample_just_before_the_root_survives_at_small_epochs():
    # 5e-7 s before the root is 2.3e9 ULP at t = 1.25 s, so the requested
    # sample and the stopping epoch are distinct and both must be retained.
    # Deduping on the 1 us epoch floor instead of float resolution would
    # discard the requested sample here.
    trajectory = propagate_6dof(
        times=np.array([0.0, 1.25 - 5.0e-7, 2.0]), t0=0.0,
        events=(_terminal_at(1.25),), rtol=1e-12, atol=1e-14, **BASE)

    times = np.asarray(trajectory.t, dtype=float)
    assert times.size == 3
    assert times[-1] == pytest.approx(1.25, abs=1.0e-9)
    assert times[-1] - times[-2] == pytest.approx(5.0e-7, rel=1e-6)
