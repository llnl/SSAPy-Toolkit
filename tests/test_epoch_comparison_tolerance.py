"""Epoch comparisons must not carry a scale-dependent relative tolerance.

The 6-DoF propagators compare absolute epochs in three places: the terminal
event de-duplication in ``propagate_6dof``, the segment-start check in
``_propagate_spacecraft_segment``, and the completeness check in
``_require_complete_segment``. All three used ``np.isclose`` defaults, whose
``rtol=1e-5`` is a tolerance of hours once the epochs are absolute GPS
seconds.
"""

import numpy as np
import pytest

from ssapy_toolkit.propagators_6dof.high_accuracy import (
    propagate_spacecraft_segments,
)
from ssapy_toolkit.propagators_6dof.sixdof import Spacecraft, _epochs_close

# GPS seconds. 1.4e9 is 2024-05-13; 3.8e9 is 2100-06-20.
GPS_2024 = 1_400_000_000.0
GPS_2100 = 3_800_000_000.0

EPOCHS = pytest.mark.parametrize("epoch", [0.0, GPS_2024, GPS_2100])


@EPOCHS
def test_identical_epochs_compare_equal(epoch):
    assert _epochs_close(epoch, epoch)


@EPOCHS
def test_hour_scale_mismatch_is_rejected(epoch):
    """np.isclose defaults accepted this at every epoch above the origin."""
    assert not _epochs_close(epoch, epoch + 3_600.0)


@EPOCHS
def test_millisecond_mismatch_is_rejected(epoch):
    assert not _epochs_close(epoch, epoch + 1.0e-3)


@EPOCHS
def test_float64_round_off_is_absorbed(epoch):
    """A few ULP of round-off must still read as the same epoch.

    This is the case a flat ``atol=1e-6`` gets wrong. 4 ULP is 9.5e-7 s at
    GPS 1.4e9 but 1.9e-6 s at GPS 3.8e9, so the fixed floor alone rejects a
    difference float64 cannot even represent as meaningful.
    """
    nudged = epoch + 4.0 * np.spacing(max(abs(epoch), 1.0))
    assert _epochs_close(epoch, nudged)


def test_ulp_term_is_what_makes_year_2100_work():
    """Pin the reason the ULP term exists, not just its effect."""
    nudged = GPS_2100 + 4.0 * np.spacing(GPS_2100)
    assert not np.isclose(GPS_2100, nudged, rtol=0.0, atol=1.0e-6)
    assert _epochs_close(GPS_2100, nudged)


@EPOCHS
def test_segment_start_rejects_hour_scale_epoch_gap(epoch):
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[1.0, 0.0, 0.0],
        t=epoch,
        inertia=np.eye(3),
    )
    with pytest.raises(ValueError, match="current spacecraft epoch"):
        propagate_spacecraft_segments(
            spacecraft,
            [{"times": [epoch + 3_600.0, epoch + 3_601.0], "mu": 0.0}],
        )


@EPOCHS
def test_segment_start_accepts_round_off_epoch_gap(epoch):
    """A segment handed the previous trajectory's final epoch must still run."""
    start = epoch + 4.0 * np.spacing(max(abs(epoch), 1.0))
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[1.0, 0.0, 0.0],
        t=epoch,
        inertia=np.eye(3),
    )
    trajectory = propagate_spacecraft_segments(
        spacecraft,
        [{"times": [start, start + 1.0], "mu": 0.0}],
    )
    assert trajectory.t[-1] == pytest.approx(start + 1.0, rel=0.0, abs=1.0e-6)
    # The nudge is a real (if tiny) head start, so at 1 m/s the drift picks up
    # the same few ULP of extra path. 1e-5 m sits above that and well below
    # any propagation error worth catching here.
    np.testing.assert_allclose(trajectory.r[-1], [1.0, 0.0, 0.0], atol=1.0e-5)
