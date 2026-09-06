"""Absolute-epoch handling in the fixed-step orbit propagators.

``rk4`` and ``leapfrog`` previously collapsed the time grid to seconds since
``t[0]`` and handed that to the force evaluation. ``accel_point_moon`` and
``accel_point_sun`` build an ``astropy.time.Time`` in the ``gps`` format from
whatever they are given, so an elapsed-seconds argument resolved the ephemeris
near 1980-01-06 rather than at the requested epoch.
"""

import importlib

import numpy as np
import pytest

from ssapy_toolkit.propagators_orbit.leap_frog import leapfrog
from ssapy_toolkit.propagators_orbit.rk4 import rk4

# 2024-05-13T16:53:02 UTC. Any epoch far from the GPS origin exposes the bug.
GPS_EPOCH = 1_400_000_000.0

R0 = [7.0e6, 0.0, 0.0]
V0 = [0.0, 7.5e3, 0.0]


def _zero_third_bodies(monkeypatch, module):
    """Silence any ephemeris terms so only the thrust profile varies.

    ``leapfrog`` carries no built-in third-body models, so the names are
    absent there and nothing needs patching.
    """
    for name in ("accel_point_moon", "accel_point_sun"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, lambda _r, _t: np.zeros(3))


def test_rk4_passes_absolute_epoch_to_third_body_models(monkeypatch):
    module = importlib.import_module("ssapy_toolkit.propagators_orbit.rk4")
    moon_epochs: list[float] = []
    sun_epochs: list[float] = []

    def moon(_r, time):
        moon_epochs.append(float(time))
        return np.zeros(3)

    def sun(_r, time):
        sun_epochs.append(float(time))
        return np.zeros(3)

    monkeypatch.setattr(module, "accel_point_moon", moon)
    monkeypatch.setattr(module, "accel_point_sun", sun)

    rk4(
        R0,
        V0,
        np.array([GPS_EPOCH, GPS_EPOCH + 10.0]),
        accel_gravity=lambda _r: np.zeros(3),
    )

    # Before the fix these were 0.0 and 10.0 -- GPS seconds, but measured from
    # 1980-01-06 instead of the requested epoch.
    for stage_epochs in (moon_epochs, sun_epochs):
        assert len(stage_epochs) == 4  # k1, k2, k3, k4 of the single step
        assert min(stage_epochs) == pytest.approx(GPS_EPOCH, rel=0.0, abs=1.0e-6)
        assert max(stage_epochs) == pytest.approx(
            GPS_EPOCH + 10.0, rel=0.0, abs=1.0e-6
        )


def test_leapfrog_passes_absolute_epoch_to_extra_accelerations():
    seen: list[float] = []

    def probe(_r, _v, time):
        seen.append(float(time))
        return np.zeros(3)

    leapfrog(R0, V0, np.array([GPS_EPOCH, GPS_EPOCH + 1.0]), accels=probe)

    assert seen
    assert min(seen) == pytest.approx(GPS_EPOCH, rel=0.0, abs=1.0e-6)
    assert max(seen) == pytest.approx(GPS_EPOCH + 1.0, rel=0.0, abs=1.0e-6)


@pytest.mark.parametrize("propagate", [rk4, leapfrog])
def test_thrust_profiles_stay_keyed_on_elapsed_seconds(monkeypatch, propagate):
    """A burn specified in elapsed seconds must not move with the epoch.

    ``build_profile`` semantics are deliberately untouched by this change, so
    an identical grid offset to an absolute epoch must reproduce the relative
    run bit for bit.
    """
    _zero_third_bodies(monkeypatch, importlib.import_module(propagate.__module__))

    elapsed = np.array([0.0, 60.0, 120.0])
    burn = (60.0, 1.0e-3)  # (start [s since t[0]], radial acceleration [m/s^2])

    r_relative, v_relative = propagate(R0, V0, elapsed, radial=burn)
    r_absolute, v_absolute = propagate(R0, V0, GPS_EPOCH + elapsed, radial=burn)

    np.testing.assert_array_equal(r_relative, r_absolute)
    np.testing.assert_array_equal(v_relative, v_absolute)

    # Guard against a vacuous comparison: the burn must actually do something.
    # Measured final-sample displacement is 1.80 m (rk4) and 3.60 m (leapfrog);
    # the floor is set well below both.
    r_unburned, _ = propagate(R0, V0, GPS_EPOCH + elapsed, radial=None)
    assert np.linalg.norm(r_absolute[-1] - r_unburned[-1]) > 1.0
