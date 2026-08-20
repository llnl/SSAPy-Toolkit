import numpy as np
from ssapy import get_body
from ssapy.body import MoonPosition
from ssapy.utils import normed

from ..time_functions import Time
from .velocity import v_from_r


def gcrf_to_lunar(r: np.ndarray, t: np.ndarray, v: np.ndarray = None) -> np.ndarray:
    """Convert GCRF position/velocity vectors to the rotating lunar frame."""

    class MoonRotator:
        def __init__(self):
            self.mpm = MoonPosition()

        def __call__(self, r: np.ndarray, t: np.ndarray) -> np.ndarray:
            if isinstance(t, Time):
                t = t.gps
            rmoon = self.mpm(t)
            vmoon = (self.mpm(t + 5.0) - self.mpm(t - 5.0)) / 10.0
            xhat = normed(rmoon.T).T
            vpar = np.einsum("ab,ab->b", xhat, vmoon) * xhat
            yhat = normed((vmoon - vpar).T).T
            zhat = np.cross(xhat, yhat, axisa=0, axisb=0).T
            rotation = np.empty((3, 3, len(t)))
            rotation[0] = xhat
            rotation[1] = yhat
            rotation[2] = zhat
            return np.einsum("abc,cb->ca", rotation, r)

    rotator = MoonRotator()
    if v is None:
        return rotator(r, t)
    r_lunar = rotator(r, t)
    return r_lunar, v_from_r(r_lunar, t)


def gcrf_to_lunar_fixed(r: np.ndarray, t: np.ndarray, v: np.ndarray = None) -> np.ndarray:
    """Convert GCRF vectors to lunar-fixed coordinates with the Moon as origin."""
    moon_body = get_body("moon")
    r_lunar = gcrf_to_lunar(r, t) - gcrf_to_lunar(moon_body.position(t).T, t)
    if v is None:
        return r_lunar
    return r_lunar, v_from_r(r_lunar, t)


def get_lunar_rv(t):
    """Return Moon GCRF position and velocity for scalar or vector time input."""
    if isinstance(t, Time):
        t = t.gps
    elif np.size(t) > 1 and isinstance(t[0], Time):
        t = np.array([ti.gps for ti in t], dtype=float)

    moon = get_body("moon")
    r = moon.position(t).T
    if np.size(t) > 1:
        v = v_from_r(r, t)
    else:
        dt = 1.0
        v = (moon.position(t + dt).T - moon.position(t - dt).T) / (2.0 * dt)
    return np.atleast_2d(r), np.atleast_2d(v)
