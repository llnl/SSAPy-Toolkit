from .leap_frog import leapfrog
from ..constants import EARTH_MU
from ..time import get_times, Time
from ssapy import Orbit
import numpy as np


def quickint(orbit=None, r0=None, v0=None, t=None):
    orbit_given_as_object = False

    # Case 1: orbit is Orbit instance (normal usage)
    if orbit is not None and isinstance(orbit, Orbit):
        orbit_given_as_object = True
        if r0 is not None or v0 is not None:
            raise ValueError("Provide either orbit or (r0, v0), not both")
        r0 = orbit.r
        v0 = orbit.v
        t0 = Time(orbit.t, format='gps')

    # Case 2: orbit and r0 both floats/arrays (treat as r0 and v0)
    elif _is_position_like(orbit) and _is_velocity_like(r0):
        if v0 is not None:
            raise ValueError("Too many positional arguments: expected orbit or (r0, v0)")
        r0, v0 = np.array(orbit, dtype=float), np.array(r0, dtype=float)
        orbit = None
        orbit_given_as_object = False
        t0 = Time(0, format='gps')

    else:
        # If orbit is given but is not Orbit instance, treat it as r0
        if orbit is not None:
            if r0 is not None:
                raise ValueError("Provide either orbit or r0, not both")
            r0 = orbit
            orbit = None
        orbit_given_as_object = False

        if r0 is not None and not isinstance(r0, np.ndarray):
            r0 = np.array(r0, dtype=float)

        if v0 is None and r0 is not None:
            r_mag = np.linalg.norm(r0)
            if r_mag == 0:
                raise ValueError("r0 cannot be zero vector for circular velocity calculation")

            v_circ = np.sqrt(EARTH_MU / r_mag)

            z_hat = np.array([0, 0, 1])
            if np.allclose(r0 / r_mag, z_hat) or np.allclose(r0 / r_mag, -z_hat):
                perp_dir = np.cross(r0, np.array([1, 0, 0]))
            else:
                perp_dir = np.cross(r0, z_hat)
            perp_dir /= np.linalg.norm(perp_dir)

            v0 = v_circ * perp_dir

        elif v0 is None:
            raise ValueError("Must provide either orbit or r0 with optional v0")

    # Handle time input
    valid_time = False
    if t is not None:
        # Accept astropy Time instance or list/array of Times
        if isinstance(t, Time):
            valid_time = True
        elif isinstance(t, (list, np.ndarray)):
            if all(isinstance(tt, Time) for tt in t):
                valid_time = True

    if not valid_time:
        t0 = Time(0, format='gps')
        # Create orbit if not originally given as Orbit object
        if not orbit_given_as_object:
            orbit = Orbit(r=r0, v=v0, t=t0)
        t = get_times(duration=(orbit.period, 's'), t0=t0)
    else:
        # If t was valid time input, set t0 from first time entry or orbit.t if available
        t0 = Time(t[0], format='gps') if isinstance(t, (list, np.ndarray)) else Time(t, format='gps')

    r, v = leapfrog(r0, v0, t)
    return r, v, t


def _is_position_like(x):
    if x is None:
        return False
    if isinstance(x, (float, int)):
        return True
    try:
        arr = np.array(x, dtype=float)
        return arr.ndim == 1 and arr.size in (2, 3)
    except Exception:
        return False


def _is_velocity_like(x):
    return _is_position_like(x)
