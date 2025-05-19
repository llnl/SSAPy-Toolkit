from .leap_frog import leapfrog
from ..time import get_times, Time
from ssapy import Orbit


def quickint(orbit=None, r0=None, v0=None, t=None):
    
    if orbit is not None:
        if r0 is not None or v0 is not None:
            raise ValueError("Provide either orbit or (r0, v0), not both")
        r0 = orbit.r
        v0 = orbit.v
    else:
        if r0 is None or v0 is None:
            raise ValueError("Must provide either orbit or both r0 and v0")
        orbit = Orbit(r=r0, v=v0, t=t0)
    t0 = Time(orbit.t, format='gps')

    if t is None:
        t = get_times(duration=(orbit.period, 's'), t0=t0)

    r, v = leapfrog(r0, v0, t)
    return r, v, t
