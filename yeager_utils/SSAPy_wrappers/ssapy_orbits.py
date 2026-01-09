import numpy as np
from astropy.time import Time

from ssapy import Orbit, rv
from ssapy.utils import get_times, points_on_circle

from .sat_kwargs import ssapy_kwargs
from .ssapy_props import ssapy_prop

# Uses the current best propagator and acceleration models in ssapy
def ssapy_orbit(
    orbit=None,
    a=None, e=0, i=0, pa=0, raan=0, ta=0,
    r=None, v=None,
    duration=(1, "day"),
    freq=(1, "min"),
    t0="2025-01-01",
    t=None,
    prop=None,
    # optional convenience: if you don't pass `prop`, we can build one
    propkw=None,
    integration_timestep=None,
    # convenience object params (used only if we need to build propkw)
    mass=250.0, area=0.022, CD=2.3, CR=1.3,
):
    """
    Returns (r, v, t) where r,v are arrays over times t.

    IMPORTANT: `prop` defaults to None (do not call ssapy_prop() in the signature).
    If `prop` is None, this function will call ssapy_prop(...) from your module.
    """
    # Build time array
    if t is None:
        t0_time = t0 if isinstance(t0, Time) else Time(t0, scale="utc")
        t = get_times(duration=duration, freq=freq, t0=t0_time)
        t0_time = t[0]
    else:
        t0_time = t[0]

    # Build propagator lazily if not provided
    if prop is None:
        if propkw is None:
            # expects you have ssapy_kwargs in the same module
            propkw = ssapy_kwargs(mass=mass, area=area, CD=CD, CR=CR)

        ode_kwargs = None
        if integration_timestep is not None:
            # SciPy solve_ivp uses max_step (seconds)
            ode_kwargs = {"max_step": float(integration_timestep)}

        # expects you have ssapy_prop in the same module
        prop = ssapy_prop(propkw=propkw, ode_kwargs=ode_kwargs)

    # Initialize orbit
    if orbit is not None:
        print(
            f"ssapy_orbit: Initializing orbit with a pre-defined orbit object: {orbit}.\n"
            f"Integrating: {t[0]} to {t[-1]}"
        )
    elif a is not None:
        print(
            "ssapy_orbit: Initializing orbit with Keplerian elements: "
            f"a={a}, e={e}, i={i}, pa={pa}, raan={raan}, ta={ta}\n"
            f"Integrating: {t[0]} to {t[-1]}"
        )
        kElements = [a, e, i, pa, raan, ta]
        try:
            orbit = Orbit.fromKeplerianElements(*kElements, t=t0_time)
        except TypeError:
            orbit = Orbit.fromKeplerianElements(*kElements, t0_time)
    elif r is not None and v is not None:
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)
        print(
            "ssapy_orbit: Initializing orbit with position (r) and velocity (v) vectors:\n"
            f"r={r},\n"
            f"v={v}\n"
            f"Integrating: {t[0]} to {t[-1]}"
        )
        try:
            orbit = Orbit(r=r, v=v, t=t0_time, propkw=propkw)
        except TypeError:
            orbit = Orbit(r, v, t0_time, propkw=propkw)
    else:
        raise ValueError(
            "ssapy_orbit: Provide either an Orbit, Keplerian elements (a,e,i,pa,raan,ta), "
            "or position/velocity vectors (r,v)."
        )

    # Propagate
    try:
        try:
            r_out, v_out = rv(orbit=orbit, time=t, propagator=prop)
        except TypeError:
            r_out, v_out = rv(orbit, t, prop)
        return r_out, v_out, t
    except (RuntimeError, ValueError) as err:
        print(err)
        return np.nan, np.nan, np.nan


def ssapy_orbit_incremented(
    orbit=None,
    a=None, e=0, i=0, pa=0, raan=0, ta=0,
    r=None, v=None,
    duration=(30, "day"),
    freq=(1, "hr"),
    t0="2025-01-01",
    t=None,
    prop=None,
    propkw=None,
    integration_timestep=None,
    mass=250.0, area=0.022, CD=2.3, CR=1.3,
    plot=False,  # kept for API compatibility; not used here
):
    """
    Incremental propagation (step-by-step): returns (r, v, t) if t was None else (r, v).
    """
    if t is None:
        t0_time = t0 if isinstance(t0, Time) else Time(t0, scale="utc")
        t = get_times(duration=duration, freq=freq, t0=t0_time)
        t0_time = t[0]
        time_is_None = True
    else:
        t0_time = t[0]
        time_is_None = False

    if prop is None:
        if propkw is None:
            propkw = ssapy_kwargs(mass=mass, area=area, CD=CD, CR=CR)
        ode_kwargs = None
        if integration_timestep is not None:
            ode_kwargs = {"max_step": float(integration_timestep)}
        prop = ssapy_prop(propkw=propkw, ode_kwargs=ode_kwargs)

    # Initialize orbit state
    if orbit is not None:
        pass
    elif a is not None:
        kElements = [a, e, i, pa, raan, ta]
        try:
            orbit = Orbit.fromKeplerianElements(*kElements, t=t0_time)
        except TypeError:
            orbit = Orbit.fromKeplerianElements(*kElements, t0_time)
    elif r is not None and v is not None:
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)
        try:
            orbit = Orbit(r=r, v=v, t=t0_time, propkw=propkw)
        except TypeError:
            orbit = Orbit(r, v, t0_time, propkw=propkw)
    else:
        raise ValueError(
            "Either an Orbit, Keplerian elements (a,e,i,pa,raan,ta), or (r,v) must be provided."
        )

    num_steps = len(t)
    r_hist = np.full((num_steps, 3), np.nan, dtype=float)
    v_hist = np.full((num_steps, 3), np.nan, dtype=float)

    r_hist[0] = np.asarray(orbit.r, dtype=float).reshape(3)
    v_hist[0] = np.asarray(orbit.v, dtype=float).reshape(3)

    try:
        for k in range(1, num_steps):
            orbit_k = Orbit(r=r_hist[k - 1], v=v_hist[k - 1], t=t[k - 1], propkw=propkw)

            try:
                r_next, v_next = rv(orbit=orbit_k, time=t[k], propagator=prop)
            except TypeError:
                r_next, v_next = rv(orbit_k, t[k], prop)

            r_hist[k] = np.asarray(r_next, dtype=float).reshape(-1, 3)[-1]
            v_hist[k] = np.asarray(v_next, dtype=float).reshape(-1, 3)[-1]

    except (RuntimeError, ValueError) as err:
        print(f"Error at time step {k}, {t[k]}: {err}")
        if time_is_None:
            return r_hist[:k], v_hist[:k], t[:k]
        return r_hist[:k], v_hist[:k]

    if time_is_None:
        return r_hist, v_hist, t
    return r_hist, v_hist


def get_similar_orbits(
    r0,
    v0,
    rad=1e5,
    num_orbits=4,
    duration=(90, "day"),
    freq=(1, "hour"),
    t0="2025-01-01",
    mass=250.0,
    CD=2.3,
    CR=1.3,
    area=None,
    integration_timestep=10.0,
    prop=None,
):
    """
    Generate similar trajectories by perturbing the initial position on a circle (via ssapy.utils.points_on_circle).

    Returns:
        trajectories: array of shape (n_times, 6, num_orbits) where [:,0:3,:]=r and [:,3:6,:]=v
        t:            times (Astropy Time array)
    """
    r0 = np.asarray(r0, dtype=float).reshape(1, 3)
    v0 = np.asarray(v0, dtype=float).reshape(1, 3)

    if area is None:
        area = float(mass) / 19000.0 + 0.01

    propkw = ssapy_kwargs(mass=mass, area=area, CD=CD, CR=CR)

    traj_list = []
    t_out = None

    for point in points_on_circle(r0, v0, rad=rad, num_points=num_orbits):
        r_hist, v_hist, t = ssapy_orbit(
            r=np.asarray(point, dtype=float).reshape(3),
            v=v0.reshape(3),
            duration=duration,
            freq=freq,
            t0=t0,
            t=None,
            prop=prop,                 # if you pass a propagator, we'll use it
            propkw=propkw,             # only used if prop is None
            integration_timestep=integration_timestep,
        )
        if t_out is None:
            t_out = t
        rv_hist = np.hstack((r_hist, v_hist))  # (n_times, 6)
        traj_list.append(rv_hist)

    trajectories = np.stack(traj_list, axis=2)  # (n_times, 6, num_orbits)
    return trajectories, t_out
