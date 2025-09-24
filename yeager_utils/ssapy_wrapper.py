# flake8: noqa: E501

from .vectors import points_on_circle
from .Time_Functions import get_times
from astropy.time import Time
from ssapy import Orbit, rv
from ssapy.accel import AccelKepler, AccelSolRad, AccelEarthRad, AccelDrag
from ssapy.body import get_body
from ssapy.gravity import AccelHarmonic, AccelThirdBody
from ssapy.propagator import SciPyPropagator

import numpy as np


def keplerian_prop():
    return SciPyPropagator(AccelKepler())


accel_3_cache = None
def threebody_prop():
    global accel_3_cache
    if accel_3_cache is None:
        accel_3_cache = AccelKepler() + AccelThirdBody(get_body("moon"))
    return SciPyPropagator(accel_3_cache)


accel_4_cache = None
def fourbody_prop():
    global accel_4_cache
    if accel_4_cache is None:
        accel_4_cache = AccelKepler() + AccelThirdBody(get_body("moon")) + AccelThirdBody(get_body("Sun"))
    return SciPyPropagator(accel_4_cache)


accel_best_cache = None
def best_prop(kwargs=dict(mass=250, area=.022, CD=2.3, CR=1.3)):
    global accel_best_cache
    if accel_best_cache is None:
        aEarth = AccelKepler() + AccelHarmonic(get_body("Earth", model="EGM2008"), 140, 140)
        aMoon = AccelThirdBody(get_body("moon")) + AccelHarmonic(get_body("moon"), 20, 20)
        aSun = AccelThirdBody(get_body("Sun"))
        aMercury = AccelThirdBody(get_body("Mercury"))
        aVenus = AccelThirdBody(get_body("Venus"))
        aMars = AccelThirdBody(get_body("Mars"))
        aJupiter = AccelThirdBody(get_body("Jupiter"))
        aSaturn = AccelThirdBody(get_body("Saturn"))
        aUranus = AccelThirdBody(get_body("Uranus"))
        aNeptune = AccelThirdBody(get_body("Neptune"))
        nonConservative = AccelSolRad(**kwargs) + AccelEarthRad(**kwargs) + AccelDrag(**kwargs)
        planets = aMercury + aVenus + aMars + aJupiter + aSaturn + aUranus + aNeptune
        accel_best_cache = aEarth + aMoon + aSun + planets + nonConservative
    return SciPyPropagator(accel_best_cache)


def ssapy_kwargs(mass=250, area=0.022, CD=2.3, CR=1.3):
    # Asteroid parameters
    kwargs = dict(
        mass=mass,  # [kg]
        area=area,  # [m^2]
        CD=CD,  # Drag coefficient
        CR=CR,  # Radiation pressure coefficient
    )
    return kwargs


def ssapy_prop(propkw=ssapy_kwargs()):
    # Accelerations - pass a body object or string of body name.
    moon = get_body("moon")
    sun = get_body("Sun")
    Mercury = get_body("Mercury")
    Venus = get_body("Venus")
    Earth = get_body("Earth", model="EGM2008")
    Mars = get_body("Mars")
    Jupiter = get_body("Jupiter")
    Saturn = get_body("Saturn")
    Uranus = get_body("Uranus")
    Neptune = get_body("Neptune")
    aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)
    aSun = AccelThirdBody(sun)
    aMoon = AccelThirdBody(moon) + AccelHarmonic(moon, 20, 20)
    aSolRad = AccelSolRad(**propkw)
    aEarthRad = AccelEarthRad(**propkw)
    accel = aEarth + aMoon + aSun + aSolRad + aEarthRad
    # Build propagator
    prop = SciPyPropagator(accel)
    return prop


# Uses the current best propagator and acceleration models in ssapy
def ssapy_orbit(orbit=None, a=None, e=0, i=0, pa=0, raan=0, ta=0, r=None, v=None, duration=(1, 'day'), freq=(1, 'min'), t0="2025-01-01", t=None, prop=ssapy_prop()):
    # Everything is in SI units, except time.
    # density #kg/m^3 --> density
    t0 = Time(t0, scale='utc')
    if t is None:
        time_is_None = True
        t = get_times(duration=duration, freq=freq, t0=t0)
    else:
        t0 = t[0]
        time_is_None = False

    if orbit is not None:
        print(f"ssapy_orbit: Initializing orbit with a pre-defined orbit object: {orbit}.\nIntegrating: {t[0]} to {t[-1]}")
    elif a is not None:
        print(f"ssapy_orbit: Initializing orbit with Keplerian elements: a={a}, e={e}, i={i}, pa={pa}, raan={raan}, ta={ta}\nIntegrating: {t[0]} to {t[-1]}")
        kElements = [a, e, i, pa, raan, ta]
        orbit = Orbit.fromKeplerianElements(*kElements, t=t0)
    elif r is not None and v is not None:
        print(f"ssapy_orbit: Initializing orbit with position (r) and velocity (v) vectors:\nr={r},\nv={v}\nIntegrating: {t[0]} to {t[-1]}")
        orbit = Orbit(r=r, v=v, t=t0)
    else:
        raise ValueError("ssapy_orbit: Either Keplerian elements (a, e, i, pa, raan, ta) or position and velocity vectors (r, v) must be provided.")

    try:
        r, v = rv(orbit=orbit, time=t, propagator=prop)
        if time_is_None:
            return r, v, t
        else:
            return r, v
    except (RuntimeError, ValueError) as err:
        print(err)
        return np.nan, np.nan, np.nan


def ssapy_orbit_incremented(
    orbit=None, a=None, e=0, i=0, pa=0, raan=0, ta=0, r=None, v=None,
    duration=(30, 'day'), freq=(1, 'hr'), t0="2025-01-01", t=None,
    prop=ssapy_prop(), propkw=ssapy_kwargs(), plot=False
):
    """
    Computes position and velocity vectors for an orbit over specified time steps.
    
    Parameters:
        orbit: Orbit object (optional). If provided, other orbital parameters are ignored.
        a, e, i, pa, raan, ta: Keplerian orbital elements (optional).
        r, v: Initial position and velocity vectors (optional).
        duration: Tuple specifying duration and unit (e.g., (30, 'day')).
        freq: Tuple specifying frequency and unit (e.g., (1, 'hr')).
        t0: Start time as a string (default: "2025-01-01").
        t: Array of time steps (optional).
        prop: SSAPy propagation object, contains the accelerations and propagator. (default: ssapy_prop()).
        propkw: Properties of the orbiting object stored in a dictionary. mass, area, CR, CD (default: ssapy_prop()).

    Returns:
        r: Array of position vectors.
        v: Array of velocity vectors.
        t: Array of time steps (if time steps were not provided initially).
    """
    t0 = Time(t0, scale='utc')
    if t is None:
        time_is_None = True
        t = get_times(duration=duration, freq=freq, t0=t0)
    else:
        t0 = t[0]
        time_is_None = False

    if orbit is not None:
        pass
    elif a is not None:
        kElements = [a, e, i, pa, raan, ta]
        orbit = Orbit.fromKeplerianElements(*kElements, t0)
    elif r is not None and v is not None:
        orbit = Orbit(r, v, t0)
    else:
        raise ValueError("Either Keplerian elements (a, e, i, pa, raan, ta) or position and velocity vectors (r, v) must be provided.")

    num_steps = len(t)
    r = np.full((num_steps, 3), np.nan)
    v = np.full((num_steps, 3), np.nan)
    r[0] = orbit.r
    v[0] = orbit.v
    try:
        for temp_i in range(1, num_steps):
            orbit = Orbit(r=r[temp_i - 1], v=v[temp_i - 1], t=t[temp_i - 1], propkw=propkw)
            r_next, v_next = rv(orbit, t[temp_i], propagator=prop)
            r[temp_i] = r_next
            v[temp_i] = v_next
    except (RuntimeError, ValueError) as err:
        print(f"Error at time step {temp_i}, {t[temp_i]}: {err}")
        if time_is_None:
            return r[:temp_i], v[:temp_i], t[:temp_i]
        else:
            return r[:temp_i], v[:temp_i]

    if time_is_None:
        return r, v, t
    else:
        return r, v


# Generate orbits near stable orbit.
def get_similar_orbits(r0, v0, rad=1e5, num_orbits=4, duration=(90, 'days'), freq=(1, 'hour'), t0="2025-1-1", mass=250):
    r0 = np.reshape(r0, (1, 3))
    v0 = np.reshape(v0, (1, 3))
    print(r0, v0)
    for idx, point in enumerate(points_on_circle(r0, v0, rad=rad, num_points=num_orbits)):
        # Calculate entire satellite trajectory
        r, v = ssapy_orbit(r=point, v=v0, duration=duration, freq=freq, t0=t0, integration_timestep=10, mass=mass, area=mass / 19000 + 0.01, CD=2.3, CR=1.3)
        if idx == 0:
            trajectories = np.concatenate((r0, v0), axis=1)[:len(r)]
        rv = np.concatenate((r, v), axis=1)
        trajectories = np.dstack((trajectories, rv))
    return trajectories


def lyapunov_exponent(r, v, duration, freq, t0, perturbation, time_between_data=1, lyapunov_type='perturbation'):
    """
    Calculate the Lyapunov exponent for a cislunar orbit.

    Parameters:
    - r: An (n,3) array of positions [x, y, z].
    - v: An (n,3) array of velocities [vx, vy, vz].
    - duration: tuple(time, unit) time with given unit to integrate the perturbed orbit.
    - freq: tuple(time, unit) time with given unit to output statevector.
    - t0: str: sets the position of the Solar System for integration.
    - perturbation: float: Small perturbation applied to the initial position and velocity.
    - time_between_data: float: the amount of time with given unit between statevectors, default is 1 in units of duration.

    Returns:
    - The Lyapunov exponent for the cislunar orbit.
    """
    num_states = len(r)
    pr, pv = ssapy_rv_from_rv(r[0] + np.random.randn(3) * perturbation, v[0] + np.random.randn(3) * perturbation, duration=duration, freq=freq, t0=t0, integration_timestep=10, mass=250, area=0.022, CD=2.3, CR=1.3)
    if lyapunov_type == 'perturbation':
        delta_states = np.linalg.norm(pr - r, axis=1)
        time_series_le = np.log(delta_states / perturbation) * 1 / time_between_data
        lyapunov_exponent = np.sum(time_series_le) / num_states
    if lyapunov_type == 'derivative':
        delta_states = np.linalg.norm(pr - r, axis=1)
        time_series_le = np.log(delta_states[1:] / delta_states[:-1]) * 1 / time_between_data
        lyapunov_exponent = np.sum(time_series_le) / num_states
    return lyapunov_exponent, time_series_le
