from .constants import RGEO
from .coordinates import points_on_circle
from .time import get_times
from astropy.time import Time
from ssapy import Orbit, rv
from ssapy.accel import AccelKepler, AccelSolRad, AccelEarthRad, AccelDrag
from ssapy.body import get_body
from ssapy.gravity import AccelHarmonic, AccelThirdBody
from ssapy.propagator import RK78Propagator

import numpy as np


def ssapy_best_prop(integration_timestep=60):
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
    aEarth = AccelKepler() + AccelHarmonic(Earth, 180, 180)
    aMoon = AccelThirdBody(moon) + AccelHarmonic(moon)
    aSun = AccelThirdBody(sun)
    aMercury = AccelThirdBody(Mercury)
    aVenus = AccelThirdBody(Venus)
    aMars = AccelThirdBody(Mars)
    aJupiter = AccelThirdBody(Jupiter)
    aSaturn = AccelThirdBody(Saturn)
    aUranus = AccelThirdBody(Uranus)
    aNeptune = AccelThirdBody(Neptune)
    aSolRad = AccelSolRad()
    aEarthRad = AccelEarthRad()
    aDrag = AccelDrag()
    planets = aMercury + aVenus + aMars + aJupiter + aSaturn + aUranus + aNeptune
    accel = aEarth + aMoon + aSun + aSolRad + aEarthRad + aDrag + planets
    # Build propagator
    prop = RK78Propagator(accel, h=integration_timestep)
    return prop


def ssapy_kwargs(mass=250, area=0.022, CD=2.3, CR=1.3):
    # Asteroid parameters
    kwargs = dict(
        mass=mass,  # [kg]
        area=area,  # [m^2]
        CD=CD,  # Drag coefficient
        CR=CR,  # Radiation pressure coefficient
    )
    return kwargs


# Uses the current best propagator and acceleration models in ssapy
def ssapy_rv_from_koe(a=RGEO, e=0, inc=0, pa=0, raan=0, trueAnomaly=0, duration=(30, 'day'), freq=(1, 'hr'), start_date="2025-01-01", times=None, integration_timestep=10, mass=250, area=0.022, CD=2.3, CR=1.3):
    # Everything is in SI units, except time.
    # density #kg/m^3 --> density
    # Time span of integration #Only takes integer number of days. Unless providing your own time object.
    t = Time(start_date, scale='utc')
    if times is None:
        times = get_times(duration=duration, freq=freq, t=t)

    kwargs = ssapy_kwargs(mass, area, CD, CR)
    kElements = [a, e, inc, pa, raan, trueAnomaly]  # (a, e, i, pa, raan, trueAnomaly, t, mu)
    orbit = Orbit.fromKeplerianElements(*kElements, t, propkw=kwargs)
    # print(f'Initial orbit: a: {a:.3e} m, e: {e:.2f}, i: {inc:.2f}, pa: {pa:.2f}, raan: {raan:.2f}, trueAnomaly: {trueAnomaly:.2f}, period: {orbit.period/units.day_to_second:.2f} days.')

    prop = ssapy_best_prop(integration_timestep)

    # Calculate entire satellite trajectory
    try:
        r, v = rv(orbit, times, prop)
        return r, v
    except (RuntimeError, ValueError) as err:
        print(err)
        return np.nan, np.nan


def ssapy_rv_from_rv(r=RGEO, v=False, duration=(30, 'day'), freq=(1, 'hr'), start_date="2025-01-01", times=None, integration_timestep=10, mass=250, area=0.022, CD=2.3, CR=1.3):
    # Everything is in SI units, except time.
    # density #kg/m^3 --> density

    # Time span of integration #Only takes integer number of days. Unless providing your own time object.
    t = Time(start_date, scale='utc')
    if times is None:
        times = get_times(duration=duration, freq=freq, t=t)

    kwargs = ssapy_kwargs(mass, area, CD, CR)
    orbit = Orbit(r, v, t, propkw=kwargs)

    prop = ssapy_best_prop(integration_timestep)
    # Calculate entire satellite trajectory
    try:
        r, v = rv(orbit, times, prop)
        return r, v
    except (RuntimeError, ValueError) as err:
        print(err)
        r, v = np.nan, np.nan
    return r, v


# Generate orbits near stable orbit.
def get_similar_orbits(r0, v0, rad=1e5, num_orbits=4, duration=(90, 'days'), freq=(1, 'hour'), start_date="2025-1-1", mass=250):
    r0 = np.reshape(r0, (1, 3))
    v0 = np.reshape(v0, (1, 3))
    print(r0, v0)
    for idx, point in enumerate(points_on_circle(r0, v0, rad=rad, num_points=num_orbits)):
        # Calculate entire satellite trajectory
        r, v = ssapy_rv_from_rv(r=point, v=v0, duration=duration, freq=freq, start_date=start_date, integration_timestep=10, mass=mass, area=mass / 19000 + 0.01, CD=2.3, CR=1.3)
        if idx == 0:
            trajectories = np.concatenate((r0, v0), axis=1)[:len(r)]
        rv = np.concatenate((r, v), axis=1)
        trajectories = np.dstack((trajectories, rv))
    return trajectories


def lyapunov_exponent(r, v, duration, freq, start_date, perturbation, time_between_data=1, lyapunov_type='perturbation'):
    """
    Calculate the Lyapunov exponent for a cislunar orbit.

    Parameters:
    - r: An (n,3) array of positions [x, y, z].
    - v: An (n,3) array of velocities [vx, vy, vz].
    - duration: tuple(time, unit) time with given unit to integrate the perturbed orbit.
    - freq: tuple(time, unit) time with given unit to output statevector.
    - start_date: str: sets the position of the Solar System for integration.
    - perturbation: float: Small perturbation applied to the initial position and velocity.
    - time_between_data: float: the amount of time with given unit between statevectors, default is 1 in units of duration.

    Returns:
    - The Lyapunov exponent for the cislunar orbit.
    """
    num_states = len(r)
    pr, pv = ssapy_rv_from_rv(r[0] + np.random.randn(3) * perturbation, v[0] + np.random.randn(3) * perturbation, duration=duration, freq=freq, start_date=start_date, integration_timestep=10, mass=250, area=0.022, CD=2.3, CR=1.3)
    if lyapunov_type == 'perturbation':
        delta_states = np.linalg.norm(pr - r, axis=1)
        time_series_le = np.log(delta_states / perturbation) * 1 / time_between_data
        lyapunov_exponent = np.sum(time_series_le) / num_states
    if lyapunov_type == 'derivative':
        delta_states = np.linalg.norm(pr - r, axis=1)
        time_series_le = np.log(delta_states[1:] / delta_states[:-1]) * 1 / time_between_data
        lyapunov_exponent = np.sum(time_series_le) / num_states
    return lyapunov_exponent, time_series_le
