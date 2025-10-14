import ssapy
from ssapy.accel import AccelKepler, AccelEarthRad, AccelSolRad, AccelDrag, AccelConstNTW
from ssapy.gravity import AccelHarmonic, AccelThirdBody
from ssapy.body import get_body
from ssapy.constants import RGEO
from ssapy.propagator import RK78Propagator, SciPyPropagator
from ssapy.plotUtils import orbit_plot
from ssapy.utils import get_times
from ssapy import Time
from yeager_utils import calc_gamma_and_heading

t0 = Time("2025-1-1T12:00:00.000", scale='utc')
times = get_times(duration=(12, 'hour'), freq=(1, 'minute'), t=t0).gps

kwargs = dict(
    mass=250,  # [kg] --> was 1e4
    area=0.25,  # [m^2]
    CD=2.3,  # Drag coefficient
    CR=1.3,  # Radiation pressure coefficient
)

earth = get_body("earth", model='egm2008')
moon = get_body("moon")
sun = get_body("sun")

# Accelerations - pass a body object or string of body name.
aEarth = AccelKepler() + AccelHarmonic(earth, 180, 180)
aMoon = AccelThirdBody(moon) + AccelHarmonic(moon)
aSun = AccelThirdBody(sun)
aSolRad = AccelSolRad()
aEarthRad = AccelEarthRad()
aDrag = AccelDrag()
accel = aEarth + aMoon + aSun + aSolRad + aEarthRad + aDrag

a = RGEO
e = 0
i = 0
pa = 0
raan = 0
trueAnomaly = 0

kElements = [a, e, i, pa, raan, trueAnomaly]

orbit = ssapy.Orbit.fromKeplerianElements(*kElements, t0, propkw=kwargs)
orbital_period = orbit.period / (60 * 60 * 24)  # in days

# Define the time indices for the maneuver in the times array
t1_index = 0  # Replace with the actual index for the start time
t2_index = 10  # Replace with the actual index for the end time
print(f"Maneuver window: {Time(times[t1_index], format='gps').ymdhms} to {Time(times[t2_index], format='gps').ymdhms}")

# Example acceleration in NTW coordinates for the maneuver
# Let's say we want to apply an acceleration of 0.01 m/s² in the tangential direction
accel_maneuver = AccelConstNTW(accelntw=[0.0, -1.0, 0.0], time_breakpoints=[times[t1_index], times[t2_index]])
accel = accel + accel_maneuver

try:
    r, v = ssapy.rv(orbit, times, propagator=SciPyPropagator(accel))
except RuntimeError as err:
    print(err)


orbit_plot(r, times)
gamma, heading = calc_gamma_and_heading(r, times)
