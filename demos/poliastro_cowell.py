from astropy import units as u
import numpy as np
from poliastro.bodies import Earth
from poliastro.core.propagation import func_twobody
from poliastro.examples import iss
from poliastro.plotting import OrbitPlotter3D
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator
from poliastro.twobody.sampling import EpochsArray
from poliastro.util import norm
import plotly.io as pio

pio.renderers.default = "plotly_mimetype+notebook_connected"

ACCEL = 2e-5


def constant_accel_factory(accel):
    """Returns a function computing a constant acceleration."""
    def constant_accel(t0, u, k):
        v = u[3:]
        norm_v = np.linalg.norm(v)
        return accel * v / norm_v

    return constant_accel


accel_func = constant_accel_factory(ACCEL)


def f(t0, state, k):
    """Computes the acceleration including perturbations."""
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = accel_func(t0, state, k)
    du_ad = np.array([0, 0, 0, ax, ay, az])
    return du_kep + du_ad


times = np.linspace(0, 10 * iss.period, 500)
ephem = iss.to_ephem(
    EpochsArray(iss.epoch + times, method=CowellPropagator(rtol=1e-11, f=f))
)

frame = OrbitPlotter3D()
frame.set_attractor(Earth)
frame.plot_ephem(ephem, label="ISS")
frame.show()


def state_to_vector(ss):
    """Converts orbital state to a numpy array."""
    r, v = ss.rv()
    x, y, z = r.to_value(u.km)
    vx, vy, vz = v.to_value(u.km / u.s)
    return np.array([x, y, z, vx, vy, vz])


k = Earth.k.to(u.km**3 / u.s**2).value
rtol = 1e-13
full_periods = 2
u0 = state_to_vector(iss)
tf = (2 * full_periods + 1) * iss.period / 2

iss_f_kep = iss.propagate(tf)
iss_f_num = iss.propagate(tf, method=CowellPropagator(rtol=rtol))

assert np.allclose(iss_f_num.r, iss_f_kep.r, rtol=rtol, atol=1e-08 * u.km)
assert np.allclose(iss_f_num.v, iss_f_kep.v, rtol=rtol, atol=1e-08 * u.km / u.s)
assert np.allclose(iss_f_num.a, iss_f_kep.a, rtol=rtol, atol=1e-08 * u.km)
assert np.allclose(iss_f_num.ecc, iss_f_kep.ecc, rtol=rtol)
assert np.allclose(iss_f_num.inc, iss_f_kep.inc, rtol=rtol, atol=1e-08 * u.rad)
assert np.allclose(iss_f_num.raan, iss_f_kep.raan, rtol=rtol, atol=1e-08 * u.rad)
assert np.allclose(iss_f_num.argp, iss_f_kep.argp, rtol=rtol, atol=1e-08 * u.rad)
assert np.allclose(iss_f_num.nu, iss_f_kep.nu, rtol=rtol, atol=1e-08 * u.rad)

orb = Orbit.circular(Earth, 500 << u.km)
tof = 20 * orb.period

ad = constant_accel_factory(1e-7)


def f2(t0, state, k):
    """Computes acceleration with a different perturbation function."""
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = ad(t0, state, k)
    du_ad = np.array([0, 0, 0, ax, ay, az])
    return du_kep + du_ad


orb_final = orb.propagate(tof, method=CowellPropagator(f=f2))
da_a0 = (orb_final.a - orb.a) / orb.a
dv_v0 = abs(norm(orb_final.v) - norm(orb.v)) / norm(orb.v)
assert np.allclose(da_a0, 2 * dv_v0, rtol=1e-2)

print("Validation successful!")
