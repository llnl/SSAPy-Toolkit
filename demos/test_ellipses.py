from yeager_utils import AccelKepler, SciPyPropagator, EARTH_RADIUS, Time, Orbit, get_times, orbit_plot_xy, rv, np, figpath
from tqdm import tqdm
print("Finished imports.")

accel = AccelKepler()
prop = SciPyPropagator(accel)

t0 = Time("2025-1-1", scale='utc')

ap = EARTH_RADIUS + 1000e3

rs = []
for peri in tqdm(np.linspace(10e3, EARTH_RADIUS, 10)):
    a = (peri + ap) / 2
    e = (ap - peri) / (peri + ap)

    kElements = [a, e, 0, 0, 0, 0]
    print("Initializing orbit.")
    orbit = Orbit.fromKeplerianElements(*kElements, t=t0)
    print(f"Orbital period: {orbit.period / 60} minutes")
    times = get_times(duration=(orbit.period, 's'), freq=(1, 's'), t0=t0)
    print("Computing orbit.")
    r, v = rv(orbit=orbit, time=times, propagator=prop)
    rs.append(r)


orbit_plot_xy(rs, save_path=figpath("testing_ellipses.jpg"), pad=500, title='Point source Earth')
