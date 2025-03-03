from astropy import units as u
from astropy.time import Time
from itertools import product
from poliastro.bodies import Sun, Earth, Mars
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.util import time_range
from poliastro.maneuver import Maneuver
from matplotlib import pyplot as plt
from poliastro.plotting import StaticOrbitPlotter

# Departure and time of flight for the mission
EPOCH_DPT = Time("2018-12-01", scale="tdb")
EPOCH_ARR = EPOCH_DPT + 2 * u.year

epochs = time_range(EPOCH_DPT, end=EPOCH_ARR)

# Origin and target orbits
earth = Ephem.from_body(Earth, epochs=epochs)
mars = Ephem.from_body(Mars, epochs=epochs)

earth_departure = Orbit.from_ephem(Sun, earth, EPOCH_DPT)
mars_arrival = Orbit.from_ephem(Sun, mars, EPOCH_ARR)

# Generate all possible combinations of type of motion and path
type_of_motion_and_path = list(product([True, False], repeat=2))

# Prograde orbits use blue color while retrograde ones are drawn in red
colors_and_styles = [
    color + style for color in ["b", "r"] for style in ["-", "--"]
]


def lambert_solution_orbits(ss_departure, ss_arrival, M):
    """Computes all available solution orbits to the Lambert's problem."""

    for (is_prograde, is_lowpath) in type_of_motion_and_path:
        ss_sol = Maneuver.lambert(
            ss_departure,
            ss_arrival,
            M=M,
            prograde=is_prograde,
            lowpath=is_lowpath,
        )
        yield ss_sol


# Generate a grid of 3x1 plots
fig, axs = plt.subplots(3, 1, figsize=(8, 8))

for ith_case, M in enumerate(range(3)):
    # Plot the orbits of the Earth and Mars
    op = StaticOrbitPlotter(ax=axs[ith_case])
    axs[ith_case].set_title(f"{M = } scenario")

    op.plot_body_orbit(Earth, EPOCH_DPT)
    op.plot_body_orbit(Mars, EPOCH_ARR)

    for ss, colorstyle in zip(
        lambert_solution_orbits(earth_departure, mars_arrival, M=M),
        colors_and_styles,
    ):
        ss_plot_traj = op.plot_maneuver(
            earth_departure, ss, color=colorstyle[0]
        )
plt.tight_layout()
plt.show()
