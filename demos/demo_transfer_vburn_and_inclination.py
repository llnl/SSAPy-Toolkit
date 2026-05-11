import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from ssapy_toolkit.Orbital_Mechanics.transfer_velocity_continuous import transfer_velocity_continuous
from ssapy_toolkit.Orbital_Mechanics.transfer_inclination_continuous import transfer_inclination_continuous
from ssapy_toolkit.constants import EARTH_RADIUS, RGEO, VGEO
from ssapy_toolkit.Plots.figpath import figpath
from ssapy_toolkit.Integrators.quick_int import quickint
from ssapy_toolkit.Time_Functions.get_times import get_times  # [40]

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST

    r0 = np.array([RGEO, 0.0, 0.0])
    v0 = np.array([0.0, VGEO, 0.0])

    r1, v1, t1 = transfer_velocity_continuous(r0=r0, v0=v0)
    r2, v2, t2 = transfer_inclination_continuous(r0=r0, v0=v0, a_thrust=1.0, i_target=np.radians(10.0))

    if make_figures:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(r1[:, 0], r1[:, 1], r1[:, 2], label="Velocity burn")
        ax.plot(r2[:, 0], r2[:, 1], r2[:, 2], label="Inclination burn")
        ax.scatter(*r0, color="black", s=50, label="Start Orbit")
        ax.scatter(*r1[-1], color="red", s=50, label="End Velocity Burn")
        ax.scatter(*r2[-1], color="green", s=50, label="End Inclination Burn")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Combined Trajectories of Velocity, Inclination Burns, and Coasting Orbit")
        ax.legend()
        plt.show()

    return {"velocity_burn": (r1, v1, t1), "inclination_burn": (r2, v2, t2)}


if __name__ == "__main__":
    main(make_figures=True)