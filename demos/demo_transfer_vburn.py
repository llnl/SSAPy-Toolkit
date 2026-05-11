import os
import sys
import numpy as np

from ssapy_toolkit.Orbital_Mechanics.transfer_velocity_continuous import transfer_velocity_continuous
from ssapy_toolkit.constants import EARTH_RADIUS, RGEO, VGEO
from ssapy_toolkit.Plots.figpath import figpath  # [39]

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main():
    r0 = np.array([RGEO, 0.0, 0.0])
    v0 = np.array([0.0, VGEO, 0.0])

    out = transfer_velocity_continuous(r0=r0, v0=v0)
    return out


if __name__ == "__main__":
    main()