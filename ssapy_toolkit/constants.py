"""Physical and astronomical constants exposed through SSAPy Toolkit.

Shared astrodynamics constants are sourced from :mod:`ssapy.constants` so the
Toolkit and base SSAPy cannot drift.  Toolkit-specific convenience constants
remain defined here for backwards compatibility.
"""

import numpy as np
from ssapy import constants as _ssapy_constants


_SSAPY_CONSTANT_NAMES = tuple(
    name for name in dir(_ssapy_constants) if name.isupper()
)

for _name in _SSAPY_CONSTANT_NAMES:
    globals()[_name] = getattr(_ssapy_constants, _name)


# Toolkit-specific material, unit, angle, and time conveniences.
W_rho = 19280  # kg/m^3, density of tungsten

au_to_m = 149597870700
pc_to_au = 206265
pc_to_m = 3.085677581e16
km_to_m = 1000

deg_to_arcsecond = 3600
rad_to_arcsecond = 206265
rad_to_deg = 57.3

day_to_second = 86400
year_to_second = 31557600
year_to_minute = 525960
year_to_hour = 8766
year_to_day = 365.25
year_to_week = 365.25 / 7
year_to_month = 365.25 / 12

kg_to_g = 1000
v_rebound_to_si = 4744 * 2 * np.pi  # au/2pi * yr to m/s
aupyr_to_mps = 4744

c = 299792458  # speed of light m/s
G = 6.67408e-11  # gravitational constant m3 kg-1 s-2
J2_wgs = 1.08262668e-3
kb = 1.38064852e-23  # Boltzmann constant m2 kg s-2 K-1
pi = np.pi


# Solar-system convenience values not currently provided by base SSAPy.
MERCURY_a = 0.3871
VENUS_a = 0.7233
EARTH_a = 1.000
MARS_a = 1.5273
JUPITER_a = 5.2028
SATURN_a = 9.5388
URANUS_a = 19.1914
NEPTUNE_a = 30.0611

MERCURY_hill = 0.1753e9
VENUS_hill = 1.0042e9
EARTH_hill = 1.4714e9
MARS_hill = 0.9827e9
JUPITER_hill = 50.5736e9
SATURN_hill = 61.6340e9
URANUS_hill = 66.7831e9
NEPTUNE_hill = 115.0307e9
CERES_hill = 0.2048e9
PLUTO_hill = 5.9921e9
ERIS_hill = 8.1176e9

SUN_RADIUS = 696340000.0
PLUTO_RADIUS = 195000.0


_TOOLKIT_CONSTANT_NAMES = (
    "W_rho",
    "au_to_m",
    "pc_to_au",
    "pc_to_m",
    "km_to_m",
    "deg_to_arcsecond",
    "rad_to_arcsecond",
    "rad_to_deg",
    "day_to_second",
    "year_to_second",
    "year_to_minute",
    "year_to_hour",
    "year_to_day",
    "year_to_week",
    "year_to_month",
    "kg_to_g",
    "v_rebound_to_si",
    "aupyr_to_mps",
    "c",
    "G",
    "J2_wgs",
    "kb",
    "pi",
    "MERCURY_a",
    "VENUS_a",
    "EARTH_a",
    "MARS_a",
    "JUPITER_a",
    "SATURN_a",
    "URANUS_a",
    "NEPTUNE_a",
    "MERCURY_hill",
    "VENUS_hill",
    "EARTH_hill",
    "MARS_hill",
    "JUPITER_hill",
    "SATURN_hill",
    "URANUS_hill",
    "NEPTUNE_hill",
    "CERES_hill",
    "PLUTO_hill",
    "ERIS_hill",
    "SUN_RADIUS",
    "PLUTO_RADIUS",
)

__all__ = sorted(set(_SSAPY_CONSTANT_NAMES + _TOOLKIT_CONSTANT_NAMES))

del _name
