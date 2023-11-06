import numpy as np
W_rho = 19280  # kg/m^3 --> density of Tungsten
LD = 384399000  # lunar semi-major axis in meters

# Distances
au_to_m = 149597870700
pc_to_au = 206265
pc_to_m = 3.085677581e16
km_to_m = 1000
# Angles
deg_to_arcsecond = 3600
rad_to_arcsecond = 206265
rad_to_deg = 57.3
# Time
day_to_second = 86400
year_to_second = 31557600
year_to_minute = 525960
year_to_hour = 8766
year_to_day = 365.25
year_to_week = 365.25 / 7
year_to_month = 365.25 / 12
# Mass
kg_to_g = 1000
# Default rebound to SI
v_rebound_to_si = 4744 * 2 * np.pi  # au/2pi * yr to m/s
aupyr_to_mps = 4744

c = 299792458  # speed of light m/s
G = 6.67408e-11  # Gravitational constant m3 kg-1 s-2
kb = 1.38064852e-23  # boltzmann constant m2 kg s-2 K-1
pi = np.pi

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

# GM
WGS84_EARTH_MU = 3.986004418e14  # [m^3/s^2]
WGS72_EARTH_MU = 3.986005e14
# angular velocity
WGS84_EARTH_OMEGA = 72.92115147e-6  # [rad/s]
WGS72_EARTH_OMEGA = WGS84_EARTH_OMEGA
# radius at equator
WGS84_EARTH_RADIUS = 6.378137e6  # [m]
WGS72_EARTH_RADIUS = 6.378135e6  # [m]

# flattening f = (a-b)/a with a,b the major,minor axes
WGS84_EARTH_FLATTENING = 1 / 298.257223563
WGS72_EARTH_FLATTENING = 1 / 298.26
# polar radius can be derived from above; [m]
WGS84_EARTH_POLAR_RADIUS = WGS84_EARTH_RADIUS * (1 - WGS84_EARTH_FLATTENING)
WGS72_EARTH_POLAR_RADIUS = WGS72_EARTH_RADIUS * (1 - WGS72_EARTH_FLATTENING)

# GEO-sync radius and velocity are derived.
RGEO = np.cbrt(WGS84_EARTH_MU / WGS84_EARTH_OMEGA**2)  # [m]
VGEO = RGEO * WGS84_EARTH_OMEGA  # [m/s]
RGEOALT = RGEO - WGS84_EARTH_RADIUS  # [m] altitude of GEO
# Rough value:
VLEO = np.sqrt(WGS84_EARTH_MU / (WGS84_EARTH_RADIUS + 500e3))  # [m/s]

# Note JGM3 values from Montenbruck & Gill code are
# reference_radius = 6378.1363e3
# earth_mu = 398600.4415e+9

# VALUES FROM WIKI UNLESS STATED
SUN_MU = 1.32712438e+20  # [m^3/s^2] IAU 1976
MOON_MU = 398600.4415e+9 / 81.300587  # [m^3/s^2] DE200
MERCURY_MU = 2.2032e13
VENUS_MU = 3.24859e14
EARTH_MU = WGS84_EARTH_MU
MARS_MU = 4.282837e13
JUPITER_MU = 1.26686534e17
SATURN_MU = 3.7931187e16
URANUS_MU = 5.793939e15
NEPTUNE_MU = 6.836529e15

# MASS [kg] Values from the DE405 ephemeris
SUN_MASS = 1.98847e+30
MOON_MASS = 7.348e22
MERCURY_MASS = 3.301e23
VENUS_MASS = 4.687e24
EARTH_MASS = 5.9722e24
MARS_MASS = 6.417e23
JUPITER_MASS = 1.899e27
SATURN_MASS = 5.685e26
URANUS_MASS = 8.682e25
NEPTUNE_MASS = 1.024e26

# RADIUS - MEAN RADIUS FROM WIKI UNLESS STATED
SUN_RADIUS = 696340.0e3
MOON_RADIUS = 1738.1e3  # 10.2138/rmg.2006.60.3
MERCURY_RADIUS = 2439.4e3
VENUS_RADIUS = 6052e3
EARTH_RADIUS = WGS84_EARTH_RADIUS
MARS_RADIUS = 3389.5e3
JUPITER_RADIUS = 69911e3
SATURN_RADIUS = 58232e3
URANUS_RADIUS = 25362e3
NEPTUNE_RADIUS = 24622e3
PLUTO_RADIUS = 195e3

# Distance from Earth to Moon
LD = 384399000  # lunar semi-major axis in meters
