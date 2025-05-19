from yeager_utils import *

# Example parameters
lat = 0.0  # Equator, degrees
lon = 0.0  # Prime meridian, degrees
t0 = Time("2025-12-20 12:00:00", format="iso", scale="utc")
a = 20000e3  # Semi-major axis: 7000 km
e = 0.0     # Eccentricity
i = 90.0  # Inclination: 30 degrees

orbit = lonlat_perigee(lon=lon, lat=lat, t0=t0, a=a, e=e, i=i)

r, v, t = quickint(orbit)

groundtrack_dashboard(r, t, show=True)