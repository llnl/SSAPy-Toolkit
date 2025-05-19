from yeager_utils import *

# Example parameters
lat = -45.0  # Equator, degrees
lon = 0.0  # Prime meridian, degrees
a = 20000e3  # Semi-major axis: 7000 km
i = 0.0  # Inclination: 30 degrees

orbit = lonlat_perigee(lon=lon, lat=lat, a=a)

r, v, t = quickint(orbit)

groundtrack_dashboard(r, t, show=True)