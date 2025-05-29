from yeager_utils import *

# Example parameters
lat = -30  # Equator, degrees
lon = 60.0  # Prime meridian, degrees
a = 10000e3  # Semi-major axis: 7000 km
i = 0.0  # Inclination: 30 degrees
t0 = Time("1999-6-1", scale="utc")

r_gcrf, v_gcrf = llh_to_gcrf(lon, lat, t=t0)
print(lon, lat, r_gcrf)

lon, lat, height = gcrf_to_llh(r_gcrf, t=t0)
print(r_gcrf, lat, height)

groundtrack_dashboard(r_gcrf, t0, show=True)

orbit = lonlat_perigee(lon=lon, lat=lat, a=a, i=0, t=t0)

r, v, t = quickint(orbit)

groundtrack_dashboard(r, t, show=True)