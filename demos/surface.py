from yeager_utils import *
import numpy as np

# Example parameters
lat = 30  # Equator, degrees
lon = -60.0  # Prime meridian, degrees
alt = 10000e3  # Semi-major axis: 7000 km'
e = 0.2
i = 0.0  # Inclination: 30 degrees
t0 = Time("1999-12-1", scale="utc")

r_gcrf, v_gcrf = llh_to_gcrf(lon, lat, t=t0)
print(lon, lat, r_gcrf)

lon, lat, height = gcrf_to_llh(r_gcrf, t=t0)
print(r_gcrf, lat, height)

print("astropy_llh_to_gcrf")
r_gcrf = astropy_llh_to_gcrf(lon=lon, lat=lat, t=t0)
print(lon, lat, r_gcrf)

print("astropy_gcrf_to_llh")
lon, lat, height = astropy_gcrf_to_llh(r_gcrf, t=t0)
print(r_gcrf, lon, lat)

groundtrack_dashboard(r_gcrf, t0, show=True)

print("Astropy surface rv")
r_gcrf, v_gcrf = astropy_surface_rv(lon, lat, t=t0)
print(r_gcrf)
groundtrack_dashboard(r_gcrf, t0, show=True)

r_gcrf, _ = astropy_surface_rv(lon, lat, t=t0)
rp = (EARTH_RADIUS + alt) * (1 - e)
r_peri = rp * (r_gcrf / np.linalg.norm(r_gcrf))
r_hat = r_peri / np.linalg.norm(r_peri)
groundtrack_dashboard(r_peri, t0, show=True)


print("Astropy lonlat_perigee rv")
orbit = lonlat_perigee(lon=lon, lat=lat, t=t0, alt=alt, i=0)
r, v, t = quickint(orbit)
groundtrack_dashboard(r, t0, show=True)