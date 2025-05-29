from astropy.coordinates import GCRS, ITRS
from astropy.coordinates.representation import CartesianRepresentation
from astropy.time import Time
from astropy import units as u
import numpy as np


def astropy_gcrf_to_lonlat(gcrf_x, gcrf_y, gcrf_z, date_time):
    """
    Convert GCRF Cartesian coordinates to latitude, longitude, and altitude.
    
    Parameters:
    -----------
    gcrf_x, gcrf_y, gcrf_z : float
        GCRF Cartesian coordinates in meters
    date_time : datetime
        UTC datetime for the conversion
    
    Returns:
    --------
    tuple : (latitude, longitude, altitude)
        Latitude and longitude in degrees, altitude in meters above ellipsoid
    """
    # Create Time object
    obs_time = Time(date_time, format='datetime', scale='utc')
    
    # Create CartesianRepresentation object
    cart_repr = CartesianRepresentation(
        x=gcrf_x * u.m,
        y=gcrf_y * u.m,
        z=gcrf_z * u.m
    )
    
    # Create GCRS coordinate object using the representation
    gcrs_coord = GCRS(cart_repr, obstime=obs_time)
    
    # Transform to ITRS (Earth-fixed)
    itrs_coord = gcrs_coord.transform_to(ITRS(obstime=obs_time))
    
    # Convert to geodetic coordinates
    geodetic = itrs_coord.earth_location
    
    # Extract latitude, longitude, and altitude
    latitude = geodetic.lat.degree
    longitude = geodetic.lon.degree
    altitude = geodetic.height.to(u.m).value
    
    return latitude, longitude, altitude


def astropy_latlon_to_gcrf(lat, lon, alt, date_time):
    """
    Convert latitude, longitude, altitude to GCRF Cartesian coordinates.
    
    Parameters:
    -----------
    lat, lon : float
        Latitude and longitude in degrees
    alt : float
        Altitude in meters above ellipsoid
    date_time : datetime
        UTC datetime for the conversion
    
    Returns:
    --------
    tuple : (x, y, z)
        GCRF Cartesian coordinates in meters
    """
    from astropy.coordinates import EarthLocation
    
    # Create Time object
    obs_time = Time(date_time, format='datetime', scale='utc')
    
    # Create EarthLocation
    earth_loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=alt*u.m)
    
    # Convert to ITRS
    itrs_coord = ITRS(earth_loc.geocentric, obstime=obs_time)
    
    # Transform to GCRS
    gcrs_coord = itrs_coord.transform_to(GCRS(obstime=obs_time))
    
    # Extract Cartesian coordinates
    x = gcrs_coord.cartesian.x.to(u.m).value
    y = gcrs_coord.cartesian.y.to(u.m).value
    z = gcrs_coord.cartesian.z.to(u.m).value
    
    return x, y, z


# Example usage and test
if __name__ == "__main__":
    from datetime import datetime
    
    # Test coordinates
    test_date = datetime(2024, 6, 15, 12, 0, 0)
    original_lat = 37.7749
    original_lon = -122.4194
    original_alt = 100.0
    
    print("GCRF <-> Lat/Lon Conversion Test")
    print("=" * 40)
    
    # Forward conversion: lat/lon -> GCRF
    gcrf_pos = astropy_latlon_to_gcrf(original_lat, original_lon, original_alt, test_date)
    print(f"Original: Lat={original_lat:10.6f}°, Lon={original_lon:10.6f}°, Alt={original_alt:8.2f}m")
    print(f"GCRF: X={gcrf_pos[0]:12.2f}m, Y={gcrf_pos[1]:12.2f}m, Z={gcrf_pos[2]:12.2f}m")
    
    # Reverse conversion: GCRF -> lat/lon
    recovered_lat, recovered_lon, recovered_alt = astropy_gcrf_to_lonlat(
        gcrf_pos[0], gcrf_pos[1], gcrf_pos[2], test_date
    )
    print(f"Recovered: Lat={recovered_lat:10.6f}°, Lon={recovered_lon:10.6f}°, Alt={recovered_alt:8.2f}m")
    
    # Calculate differences
    lat_diff = abs(recovered_lat - original_lat)
    lon_diff = abs(recovered_lon - original_lon)
    alt_diff = abs(recovered_alt - original_alt)
    
    print(f"\nDifferences:")
    print(f"Lat: {lat_diff:12.9f}° ({lat_diff * 111000:8.3f}m)")
    print(f"Lon: {lon_diff:12.9f}° ({lon_diff * 111000 * np.cos(np.radians(original_lat)):8.3f}m)")
    print(f"Alt: {alt_diff:12.3f}m")
    
    # Check if round-trip is successful
    tolerance = 1e-6  # degrees
    if lat_diff < tolerance and lon_diff < tolerance and alt_diff < 1.0:
        print("\n✓ Round-trip conversion successful!")
    else:
        print("\n✗ Round-trip conversion failed!")