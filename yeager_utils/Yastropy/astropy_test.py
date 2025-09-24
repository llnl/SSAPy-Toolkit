

if __name__ == "__main__":
    import numpy as np

    from yeager_utils import *
    from datetime import datetime
    
    # Test coordinates
    test_date = Time("2025-6-1", scale="utc")
    original_lat = 37.7749
    original_lon = -122.4194
    original_alt = 100.0
    
    print("GCRF <-> Lat/Lon Conversion Test")
    print("=" * 40)
    
    # Forward conversion: lat/lon -> GCRF
    gcrf_pos = astropy_llh_to_gcrf(lon=original_lon, lat=original_lat, alt=original_alt, t=test_date)
    print(gcrf_pos)
    print(f"Original: Lat={original_lat:10.6f}°, Lon={original_lon:10.6f}°, Alt={original_alt:8.2f}m")
    print(f"GCRF: X={gcrf_pos[0, 0]:12.2f}m, Y={gcrf_pos[0, 1]:12.2f}m, Z={gcrf_pos[0, 2]:12.2f}m")
    
    # Reverse conversion: GCRF -> lat/lon
    recovered_lon, recovered_lat, recovered_alt = astropy_gcrf_to_llh(r_gcrf=gcrf_pos, t=test_date)
    print(f"Recovered: Lat={recovered_lat[0]:10.6f}°, Lon={recovered_lon[0]:10.6f}°, Alt={recovered_alt[0]:8.2f}m")
    
    # Calculate differences
    lat_diff = abs(recovered_lat[0] - original_lat)
    lon_diff = abs(recovered_lon[0] - original_lon)
    alt_diff = abs(recovered_alt[0] - original_alt)
    
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