from yeager_utils import j2000_to_gcrf
import numpy as np
import pandas as pd


# Load orbit data from CSV
csv_file = "00104.csv"
orbit_data = pd.read_csv(csv_file)

# Extract J2000 positions (X, Y, Z) in meters
pos_j2000 = orbit_data[['X (m)', 'Y (m)', 'Z (m)']].values

# Get timestamps and preprocess to match 'isot' format
# Original: '2022-09-19 00:00:00.000000Z' -> '2022-09-19T00:00:00'
timestamps = orbit_data['Timestamp'].str.replace(' ', 'T').str.replace('.000000Z', '').values

# Convert to GCRF at each timestamp
pos_gcrf_list = []
for i, obstime in enumerate(timestamps):
    # Convert each position at its corresponding timestamp
    pos_gcrf = j2000_to_gcrf(pos_j2000[i:i + 1], obstime=obstime)
    pos_gcrf_list.append(pos_gcrf[0])  # Extract single row

pos_gcrf = np.array(pos_gcrf_list)

# Print results for the first few entries
n_display = min(5, len(timestamps))  # Show up to 5 rows
print("J2000 Positions (m):")
for i in range(n_display):
    print(f"{timestamps[i]}: {pos_j2000[i]}")
print("\nGCRF Positions (m):")
for i in range(n_display):
    print(f"{timestamps[i]}: {pos_gcrf[i]}")

# Optional: Compare with a fixed obstime (e.g., March 06, 2025)
fixed_obstime = "2025-03-06T12:00:00"
pos_gcrf_fixed = j2000_to_gcrf(pos_j2000, obstime=fixed_obstime)
print(f"\nGCRF Positions (m) at fixed time {fixed_obstime}:")
for i in range(n_display):
    print(f"{timestamps[i]} (original time): {pos_gcrf_fixed[i]}")
