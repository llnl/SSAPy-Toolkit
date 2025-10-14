#!/usr/bin/env python3
"""
Hardcoded groundtrack validation using yeager_utils (+ssapy), no file I/O.

Uses only your utilities:
- yeager_utils.get_times, gcrf_to_lonlat
- yeager_utils.ssapy_wrapper.{best_prop, keplerian_prop}
- ssapy.Orbit, ssapy.rv
- yeager_utils.Coordinates.on_sky_distance.lonlat_distance  (returns METERS)

Prints summary metrics + a short preview. No CSVs are written.
"""

import numpy as np

# From your utilities
from yeager_utils import Time, get_times, gcrf_to_lonlat
from yeager_utils.Coordinates.on_sky_distance import lonlat_distance
from yeager_utils.ssapy_wrapper import keplerian_prop, best_prop
from ssapy import Orbit, rv

# ---------------- Hardcoded Inputs ----------------
epoch_iso = "2025-10-13T00:00:00"   # epoch (UTC)
# Keplerian elements at epoch
a_m      = 6878137.0   # meters  (~500 km circular-ish LEO)
e        = 0.0010
i_deg    = 51.6
pa_deg   = 0.0
raan_deg = 40.0
ta_deg   = 10.0        # true anomaly at epoch [deg]

# time grid
duration_s = 5400       # seconds
step_s     = 60         # seconds

# test threshold
tolerance_km = 50.0
# --------------------------------------------------

def main():
    # Epoch & orbit
    t0 = Time(epoch_iso)
    orbit = Orbit.fromKeplerianElements(
        a=a_m,
        e=e,
        i=np.radians(i_deg),
        pa=np.radians(pa_deg),
        raan=np.radians(raan_deg),
        trueAnomaly=np.radians(ta_deg),
        t=t0
    )

    # Times
    times = get_times(duration=duration_s, freq=(step_s, 's'), t0=t0)

    # Reference: richer physics
    prop_ref  = best_prop()
    r_ref, v_ref = rv(orbit=orbit, time=times, propagator=prop_ref)
    lon_ref, lat_ref, h_ref = gcrf_to_lonlat(r_ref, times)

    # Test: simple two-body
    prop_test = keplerian_prop()
    r_test, v_test = rv(orbit=orbit, time=times, propagator=prop_test)
    lon_test, lat_test, h_test = gcrf_to_lonlat(r_test, times)

    # Compute great-circle errors; lonlat_distance returns **meters**, convert to km
    errs_km = np.array([
        lonlat_distance(np.radians(lat_ref[i]),  np.radians(lat_test[i]),
                        np.radians(lon_ref[i]),  np.radians(lon_test[i])) / 1000.0
        for i in range(len(lat_ref))
    ])

    # sanity check: cannot exceed antipodal arc (~20037 km)
    antipode_km = np.pi * 6378137.0 / 1000.0
    if np.nanmax(errs_km) > antipode_km + 1e-3:
        print(f"[warn] error exceeds antipodal distance ({antipode_km:.1f} km). "
              "This usually indicates a units or angle bug.")

    mean_km = float(np.nanmean(errs_km))
    rms_km  = float(np.sqrt(np.nanmean(errs_km**2)))
    max_km  = float(np.nanmax(errs_km))

    print("Inputs:")
    print(f"  epoch = {epoch_iso}")
    print(f"  a={a_m:.1f} m, e={e:.5f}, i={i_deg:.3f} deg, ω={pa_deg:.3f} deg, Ω={raan_deg:.3f} deg, ν={ta_deg:.3f} deg")
    print(f"  duration={duration_s}s, step={step_s}s, samples={len(times)}")

    print("\n=== Groundtrack Error [km] (test vs reference) ===")
    print(f"Mean: {mean_km:.3f}")
    print(f"RMS : {rms_km:.3f}")
    print(f"Max : {max_km:.3f}")
    print(f"RESULT: {'PASS' if max_km <= tolerance_km else 'FAIL'} (threshold {tolerance_km:.1f} km)")

    # Small preview of first few points (time, ref lat/lon → test lat/lon)
    n_preview = min(5, len(times))
    print("\nPreview (first few samples):")
    for i in range(n_preview):
        print(f"{times.iso[i]}  "
              f"ref(lat,lon)=({lat_ref[i]: .4f},{lon_ref[i]: .4f})  "
              f"test=({lat_test[i]: .4f},{lon_test[i]: .4f})  "
              f"err_km={errs_km[i]:.2f}")

if __name__ == "__main__":
    main()
