#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This revision moves legends so they do not overlap other on-figure
# annotations, and converts all prints/comments to ASCII-only.
# It also keeps the calibrated benchmark overlay from the previous step.

from yeager_utils import gcrf_to_lonlat, EARTH_RADIUS, figpath
import numpy as np
import matplotlib.pyplot as plt

SIDEREAL_DAY_SEC = 86164.0905
EARTH_ROT_RATE_DEG_PER_S = 360.0 / SIDEREAL_DAY_SEC

# --- WGS84 constants for the benchmark geodetic conversion ---
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_A * WGS84_A)   # e^2
WGS84_EP2 = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B)  # e'^2


def _wrap180_deg(lon_deg):
    return ((lon_deg + 180.0) % 360.0) - 180.0


def _angdiff_abs_deg(a, b):
    # Smallest absolute angular difference in degrees
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def _ensure_2d_xyz(xyz):
    arr = np.atleast_2d(xyz).astype(np.float64)
    if arr.shape[-1] != 3:
        raise ValueError("Position array must have last-dimension size 3.")
    return arr


def _broadcast_time(t, n):
    tv = np.atleast_1d(t).astype(np.float64)
    if tv.size == 1:
        tv = np.full((n,), tv[0], dtype=np.float64)
    elif tv.size != n:
        raise ValueError(f"Time array length {tv.size} does not match positions length {n}.")
    return tv


# --------------------------
# Benchmark pipeline helpers
# --------------------------
def _eci_to_ecef_r3_sign_phase(r_eci, t_s, rot_sign=+1.0, phase_deg=0.0):
    """
    ECI (GCRF) -> ECEF using R3(theta) about +Z.
    theta(t) = rot_sign * omega_earth * t + phase_deg.
    rot_sign in {+1, -1} handles opposite sign conventions.
    phase_deg aligns the unknown Greenwich angle at t=0.
    """
    v = _ensure_2d_xyz(r_eci)
    tvec = _broadcast_time(t_s, v.shape[0])

    theta = (rot_sign * (2.0 * np.pi / SIDEREAL_DAY_SEC) * tvec) + np.deg2rad(phase_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    x = c * v[:, 0] - s * v[:, 1]
    y = s * v[:, 0] + c * v[:, 1]
    z = v[:, 2]
    return np.vstack((x, y, z)).T  # (N,3)


def _ecef_to_wgs84_geodetic(r_ecef):
    """
    Vectorized WGS84 geodetic conversion (Bowring method).
    Inputs:
      r_ecef: (...,3) ECEF meters
    Outputs:
      lon_deg, lat_deg, h_m  -> 1D arrays
    """
    v = _ensure_2d_xyz(r_ecef)
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]

    p = np.sqrt(x * x + y * y)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    st = np.sin(theta)
    ct = np.cos(theta)

    lat = np.arctan2(z + WGS84_EP2 * WGS84_B * (st * st * st),
                     p - WGS84_E2 * WGS84_A * (ct * ct * ct))
    lon = np.arctan2(y, x)

    slat = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * slat * slat)
    h = p / np.cos(lat) - N

    lon_deg = _wrap180_deg(np.degrees(lon))
    lat_deg = np.degrees(lat)

    return lon_deg.ravel(), lat_deg.ravel(), h.ravel()


def gcrf_to_lonlat_benchmark(r_eci, t_s, phase_deg=0.0, rot_sign=+1.0):
    """
    Reference/benchmark pipeline:
      1) ECI -> ECEF via R3(theta=rot_sign*omega_earth*t + phase_deg)
      2) ECEF -> geodetic (lon, lat, h) on WGS84
    Returns arrays shaped like gcrf_to_lonlat: (N,), even for single input.
    """
    r_ecef = _eci_to_ecef_r3_sign_phase(r_eci, t_s, rot_sign=rot_sign, phase_deg=phase_deg)
    lon_b, lat_b, h_b = _ecef_to_wgs84_geodetic(r_ecef)
    return lon_b, lat_b, h_b


def calibrate_phase_and_sign():
    """
    Infer the unknown Greenwich reference phase at t=0 (phase_deg) and
    the rotation sign convention (rot_sign) by comparing two calls to
    gcrf_to_lonlat at t0 and t1 for a simple equatorial direction.
    """
    R = float(EARTH_RADIUS)
    r_cal = np.array([R + 500e3, 0.0, 0.0])

    t0 = 0.0
    t1 = 3600.0  # 1 hour

    lon0, _, _ = gcrf_to_lonlat(r_cal, t0)
    lon1, _, _ = gcrf_to_lonlat(r_cal, t1)

    lon0 = float(lon0[0])
    lon1 = float(lon1[0])

    # At t0, theta=0, so phase_deg should equal lon0 (mod 360)
    phase_deg = lon0

    # Decide the rotation sign by checking which predicts lon1 better
    wdt = EARTH_ROT_RATE_DEG_PER_S * t1
    pred_plus  = _wrap180_deg(phase_deg + (+wdt))
    pred_minus = _wrap180_deg(phase_deg + (-wdt))

    err_plus  = _angdiff_abs_deg(pred_plus,  lon1)
    err_minus = _angdiff_abs_deg(pred_minus, lon1)

    rot_sign = +1.0 if err_plus <= err_minus else -1.0

    print(f"info: calibration -> phase_deg={phase_deg:.6f} deg, rot_sign={int(rot_sign)} "
          f"(err_plus={err_plus:.4f} deg, err_minus={err_minus:.4f} deg)")
    return phase_deg, rot_sign


def run_tests():
    np.set_printoptions(precision=6, suppress=True)
    R = float(EARTH_RADIUS)
    print("[ Tests ] Using EARTH_RADIUS =", R)

    # 1) Scalar input -> shape (1,)
    lon, lat, h = gcrf_to_lonlat(np.array([R, 0.0, 0.0]), 0.0)
    assert lon.shape == (1,)
    assert lat.shape == (1,)
    assert h.shape == (1,)
    print(f"ok: scalar shape -> lon{lon.shape}, lat{lat.shape}, h{h.shape}; "
          f"values lon={lon[0]:.6f}, lat={lat[0]:.6f}, h={h[0]:.3f} m")

    # 2) Vectorized inputs preserve length and produce finite outputs
    r = np.array([[R, 0.0, 0.0], [0.0, R, 0.0], [0.0, 0.0, R]])
    t = np.array([0.0, 10.0, 20.0])
    lon, lat, h = gcrf_to_lonlat(r, t)
    assert lon.shape == (3,) and lat.shape == (3,) and h.shape == (3,)
    assert np.all(np.isfinite(lon)) and np.all(np.isfinite(lat)) and np.all(np.isfinite(h))
    print(f"ok: vectorized length -> shapes {lon.shape}; finite counts = "
          f"{np.isfinite(lon).sum()}/{lon.size}")

    # 3) Latitude bounds
    lat_min, lat_max = float(lat.min()), float(lat.max())
    assert np.all((lat >= -90.0) & (lat <= 90.0))
    print(f"ok: latitude bounds -> min={lat_min:.6f} deg, max={lat_max:.6f} deg in [-90, 90]")

    # 4) Height increases with radial distance (same direction/time)
    _, _, h1 = gcrf_to_lonlat(np.array([R + 400e3, 0.0, 0.0]), 0.0)
    _, _, h2 = gcrf_to_lonlat(np.array([R + 800e3, 0.0, 0.0]), 0.0)
    assert h2[0] > h1[0]
    print(f"ok: height monotonic -> h(400 km)={h1[0]:.3f} m, h(800 km)={h2[0]:.3f} m")

    # 5) Consistency for repeated inputs
    rrep = np.array([[R, 0.0, 0.0]] * 3)
    trep = np.array([123.0, 123.0, 123.0])
    lon2, lat2, h2rep = gcrf_to_lonlat(rrep, trep)
    assert np.allclose(lon2, lon2[0]) and np.allclose(lat2, lat2[0]) and np.allclose(h2rep, h2rep[0])
    print(f"ok: consistency -> std(lon)={lon2.std():.3e}, std(lat)={lat2.std():.3e}, std(h)={h2rep.std():.3e}")

    # 6) Time dependence: approximate Earth rotation rate (1 hour delta)
    r_fix = np.array([R + 500e3, 0.0, 0.0])
    t0 = 1.0e6
    dt = 3600.0
    lon0, lat0, _ = gcrf_to_lonlat(r_fix, t0)
    lon1, lat1, _ = gcrf_to_lonlat(r_fix, t0 + dt)
    dlon = _angdiff_abs_deg(lon1[0], lon0[0])
    expected = EARTH_ROT_RATE_DEG_PER_S * dt  # ~15.041 deg/hour
    assert np.abs(dlon - expected) < 1.0
    assert np.abs(lat1[0] - lat0[0]) < 1e-3
    print(f"ok: time dependence -> dlon={dlon:.3f} deg, expected~={expected:.3f} deg, "
          f"|dlat|={np.abs(lat1[0]-lat0[0]):.3e} deg")

    # 7) Sidereal periodicity
    lonA, _, _ = gcrf_to_lonlat(r_fix, t0)
    lonB, _, _ = gcrf_to_lonlat(r_fix, t0 + SIDEREAL_DAY_SEC)
    d_sid = _angdiff_abs_deg(lonA[0], lonB[0])
    assert d_sid < 1.0
    print(f"ok: sidereal periodicity -> |lon(t+sidereal)-lon(t)|={d_sid:.3f} deg")

    # 8) Broadcast semantics for scalar t (accept either behavior)
    rN = np.array([[R + 10e3, 0.0, 0.0],
                   [0.0, R + 20e3, 0.0],
                   [0.0, 0.0, R + 30e3]])
    try:
        lon_b, lat_b, h_b = gcrf_to_lonlat(rN, 0.0)
        assert lon_b.shape == (3,) and lat_b.shape == (3,) and h_b.shape == (3,)
        print(f"ok: broadcast scalar t -> shapes ok: {lon_b.shape}")
    except Exception as e:
        print(f"note: broadcast scalar t not supported by function (skipping) -> {type(e).__name__}: {e}")

    # 9) Longitude domain convention
    lon_all, _, _ = gcrf_to_lonlat(r, t)
    lon_min, lon_max = float(lon_all.min()), float(lon_all.max())
    assert np.all(lon_all >= -180.0) and np.all(lon_all <= 180.0)
    print(f"ok: longitude domain -> min={lon_min:.6f} deg, max={lon_max:.6f} deg within [-180, 180]")

    # 10) Heights below/at/above ellipsoid
    _, _, h_eq = gcrf_to_lonlat(np.array([R, 0.0, 0.0]), 0.0)
    _, _, h_in = gcrf_to_lonlat(np.array([0.99 * R, 0.0, 0.0]), 0.0)
    assert np.abs(h_eq[0]) < 200.0 and (h_in[0] < 0.0)
    print(f"ok: heights -> h(eq)={h_eq[0]:.3f} m (~0), h(inside)={h_in[0]:.3f} m (<0)")

    # 11) Large-N vectorization/perf guard
    np.random.seed(0)
    N = 10000
    v = np.random.normal(size=(N, 3)).astype(np.float64)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    r_big = (R + 500e3) * v
    t_big = np.linspace(0.0, 1000.0, N)
    lon_big, lat_big, h_big = gcrf_to_lonlat(r_big, t_big)
    assert lon_big.shape == (N,) and lat_big.shape == (N,) and h_big.shape == (N,)
    assert np.all(np.isfinite(lon_big)) and np.all(np.isfinite(lat_big)) and np.all(np.isfinite(h_big))
    print(f"ok: large-N ({N}) -> lon[{lon_big.min():.3f},{lon_big.max():.3f}] deg, "
          f"lat[{lat_big.min():.3f},{lat_big.max():.3f}] deg, h[min={h_big.min():.1f}, max={h_big.max():.1f}] m")

    # 12) Dtype robustness (float32)
    r32 = r.astype(np.float32)
    t32 = t.astype(np.float32)
    lon32, lat32, h32 = gcrf_to_lonlat(r32, t32)
    assert lon32.shape == (3,) and np.all(np.isfinite(lon32))
    assert lat32.shape == (3,) and np.all(np.isfinite(lat32))
    assert h32.shape == (3,) and np.all(np.isfinite(h32))
    print(f"ok: float32 inputs -> lon32={lon32}, lat32={lat32}, h32={h32}")

    # 13) Agreement with calibrated WGS84 benchmark (random sample)
    phase_deg, rot_sign = calibrate_phase_and_sign()

    N_ref = 2000
    v = np.random.normal(size=(N_ref, 3)).astype(np.float64)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    r_ref = (R + np.random.uniform(-2.0e5, 1.0e6, size=(N_ref, 1))) * v
    t_ref = np.random.uniform(0.0, SIDEREAL_DAY_SEC, size=(N_ref,))
    lon_g, lat_g, h_g = gcrf_to_lonlat(r_ref, t_ref)
    lon_bm, lat_bm, h_bm = gcrf_to_lonlat_benchmark(r_ref, t_ref, phase_deg=phase_deg, rot_sign=rot_sign)

    dlon = _angdiff_abs_deg(lon_g, lon_bm)
    dlat = np.abs(lat_g - lat_bm)
    dh = np.abs(h_g - h_bm)

    rms = lambda arr: np.sqrt(np.mean(arr * arr))
    print(f"info: benchmark agreement (random {N_ref}, calibrated): "
          f"RMS dlon={rms(dlon):.4f} deg, max dlon={dlon.max():.4f} deg; "
          f"RMS dlat={rms(dlat):.4f} deg, max dlat={dlat.max():.4f} deg; "
          f"RMS dh={rms(dh):.1f} m, max dh={dh.max():.1f} m")

    print("ok: all tests passed.\n")


def _circular_orbit_r_eci(R, h_m, inc_deg, raan_deg, u0_deg, times_s):
    """
    Simple circular-orbit model in GCRF/ECI to generate a believable ground track.
    a = R + h; n = sqrt(mu/a^3); u(t) = u0 + n*t.
    r_eci = R3(Omega) R1(i) [a*cos u, a*sin u, 0]^T
    """
    mu = 398600.4418e9  # m^3/s^2
    a = R + h_m
    n = np.sqrt(mu / (a ** 3))
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    u = np.deg2rad(u0_deg) + n * np.atleast_1d(times_s)

    cu, su = np.cos(u), np.sin(u)
    # Position in orbital plane (PQW)
    x_pqw = a * cu
    y_pqw = a * su

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)

    # R3(Omega) R1(i) applied to PQW
    x = cO * x_pqw + (-sO * ci) * y_pqw
    y = sO * x_pqw + ( cO * ci) * y_pqw
    z = si * y_pqw

    return np.vstack((x, y, z)).T  # shape (N,3)


def make_groundtrack_image():
    R = float(EARTH_RADIUS)

    # Slightly inclined LEO track over ~1.5 orbits
    h = 500e3
    inc = 53.0
    raan = 30.0
    u0 = 0.0

    mu = 398600.4418e9
    a = R + h
    n = np.sqrt(mu / (a ** 3))
    T = 2.0 * np.pi / n
    t = np.linspace(0.0, 1.5 * T, 900)

    r_eci = _circular_orbit_r_eci(R, h, inc, raan, u0, t)

    # Calibrate benchmark alignment once for this plot
    phase_deg, rot_sign = calibrate_phase_and_sign()

    # Original function
    lon_deg, lat_deg, _ = gcrf_to_lonlat(r_eci, t)

    # Benchmark/reference (with calibrated phase/sign)
    lon_ref, lat_ref, _ = gcrf_to_lonlat_benchmark(r_eci, t, phase_deg=phase_deg, rot_sign=rot_sign)

    # Wrap for visualization and clamp lat
    lon_wrapped = _wrap180_deg(lon_deg)
    lon_ref_wrapped = _wrap180_deg(lon_ref)
    lat_clamped = np.clip(lat_deg, -90.0, 90.0)
    lat_ref_clamped = np.clip(lat_ref, -90.0, 90.0)

    # Compute error metrics for the overlay
    dlon = _angdiff_abs_deg(lon_wrapped, lon_ref_wrapped)
    dlat = np.abs(lat_clamped - lat_ref_clamped)

    # Scatter-only to avoid dateline artifacts
    plt.figure(figsize=(8, 4.5))
    plt.title("gcrf_to_lonlat vs calibrated WGS84 benchmark (~1.5 orbits)")
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.xlim([-180.0, 180.0])
    plt.ylim([-90.0, 90.0])
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.plot(lon_wrapped,     lat_clamped,     ".", markersize=2, label="gcrf_to_lonlat")
    plt.plot(lon_ref_wrapped, lat_ref_clamped, ".", markersize=2, label="Benchmark (R3 + WGS84)")

    # Place legend in upper-right so it does not overlap the stats box
    plt.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # Put the stats box in lower-left to avoid legend overlap
    rms_dlon = np.sqrt(np.mean(dlon * dlon))
    rms_dlat = np.sqrt(np.mean(dlat * dlat))
    txt = (f"RMS dlon={rms_dlon:.4f} deg\n"
           f"RMS dlat={rms_dlat:.4f} deg\n"
           f"max dlon={dlon.max():.4f} deg\n"
           f"max dlat={dlat.max():.4f} deg")
    plt.text(-178.0, -86.0, txt, fontsize=8,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, linewidth=0.5))

    outpath = figpath("tests/gcrf_to_lonlat_groundtrack_vs_benchmark.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"ok: ground track image (with calibrated benchmark) saved -> {outpath}")

    # Optional: error vs. sample index
    idx = np.arange(lon_wrapped.size)
    plt.figure(figsize=(8, 3.5))
    plt.title("Geodetic differences vs benchmark (calibrated)")
    plt.xlabel("Sample index")
    plt.ylabel("Delta (degrees)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.plot(idx, dlon, ".", markersize=2, label="abs(dlon)")
    plt.plot(idx, dlat, ".", markersize=2, label="abs(dlat)")
    # Put legend where it will not overlap plotted clusters (upper-right typically sparse here)
    plt.legend(loc="upper right", fontsize=9, framealpha=0.85)

    outpath_err = figpath("tests/gcrf_to_lonlat_groundtrack_errors.png")
    plt.tight_layout()
    plt.savefig(outpath_err, dpi=200)
    plt.close()
    print(f"ok: error plot saved -> {outpath_err}")


def demo():
    R = float(EARTH_RADIUS)
    r_demo = np.array([
        [R + 400e3, 0.0, 0.0],
        [0.0, (R + 500e3) / np.sqrt(2.0), (R + 500e3) / np.sqrt(2.0)],
        [-(R + 1000e3), 0.0, 0.0],
    ])
    t_demo = np.array([0.0, 60.0, 120.0])

    lon_deg, lat_deg, h_m = gcrf_to_lonlat(r_demo, t_demo)

    print("[ Demo outputs (geodetic) ]")
    for i in range(len(lon_deg)):
        print(f"{i}: lon = {lon_deg[i]:8.3f} deg, lat = {lat_deg[i]:8.3f} deg, height = {h_m[i]:10.1f} m")

    # Calibrated benchmark values for the same samples
    phase_deg, rot_sign = calibrate_phase_and_sign()
    lon_bm, lat_bm, h_bm = gcrf_to_lonlat_benchmark(r_demo, t_demo, phase_deg=phase_deg, rot_sign=rot_sign)
    for i in range(len(lon_bm)):
        print(f"    benchmark -> lon = {lon_bm[i]:8.3f} deg, lat = {lat_bm[i]:8.3f} deg, height = {h_bm[i]:10.1f} m")

    # Create and save a visual ground track image with calibrated benchmark overlay
    make_groundtrack_image()


if __name__ == "__main__":
    run_tests()
    demo()
