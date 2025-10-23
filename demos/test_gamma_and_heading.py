import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from pathlib import Path

from astropy.time import Time
from astropy.coordinates import ITRS, GCRS, CartesianRepresentation
import astropy.units as u

# ---------------------------------------------------------------------
# Project utilities from this codebase
# ---------------------------------------------------------------------
from yeager_utils import (
    ssapy_orbit,
    calc_gamma_and_heading,              # imported for optional use elsewhere
    figpath,
    EARTH_RADIUS,                        # [m]
    groundtrack_dashboard_gamma_heading  # dashboard figure (gamma + heading time series)
)

try:
    from ssapy import groundTrack
    SSAPY_AVAILABLE = True
except Exception:
    SSAPY_AVAILABLE = False

# =====================================================================
# GLOBAL CONFIG
# =====================================================================
# Scenario (for the dashboard figure)
FS_BASE  = 20
FS_TITLE = FS_BASE + 2
FS_LABEL = FS_BASE
FS_TICKS = FS_BASE - 4

a_m      = EARTH_RADIUS + 7000e3   # [m]
e        = 0.2
i_deg    = 0.0
t0_str   = "2025-01-01T00:00:00"
duration = (1, "day")
freq     = (30, "s")

# Quiver-panel defaults (2×3 heading/gamma maps)
GRID_AZ_STEP_DEG    = 45.0
GRID_LAT_STEP_DEG   = 45.0
D_DISP_M            = 10000.0       # small step to define local velocity direction
DT_S                 = 1.0
ARROW_LEN_DEG        = 22.5
ARROW_WIDTH          = 0.035        # thicker 2D arrows
ARROW_EDGE_LW        = 1.4          # outline
HEADLENGTH           = 13
HEADWIDTH            = 12
EQUAL_ONSCREEN_LEN   = True         # normalize on-screen length so arrows look comparable
DRAW_RADIAL_MARKERS  = True         # mark Up/Down (undefined heading) with colored squares

# 3D arrow defaults (single plot)
GRID_LON_STEP_DEG    = 45.0
GRID_LAT_STEP_DEG3   = 45.0
ALTITUDE_M           = 700e3
ARROW_LEN_KM_3D      = 1200.0
ARROW_LINEWIDTH_3D   = 3.0
ARROW_HEAD_RATIO_3D  = 0.20
FRAME_FOR_3D         = "ITRF"        # "ITRF" or "GCRF"
TITLE_3D             = "3D arrows — categories colored (Up/Down/East/West/North/South)"

# =====================================================================
# NUMPY HELPERS (avoid math/typing; keep everything in numpy)
# =====================================================================
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n == 0.0 else v / n

def _enu_basis_from_r(vec):
    """
    Local ENU at position 'vec' (meters) in the same frame as 'vec'.
    e_hat ~ East, n_hat ~ North, u_hat ~ Up (radially outward).
    Assumes Earth's spin axis is +Z in that frame.
    """
    u_hat = _unit(vec)
    k = np.array([0.0, 0.0, 1.0])  # Earth's spin axis
    e_hat = np.cross(k, u_hat)
    if np.linalg.norm(e_hat) < 1e-12:
        e_hat = np.cross(np.array([1.0, 0.0, 0.0]), u_hat)
    e_hat = _unit(e_hat)
    n_hat = _unit(np.cross(u_hat, e_hat))
    return e_hat, n_hat, u_hat

def _heading_gamma_from_pair(
    r0, r1, dt_s,
    vt_rel_eps=1e-8,  # relative horizontal threshold
    vt_abs_eps=0.0,   # absolute horizontal threshold
    gamma_radial_eps_deg=89.999
):
    """
    Frame-agnostic local heading/gamma from two nearby positions in the same frame.
      - heading: compass azimuth of horizontal velocity (0°=N, 90°=E)
      - gamma:   flight-path angle (+up), in degrees
    """
    e_hat, n_hat, u_hat = _enu_basis_from_r(r0)
    v = (np.asarray(r1, float) - np.asarray(r0, float)) / float(dt_s)

    ve = float(np.dot(v, e_hat))
    vn = float(np.dot(v, n_hat))
    vu = float(np.dot(v, u_hat))

    vt   = np.sqrt(ve*ve + vn*vn)
    vmag = np.sqrt(vt*vt + vu*vu)
    gamma = np.degrees(np.arctan2(vu, vt))  # [-90, 90]

    # Robust handling: for nearly radial motion (Up/Down), heading is undefined.
    vt_eps = vt_abs_eps + vt_rel_eps * (vmag if np.isfinite(vmag) else 0.0)
    is_radial = (vt <= vt_eps) or (np.abs(gamma) >= gamma_radial_eps_deg)

    if is_radial:
        heading = np.nan
    else:
        heading = (np.degrees(np.arctan2(ve, vn)) + 360.0) % 360.0  # 0°=N, 90°=E

    return gamma, heading

def _itrs_dirs_to_gcrs(r0_itrs_m, dirs_itrs, obstime):
    """
    Rotate an ITRF point and a dict of unit direction vectors into GCRS at obstime.
    Returns: r0_gcrs (meters), dirs_gcrs (same keys, unit vectors).
    """
    r0_itrs = CartesianRepresentation(*(np.asarray(r0_itrs_m, float) * u.m))
    r0_gcrs_vec = ITRS(r0_itrs, obstime=obstime).transform_to(GCRS(obstime=obstime)).cartesian.xyz.to_value(u.m)
    r0_gcrs = np.array(r0_gcrs_vec, float)

    eps = 1.0  # 1 m step for differential direction mapping
    dirs_gcrs = {}
    for key, d_itrs in dirs_itrs.items():
        p1_itrs = CartesianRepresentation(*((np.asarray(r0_itrs_m, float) + eps * np.asarray(d_itrs, float)) * u.m))
        p1_gcrs_vec = ITRS(p1_itrs, obstime=obstime).transform_to(GCRS(obstime=obstime)).cartesian.xyz.to_value(u.m)
        d_gcrs = np.array(p1_gcrs_vec, float) - r0_gcrs
        n = np.linalg.norm(d_gcrs)
        dirs_gcrs[key] = d_gcrs / (1e-18 if n == 0.0 else n)
    return r0_gcrs, dirs_gcrs

def _dir_from_heading_gamma(e_hat, n_hat, u_hat, heading_deg, gamma_deg):
    """
    Convert heading (0°=N, 90°=E) and gamma (+up) to a 3D unit vector
    expressed in the same frame as the provided ENU basis.
    """
    h = np.radians(heading_deg)
    g = np.radians(gamma_deg)
    vec = (np.cos(g) * np.sin(h)) * e_hat + (np.cos(g) * np.cos(h)) * n_hat + (np.sin(g)) * u_hat
    return _unit(vec)

def _wrap_deg(d): return (np.asarray(d, float) % 360.0)

def _ang_diff_deg(a, b):
    a = _wrap_deg(a)
    b = _wrap_deg(b)
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)

# =====================================================================
# 2×3 HEADING/GAMMA PANELS (ITRF or GCRF)
# =====================================================================
def heading_gamma_panel(
    frame_label="ITRF",           # "ITRF" or "GCRF"
    altitude_m=700e3,
    az_step_deg=GRID_AZ_STEP_DEG,
    lat_step_deg=GRID_LAT_STEP_DEG,
    d_disp_m=D_DISP_M,
    dt_s=DT_S,
    t0=t0_str,
    arrow_len_deg=ARROW_LEN_DEG,
    arrow_width=ARROW_WIDTH,
    arrow_edge_lw=ARROW_EDGE_LW,
    headlength=HEADLENGTH,
    headwidth=HEADWIDTH,
    equal_onscreen_length=EQUAL_ONSCREEN_LEN,
    draw_radial_markers=DRAW_RADIAL_MARKERS,
    show_points=True,
):
    """
    Build a 2×3 panel of quiver plots (Up, Down, East, West, North, South),
    where arrow direction encodes heading (0°=N, 90°=E) and color encodes gamma.
    The small 2-point tracks are constructed directly in the requested frame.
    """
    frame_label_u = str(frame_label).upper().strip()
    if frame_label_u not in ("ITRF", "GCRF"):
        frame_label_u = "ITRF"

    R = float(EARTH_RADIUS) + float(altitude_m)
    lons = np.arange(0.0, 360.0, float(az_step_deg), dtype=float)
    lats = np.arange(-90.0, 90.0 + 1e-9, float(lat_step_deg), dtype=float)
    obstime = Time(t0, scale="utc")

    dirs = ["Up", "Down", "East", "West", "North", "South"]
    data = {d: {"lon": [], "lat": [], "heading": [], "gamma": []} for d in dirs}

    for lat_deg in lats:
        phi = np.radians(lat_deg)
        for lon_deg in lons:
            lam = np.radians(lon_deg)

            r0_itrs = R * np.array([
                np.cos(phi) * np.cos(lam),
                np.cos(phi) * np.sin(lam),
                np.sin(phi)
            ])

            e_hat, n_hat, u_hat = _enu_basis_from_r(r0_itrs)
            dirs_itrs = {
                "Up":    +u_hat,
                "Down":  -u_hat,
                "East":  +e_hat,
                "West":  -e_hat,
                "North": +n_hat,
                "South": -n_hat,
            }

            if frame_label_u == "GCRF":
                r0_gcrs, dirs_gcrs = _itrs_dirs_to_gcrs(r0_itrs, dirs_itrs, obstime)
                for name, dhat in dirs_gcrs.items():
                    r1 = r0_gcrs + float(d_disp_m) * dhat
                    g0, h0 = _heading_gamma_from_pair(r0_gcrs, r1, dt_s)
                    data[name]["lon"].append(lon_deg)
                    data[name]["lat"].append(lat_deg)
                    data[name]["heading"].append(h0)
                    data[name]["gamma"].append(g0)
            else:
                for name, dhat in dirs_itrs.items():
                    r1 = r0_itrs + float(d_disp_m) * dhat
                    g0, h0 = _heading_gamma_from_pair(r0_itrs, r1, dt_s)
                    data[name]["lon"].append(lon_deg)
                    data[name]["lat"].append(lat_deg)
                    data[name]["heading"].append(h0)
                    data[name]["gamma"].append(g0)

    fig, axes = plt.subplots(2, 3, figsize=(24, 14), constrained_layout=True)
    axes = axes.ravel()

    vmin, vmax = -90.0, 90.0
    gamma_norm = Normalize(vmin=vmin, vmax=vmax)

    for ax, name in zip(axes, dirs):
        lon_arr = np.asarray(data[name]["lon"], float)
        lat_arr = np.asarray(data[name]["lat"], float)
        hdg_arr = np.asarray(data[name]["heading"], float)
        gam_arr = np.asarray(data[name]["gamma"], float)

        # Map heading to lon/lat-plane vectors; correct lon by cos(lat)
        lat_rad = np.radians(lat_arr)
        hrad    = np.radians(hdg_arr)
        coslat  = np.cos(lat_rad)
        coslat  = np.where(np.abs(coslat) < 1e-12, 1e-12, coslat)

        u_dir = np.sin(hrad) / coslat   # +lon component
        v_dir = np.cos(hrad)            # +lat component

        # Normalize on-screen arrow length to keep arrows visually comparable
        if EQUAL_ONSCREEN_LEN:
            norm = np.sqrt(u_dir*u_dir + v_dir*v_dir)
            norm = np.where(norm == 0.0, 1.0, norm)
            u_plot = (u_dir / norm) * ARROW_LEN_DEG
            v_plot = (v_dir / norm) * ARROW_LEN_DEG
        else:
            u_plot = u_dir * ARROW_LEN_DEG
            v_plot = v_dir * ARROW_LEN_DEG

        mask = ~np.isfinite(u_plot) | ~np.isfinite(v_plot) | ~np.isfinite(gam_arr)

        # For Up/Down (undefined heading), draw colored squares at the sites
        if DRAW_RADIAL_MARKERS:
            mask_radial = np.isnan(hdg_arr) & np.isfinite(gam_arr)
            if np.any(mask_radial):
                ax.scatter(
                    lon_arr[mask_radial], lat_arr[mask_radial],
                    c=np.clip(gam_arr[mask_radial], vmin, vmax),
                    cmap="coolwarm", norm=gamma_norm,
                    s=52, marker="s", zorder=2.6, alpha=0.95,
                )

        lon_plot = lon_arr[~mask]
        lat_plot = lat_arr[~mask]
        u_plot   = u_plot[~mask]
        v_plot   = v_plot[~mask]
        c_plot   = np.clip(gam_arr[~mask], vmin, vmax)

        ax.scatter(lon_plot, lat_plot, s=30, alpha=0.55, zorder=2)

        q = ax.quiver(
            lon_plot, lat_plot, u_plot, v_plot, c_plot,
            cmap="coolwarm", norm=gamma_norm,
            angles="xy", scale_units="xy", scale=1.0,
            width=ARROW_WIDTH,                 # thicker shaft
            linewidths=ARROW_EDGE_LW,          # thicker outline
            headlength=HEADLENGTH,
            headwidth=HEADWIDTH,
            headaxislength=HEADLENGTH,
            pivot="tail",
            zorder=3,
        )

        ax.set_xlim(-2, 362)
        ax.set_ylim(-95, 95)
        ax.set_xticks(np.arange(0, 361, 45))
        ax.set_yticks(np.arange(-90, 91, 45))
        ax.grid(True, alpha=0.35)
        ax.set_title(f"{name} velocity — {frame_label_u}", fontsize=FS_TITLE)
        ax.set_xlabel("Longitude (deg)", fontsize=FS_LABEL)
        ax.set_ylabel("Latitude (deg)", fontsize=FS_LABEL)
        ax.tick_params(axis="both", labelsize=FS_TICKS)

        cb = plt.colorbar(q, ax=ax, pad=0.01, fraction=0.046)
        cb.set_label("Gamma (deg)", fontsize=FS_LABEL)
        cb.ax.tick_params(labelsize=FS_TICKS)

    return fig

# =====================================================================
# SINGLE 3D PLOT (one plot, colored categories, black axes + circles)
# =====================================================================
def _itrs_point_and_dir_to_gcrs(r0_itrs_m, dhat_itrs, obstime):
    r0_itrs = CartesianRepresentation(*(np.asarray(r0_itrs_m, float) * u.m))
    r0_gcrs_vec = ITRS(r0_itrs, obstime=obstime).transform_to(GCRS(obstime=obstime)).cartesian.xyz.to_value(u.m)
    r0_gcrs = np.array(r0_gcrs_vec, float)

    eps = 1.0
    p1_itrs = CartesianRepresentation(*((np.asarray(r0_itrs_m, float) + eps * np.asarray(dhat_itrs, float)) * u.m))
    p1_gcrs_vec = ITRS(p1_itrs, obstime=obstime).transform_to(GCRS(obstime=obstime)).cartesian.xyz.to_value(u.m)
    d_gcrs = np.array(p1_gcrs_vec, float) - r0_gcrs
    return r0_gcrs, _unit(d_gcrs)

def heading_gamma_arrows_3d_single(
    frame_label="ITRF",
    t0="2025-01-01T00:00:00",
    altitude_m=ALTITUDE_M,
    lon_step_deg=GRID_LON_STEP_DEG,
    lat_step_deg=GRID_LAT_STEP_DEG3,
    arrow_len_km=ARROW_LEN_KM_3D,
    title=TITLE_3D,
):
    """
    Single 3D visualization:
      * draw the Earth, X/Y/Z axes, and three great circles,
      * place arrow bases at a lon/lat grid on a sphere of radius (R_earth + altitude),
      * for each category (Up/Down/East/West/North/South), draw heading/gamma arrows
        with distinct colors. No colorbar is used; direction + color are the cues.
    """
    frame_u = str(frame_label).upper().strip()
    if frame_u not in ("ITRF", "GCRF"):
        frame_u = "ITRF"
    obstime = Time(t0, scale="utc")

    cat_colors = {
        "Up":    "#9467bd",
        "Down":  "#8c564b",
        "East":  "#1f77b4",
        "West":  "#ff7f0e",
        "North": "#2ca02c",
        "South": "#d62728",
    }
    cat_specs = {
        "Up":    {"heading":   0.0, "gamma": +90.0},
        "Down":  {"heading":   0.0, "gamma": -90.0},
        "East":  {"heading":  90.0, "gamma":   0.0},
        "West":  {"heading": 270.0, "gamma":   0.0},
        "North": {"heading":   0.0, "gamma":   0.0},
        "South": {"heading": 180.0, "gamma":   0.0},
    }

    Rm = float(EARTH_RADIUS)
    Rk = Rm / 1e3
    Hm = float(altitude_m)
    Hk = Hm / 1e3
    Rtot_km = Rk + Hk

    lons = np.arange(0.0, 360.0, float(lon_step_deg), dtype=float)
    lats = np.arange(-90.0, 90.0 + 1e-9, float(lat_step_deg), dtype=float)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Earth mesh
    theta = np.linspace(0.0, 2.0*np.pi, 80)
    phi   = np.linspace(0.0, np.pi, 40)
    tt, pp = np.meshgrid(theta, phi)
    ex = Rk * np.sin(pp) * np.cos(tt)
    ey = Rk * np.sin(pp) * np.sin(tt)
    ez = Rk * np.cos(pp)
    ax.plot_surface(ex, ey, ez, color="lightblue", alpha=0.35, linewidth=0.0, zorder=0)

    legend_handles = []
    for cat, spec in cat_specs.items():
        hdeg = float(spec["heading"]); gdeg = float(spec["gamma"])
        Xb, Yb, Zb, U, V, W = [], [], [], [], [], []

        for lat_deg in lats:
            phi_rad = np.radians(lat_deg)
            for lon_deg in lons:
                lam = np.radians(lon_deg)
                rhat = np.array([np.cos(phi_rad) * np.cos(lam),
                                 np.cos(phi_rad) * np.sin(lam),
                                 np.sin(phi_rad)], float)
                r0_itrs_m = (Rm + Hm) * rhat
                e_hat, n_hat, u_hat = _enu_basis_from_r(r0_itrs_m)
                dhat_itrs = _dir_from_heading_gamma(e_hat, n_hat, u_hat, hdeg, gdeg)

                if frame_u == "GCRF":
                    base_gcrs_m, dhat_gcrs = _itrs_point_and_dir_to_gcrs(r0_itrs_m, dhat_itrs, obstime)
                    base = base_gcrs_m / 1e3
                    dvec = dhat_gcrs
                else:
                    base = r0_itrs_m / 1e3
                    dvec = dhat_itrs

                Xb.append(base[0]); Yb.append(base[1]); Zb.append(base[2])
                U.append(dvec[0]);  V.append(dvec[1]);  W.append(dvec[2])

        ax.scatter(Xb, Yb, Zb, s=10, c=cat_colors[cat], alpha=0.28, zorder=2)
        ax.quiver(
            Xb, Yb, Zb, U, V, W,
            length=float(arrow_len_km),
            normalize=True,
            color=cat_colors[cat],
            linewidths=ARROW_LINEWIDTH_3D,
            arrow_length_ratio=ARROW_HEAD_RATIO_3D,
            zorder=3,
        )
        legend_handles.append(Line2D([0], [0], color=cat_colors[cat], lw=4.0, label=cat))

    ax.legend(handles=legend_handles, loc="upper left", fontsize=FS_TICKS)
    ax.set_xlabel("X (km)", fontsize=FS_LABEL)
    ax.set_ylabel("Y (km)", fontsize=FS_LABEL)
    ax.set_zlabel("Z (km)", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICKS)

    rmax = (Rtot_km + float(ARROW_LEN_KM_3D)) * 1.05
    ax.set_xlim([-rmax, rmax]); ax.set_ylim([-rmax, rmax]); ax.set_zlim([-rmax, rmax])

    # Black axis lines and labels for orientation
    ax.plot([-rmax, rmax], [0.0, 0.0], [0.0, 0.0], color="black", linewidth=2.2, zorder=4)
    ax.plot([0.0, 0.0], [-rmax, rmax], [0.0, 0.0], color="black", linewidth=2.2, zorder=4)
    ax.plot([0.0, 0.0], [0.0, 0.0], [-rmax, rmax], color="black", linewidth=2.2, zorder=4)
    ax.text(rmax, 0.0, 0.0, "X", fontsize=FS_LABEL, color="black", ha="left", va="center")
    ax.text(0.0, rmax, 0.0, "Y", fontsize=FS_LABEL, color="black", ha="center", va="bottom")
    ax.text(0.0, 0.0, rmax, "Z", fontsize=FS_LABEL, color="black", ha="center", va="bottom")

    # Three black great circles (XY, YZ, XZ) at the arrow-base radius to aid visual checks
    t = np.linspace(0.0, 2.0*np.pi, 721)
    x_xy = Rtot_km * np.cos(t); y_xy = Rtot_km * np.sin(t); z_xy = np.zeros_like(t)
    ax.plot(x_xy, y_xy, z_xy, color="black", linewidth=2.0, alpha=0.9, zorder=4)
    x_yz = np.zeros_like(t); y_yz = Rtot_km * np.cos(t); z_yz = Rtot_km * np.sin(t)
    ax.plot(x_yz, y_yz, z_yz, color="black", linewidth=2.0, alpha=0.9, zorder=4)
    x_xz = Rtot_km * np.cos(t); y_xz = np.zeros_like(t); z_xz = Rtot_km * np.sin(t)
    ax.plot(x_xz, y_xz, z_xz, color="black", linewidth=2.0, alpha=0.9, zorder=4)

    try: ax.set_box_aspect((1, 1, 1))
    except Exception: pass
    try: ax.set_proj_type("ortho")
    except Exception: pass

    ax.set_title(title if title else f"3D heading/gamma arrows — {frame_u}", fontsize=FS_TITLE)
    return fig

# =====================================================================
# ECEF -> GEODETIC (WGS84) — used to sanity-check groundTrack ellipsoid output
# =====================================================================
def ecef_to_geodetic_wgs84(x, y, z, a=6378137.0, f=1.0/298.257223563):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    e2 = f*(2.0 - f); b = a*(1.0 - f)
    p = np.sqrt(x*x + y*y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p*(1.0 - e2))
    at_pole = p < 1e-12
    if np.any(at_pole):
        lat = np.where(at_pole, np.sign(z) * (np.pi/2.0), lat)
    for _ in range(5):
        sinphi = np.sin(lat); cosphi = np.cos(lat)
        N = a / np.sqrt(1.0 - e2*sinphi*sinphi)
        h = p / np.where(cosphi == 0.0, 1e-18, cosphi) - N
        lat = np.arctan2(z, p*(1.0 - e2*N/(N + h)))
    sinphi = np.sin(lat); cosphi = np.cos(lat)
    N = a / np.sqrt(1.0 - e2*sinphi*sinphi)
    h = p / np.where(cosphi == 0.0, 1e-18, cosphi) - N
    h = np.where(at_pole, np.abs(z) - b, h)
    lon = (lon + np.pi) % (2.0*np.pi) - np.pi
    return lon, lat, h

# =====================================================================
# TEST UTIL
# =====================================================================
def _ok(msg):   print("[PASS]", msg)
def _fail(msg): print("[FAIL]", msg)
def _rms(x):    return float(np.sqrt(np.nanmean(np.asarray(x, float)**2)))

def _assert_scalar_close(name, x, ref, tol):
    d = float(np.max(np.abs(np.asarray(x, float) - float(ref))))
    if d <= tol:
        _ok(f"{name}: |Δ| ≤ {tol:g} (got {d:.6g})"); return True
    _fail(f"{name}: |Δ| = {d:.6g} > {tol:g}"); return False

# =====================================================================
# TESTS
#   Notes:
#   * The calc-vs-local heading comparison tests are intentionally omitted here.
#     Those depend on the exact azimuth convention inside calc_gamma_and_heading,
#     which may not be the same as the ENU course used in these panels.
#   * The tests below focus on local ENU expectations, GCRF/ITRF invariance of
#     the local calculation, the ellipsoid geodetic mapping, and a simple
#     equatorial sanity check using the local finite-difference method.
# =====================================================================
def test_itrf_local_expected():
    R = float(EARTH_RADIUS) + 700e3
    lons = np.array([0.0, 90.0, 180.0, 270.0])
    lats = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    dt_s = 1.0
    d = 1000.0

    ok_all = True
    for lat_deg in lats:
        phi = np.radians(lat_deg)
        for lon_deg in lons:
            lam = np.radians(lon_deg)
            r0 = R * np.array([np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)])
            e_hat, n_hat, u_hat = _enu_basis_from_r(r0)
            dirs = {
                "Up":    +u_hat, "Down":  -u_hat,
                "East":  +e_hat, "West":  -e_hat,
                "North": +n_hat, "South": -n_hat,
            }
            for name, dhat in dirs.items():
                r1 = r0 + d * dhat
                g, h = _heading_gamma_from_pair(r0, r1, dt_s)
                tag = f"ITRF {name} @ lat={lat_deg:.1f}, lon={lon_deg:.1f}"
                if name == "Up":
                    ok_all &= _assert_scalar_close(tag+" gamma", g, +90.0, 1e-3)
                elif name == "Down":
                    ok_all &= _assert_scalar_close(tag+" gamma", g, -90.0, 1e-3)
                else:
                    if np.abs(lat_deg) >= 85.0:  # skip near poles (numerical degeneracy)
                        continue
                    ok_all &= _assert_scalar_close(tag+" gamma", g, 0.0, 1e-3)
                    h_ref = {"East": 90.0, "West": 270.0, "North": 0.0, "South": 180.0}[name]
                    ddeg = float(np.max(_ang_diff_deg(h, h_ref)))
                    if ddeg <= 1e-3:
                        _ok(f"{tag} heading |Δ| ≤ 0.001°")
                    else:
                        _fail(f"{tag} heading |Δ| {ddeg:.6f}° > 0.001°"); ok_all &= False
    return ok_all

def test_itrf_gcrf_invariance():
    R = float(EARTH_RADIUS) + 700e3
    lons = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0])
    lats = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    dt_s = 1.0
    d = 500.0
    obstime = Time(t0_str, scale="utc")

    ok_all = True
    for lat_deg in lats:
        phi = np.radians(lat_deg)
        for lon_deg in lons:
            lam = np.radians(lon_deg)
            r0 = R * np.array([np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)])
            e_hat, n_hat, u_hat = _enu_basis_from_r(r0)
            dirs = {"Up":+u_hat, "Down":-u_hat, "East":+e_hat, "West":-e_hat, "North":+n_hat, "South":-n_hat}
            for name, dhat in dirs.items():
                r1 = r0 + d * dhat
                g_itrf, h_itrf = _heading_gamma_from_pair(r0, r1, dt_s)

                r0_g = ITRS(CartesianRepresentation(*(r0 * u.m)), obstime=obstime)\
                        .transform_to(GCRS(obstime=obstime)).cartesian.xyz.to_value(u.m)
                r1_g = ITRS(CartesianRepresentation(*(r1 * u.m)), obstime=obstime)\
                        .transform_to(GCRS(obstime=obstime)).cartesian.xyz.to_value(u.m)

                g_gcrf, h_gcrf = _heading_gamma_from_pair(r0_g, r1_g, dt_s)
                tag = f"Invariance {name} @ lat={lat_deg:.1f}, lon={lon_deg:.1f}"

                ok_all &= _assert_scalar_close(tag+" gamma", g_gcrf, g_itrf, 1e-3)

                if not (np.isnan(h_itrf) or np.isnan(h_gcrf)):
                    diff = float(np.max(_ang_diff_deg(h_gcrf, h_itrf)))
                    if diff <= 0.5:
                        _ok(tag+" heading |Δ| ≤ 0.5°")
                    else:
                        _fail(tag+f" heading > 0.5° (got {diff:.3f}°)")
                        ok_all &= False
    return ok_all

def test_groundtrack_ellipsoid_consistency():
    if not SSAPY_AVAILABLE:
        print("[SKIP] groundTrack consistency — ssapy not available")
        return True

    r, v, t = ssapy_orbit(
        a=EARTH_RADIUS + 500e3, e=0.01, i=np.radians(10.0),
        pa=0.0, raan=0.0, ta=0.0, duration=(2, "hour"), freq=(2, "min"), t0=t0_str
    )
    x, y, z = groundTrack(r, t, format="cartesian")
    lon_gt, lat_gt, h_gt = groundTrack(r, t, format="geodetic")  # rad, rad, m
    lon_el, lat_el, h_el = ecef_to_geodetic_wgs84(x, y, z)

    dlon = ((lon_el - lon_gt + np.pi) % (2.0*np.pi)) - np.pi
    dlat = lat_el - lat_gt
    dh   = h_el - h_gt

    ok = True
    ok &= _assert_scalar_close("groundTrack lon(rad) max|Δ|", np.max(np.abs(dlon)), 0.0, 5e-10)
    ok &= _assert_scalar_close("groundTrack lat(rad) max|Δ|", np.max(np.abs(dlat)), 0.0, 5e-10)
    ok &= _assert_scalar_close("groundTrack h(m)   max|Δ|",   np.max(np.abs(dh)),   0.0, 1e-3)
    return ok

def test_equatorial_local_only():
    r, v, t = ssapy_orbit(
        a=EARTH_RADIUS + 400e3, e=0.0, i=0.0, pa=0.0, raan=0.0, ta=0.0,
        duration=(2, "hour"), freq=(60, "s"), t0=t0_str
    )
    r_arr = np.asarray(r, float)
    t_unx = np.asarray(t.unix if hasattr(t, "unix") else t, float)
    dt = np.diff(t_unx)
    r0 = r_arr[:-1]; r1 = r_arr[1:]
    dt = np.where(dt == 0.0, 1e-6, dt)

    g_loc, h_loc = [], []
    for i in range(len(dt)):
        g_i, h_i = _heading_gamma_from_pair(r0[i], r1[i], dt[i])
        g_loc.append(g_i); h_loc.append(h_i)
    g_loc = np.asarray(g_loc, float)
    h_loc = np.asarray(h_loc, float)

    g_rms = _rms(g_loc)
    if g_rms <= 2.5:
        _ok(f"local equatorial: gamma RMS {g_rms:.3f}° ≤ 2.5°")
    else:
        _fail(f"local equatorial: gamma RMS {g_rms:.3f}° > 2.5°"); return False

    def rms_to(val):
        d = _ang_diff_deg(h_loc, val)
        return float(np.sqrt(np.nanmean(d*d)))
    rms90  = rms_to(90.0)
    rms270 = rms_to(270.0)
    rms_ok = min(rms90, rms270)
    if rms_ok <= 10.0:
        _ok(f"local equatorial: heading RMS {rms_ok:.3f}° ≤ 10° to {{90|270}}")
    else:
        _fail(f"local equatorial: heading RMS {rms_ok:.3f}° > 10°"); return False

    alt_std = float(np.nanstd(np.linalg.norm(r_arr, axis=1) - float(EARTH_RADIUS)))
    if alt_std <= 1.5e4:
        _ok(f"local equatorial: altitude std {alt_std:.1f} m ≤ 15000 m")
        return True
    _fail(f"local equatorial: altitude std {alt_std:.1f} m > 15000 m")
    return False

# =====================================================================
# MAIN: build the dashboard + 2×3 panels + single 3D plot + run tests
# =====================================================================
if __name__ == "__main__":
    # Dashboard using the gamma/heading utility (produces map + 3D + time series)
    r, v, t = ssapy_orbit(
        a=a_m, e=e, i=np.radians(i_deg), pa=0.0, raan=0.0, ta=0.0,
        duration=duration, freq=freq, t0=t0_str
    )
    tag = f"a{int(round(a_m/1000))}km_e{e:.2f}_i{int(round(i_deg))}deg_equatorial"

    fig_dash = groundtrack_dashboard_gamma_heading([r], [t], show=False, save_path=None, fontsize=FS_BASE)
    out_dash = Path(figpath(f"dashboard_gamma_heading_{tag}")).with_suffix(".png")
    out_dash.parent.mkdir(parents=True, exist_ok=True)
    fig_dash.savefig(out_dash, dpi=220, bbox_inches="tight")
    print("Saved dashboard:", out_dash)
    plt.close(fig_dash)

    # 2×3 heading/gamma panels in ITRF and GCRF
    fig_itrf_panel = heading_gamma_panel(
        frame_label="ITRF",
        altitude_m=700e3,
        az_step_deg=GRID_AZ_STEP_DEG,
        lat_step_deg=GRID_LAT_STEP_DEG,
        d_disp_m=D_DISP_M,
        dt_s=DT_S,
        t0=t0_str,
        arrow_len_deg=ARROW_LEN_DEG,
        arrow_width=ARROW_WIDTH,
        arrow_edge_lw=ARROW_EDGE_LW,
        headlength=HEADLENGTH,
        headwidth=HEADWIDTH,
        equal_onscreen_length=EQUAL_ONSCREEN_LEN,
        draw_radial_markers=DRAW_RADIAL_MARKERS,
        show_points=True,
    )
    out_itrf_panel = Path(figpath("heading_gamma_panel_ITRF")).with_suffix(".png")
    out_itrf_panel.parent.mkdir(parents=True, exist_ok=True)
    fig_itrf_panel.savefig(out_itrf_panel, dpi=240, bbox_inches="tight")
    print("Saved:", out_itrf_panel)
    plt.close(fig_itrf_panel)

    fig_gcrf_panel = heading_gamma_panel(
        frame_label="GCRF",
        altitude_m=700e3,
        az_step_deg=GRID_AZ_STEP_DEG,
        lat_step_deg=GRID_LAT_STEP_DEG,
        d_disp_m=D_DISP_M,
        dt_s=DT_S,
        t0=t0_str,
        arrow_len_deg=ARROW_LEN_DEG,
        arrow_width=ARROW_WIDTH,
        arrow_edge_lw=ARROW_EDGE_LW,
        headlength=HEADLENGTH,
        headwidth=HEADWIDTH,
        equal_onscreen_length=EQUAL_ONSCREEN_LEN,
        draw_radial_markers=DRAW_RADIAL_MARKERS,
        show_points=True,
    )
    out_gcrf_panel = Path(figpath("heading_gamma_panel_GCRF")).with_suffix(".png")
    fig_gcrf_panel.savefig(out_gcrf_panel, dpi=240, bbox_inches="tight")
    print("Saved:", out_gcrf_panel)
    plt.close(fig_gcrf_panel)

    # Single 3D plot (one figure, category-colored arrows, no gamma colorbar)
    fig3d = heading_gamma_arrows_3d_single(
        frame_label=FRAME_FOR_3D,  # "ITRF" or "GCRF"
        t0=t0_str,
        altitude_m=ALTITUDE_M,
        lon_step_deg=GRID_LON_STEP_DEG,
        lat_step_deg=GRID_LAT_STEP_DEG3,
        arrow_len_km=ARROW_LEN_KM_3D,
        title=TITLE_3D,
    )
    out3d = Path(figpath(f"arrows_3d_categories_{FRAME_FOR_3D}")).with_suffix(".png")
    out3d.parent.mkdir(parents=True, exist_ok=True)
    fig3d.savefig(out3d, dpi=220, bbox_inches="tight")
    print("Saved:", out3d)
    plt.close(fig3d)

    # Validation tests that make sense independent of calc_gamma_and_heading’s azimuth convention
    print("\n=== Running validation tests ===")
    results = []
    results.append(("ITRF local expected", test_itrf_local_expected()))
    results.append(("ITRF↔GCRF invariance", test_itrf_gcrf_invariance()))
    results.append(("groundTrack ellipsoid consistency", test_groundtrack_ellipsoid_consistency()))
    results.append(("Equatorial — LOCAL ONLY", test_equatorial_local_only()))

    n_pass = sum(1 for _, ok in results if ok)
    n_tot  = len(results)
    print(f"\nTest summary: {n_pass}/{n_tot} groups passed.")
    for name, ok in results:
        print(f" - {name}: {'PASS' if ok else 'FAIL'}")

    if n_pass != n_tot:
        raise AssertionError("One or more validation tests FAILED.")
    else:
        print("All validation tests PASSED.")
