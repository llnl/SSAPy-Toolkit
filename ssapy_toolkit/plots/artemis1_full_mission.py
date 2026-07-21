"""
ssapy_toolkit/plots/artemis1_full_mission.py
-----------------------------------------------
Plots the COMPLETE real Artemis I trajectory using NASA's actual
post-flight ephemeris (the CCSDS OEM file NASA published via AROW,
the Artemis Real-time Orbit Website) -- not Horizons, not an
approximation. This supersedes artemis1_horizons_plot.py, whose
Horizons-sourced data only covered the outbound leg (Horizons' own
archive for target -1023 appears to stop around Nov 22, 2022, well
short of the real mission's Dec 11 end, regardless of date range or
center coordinate -- confirmed by direct testing).

Data source
-----------
NASA, "Track NASA's Artemis I Mission in Real Time" (AROW):
    https://www.nasa.gov/missions/artemis/orion/track-nasas-artemis-i-mission-in-real-time/
Direct file:
    https://www.nasa.gov/wp-content/uploads/2022/08/post-tli-orion-asflown-20221213-eph-oem.zip
File: Post_TLI_Orion_AsFlown_20221213_EPH_OEM.asc (CCSDS OEM v2.0)
Coverage: 2022-11-16T08:44:51 UTC (Orion/ICPS separation) through
          2022-12-11T17:19:52 UTC (just before Entry Interface) --
          the ENTIRE mission, both outbound and return legs, one
          continuous file.

Why this is simpler than the Horizons-based loader
----------------------------------------------------
1. REF_FRAME = EME2000 in the file header. EME2000 (Earth Mean
   Equator and Equinox of J2000) is the same equatorial/ICRF-aligned
   convention already used as "GCRF" throughout this toolkit -- NO
   ecliptic-to-equatorial rotation is needed this time (unlike the
   Horizons data, which was in the ecliptic frame).
2. CENTER_NAME = EARTH. This data is genuinely Earth-centered, so it
   uses moon_plot_3d's DEFAULT r_frame='gcrf' path (the same one
   OrbitalState-propagated trajectories use) -- no moon_centered
   special-casing needed.
3. One continuous file, no gap to patch over with a second query.

Run:
    conda activate myenv
    cd C:/Users/diamond10/SSAPy-Toolkit
    python -m ssapy_toolkit.plots.artemis1_full_mission
"""

import sys
import pathlib
import datetime
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ssapy_toolkit.plots.moon_plot_3d import moon_plot_3d

try:
    from ssapy_toolkit.plots.figpath import FIG_DIR
    OUT_DIR = pathlib.Path(FIG_DIR)
except Exception:
    OUT_DIR = pathlib.Path.home() / "yu_figures" / "demo_gallery" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Path to the real OEM ephemeris file. Place
# Post_TLI_Orion_AsFlown_20221213_EPH_OEM.asc in the same directory as
# this script (or update this path).
OEM_PATH = pathlib.Path(__file__).resolve().parent / "Post_TLI_Orion_AsFlown_20221213_EPH_OEM.asc"

GPS_EPOCH = datetime.datetime(1980, 1, 6, tzinfo=datetime.timezone.utc)
# GPS-UTC offset: 18s, stable since the last leap second (2017-01-01)
# through at least mid-2026 (no new leap second added since) -- valid
# for this Nov-Dec 2022 data.
GPS_UTC_OFFSET_S = 18.0


def _utc_iso_to_gps(iso_str: str) -> float:
    dt = datetime.datetime.fromisoformat(iso_str).replace(tzinfo=datetime.timezone.utc)
    return (dt - GPS_EPOCH).total_seconds() + GPS_UTC_OFFSET_S


def parse_oem_file(path: pathlib.Path, downsample: int = 12,
                    start_utc: str = None, stop_utc: str = None):
    """Parse the CCSDS OEM ephemeris file into (r_m, t_gps) arrays.

    Parameters
    ----------
    path : pathlib.Path
        Path to the .asc OEM file.
    downsample : int
        Keep every Nth data row, AFTER any start_utc/stop_utc window has
        already narrowed the data. This matters: applying downsample to
        the full 25-day file and THEN windowing to a short phase (e.g.
        the ~10-day DRO segment) would leave that close-up looking just
        as sparse as the full-mission view. Filtering first means a
        tight time window still gets full point density.
    start_utc, stop_utc : str, optional
        ISO8601 UTC bounds (e.g. "2022-11-21T00:00:00") to select only
        a portion of the mission -- e.g. just the DRO phase, for a
        close-up view. If omitted, the entire file is used.

    Returns
    -------
    r_m   : ndarray (N, 3) -- position, meters, Earth-centered EME2000
            (same convention as GCRF elsewhere in this toolkit).
    t_gps : ndarray (N,)   -- GPS seconds.
    """
    times, xs, ys, zs = [], [], [], []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("CCSDS", "COMMENT", "CREATION",
                                              "ORIGINATOR", "META_", "OBJECT_",
                                              "CENTER_", "REF_FRAME", "TIME_SYSTEM",
                                              "START_TIME", "USEABLE_", "STOP_TIME")):
                continue
            parts = line.split()
            if len(parts) != 7:
                continue  # skip any malformed/unexpected lines defensively
            ts, x, y, z, vx, vy, vz = parts
            if start_utc is not None and ts < start_utc:
                continue
            if stop_utc is not None and ts > stop_utc:
                continue
            times.append(ts)
            xs.append(float(x))
            ys.append(float(y))
            zs.append(float(z))

    times = times[::downsample]
    xs = xs[::downsample]
    ys = ys[::downsample]
    zs = zs[::downsample]

    t_gps = np.array([_utc_iso_to_gps(t) for t in times])
    r_km = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=1)
    r_m = r_km * 1e3   # km -> meters

    return r_m, t_gps


if __name__ == "__main__":
    if not OEM_PATH.exists():
        print(f"[artemis1_full_mission] OEM file not found at {OEM_PATH}")
        print("Download it from:")
        print("  https://www.nasa.gov/wp-content/uploads/2022/08/"
              "post-tli-orion-asflown-20221213-eph-oem.zip")
        print(f"Unzip it and place the .asc file at: {OEM_PATH}")
        sys.exit(1)

    # ── Full mission ────────────────────────────────────────────────────────
    r_m, t_gps = parse_oem_file(OEM_PATH, downsample=12)
    print(f"Parsed {len(t_gps)} points from the real Artemis I OEM ephemeris "
          f"(downsampled from the full file)")
    print(f"  Mission span: {(t_gps[-1]-t_gps[0])/86400:.2f} days")
    print(f"  Distance from Earth: {np.linalg.norm(r_m, axis=1).min()/1e3:.0f} "
          f"to {np.linalg.norm(r_m, axis=1).max()/1e3:.0f} km")

    fig, ax = moon_plot_3d(
        r=r_m, t=t_gps,
        r_frame='gcrf',   # this data is genuinely Earth-centered -- default path
        shade_ambient=0.22, shade_diffuse=0.78,
        show_earth=True,     # force Earth visible regardless of zoom level
        show_lagrange=False, # disabled pending review of lagrange_points_lunar_fixed_frame's
                              # source -- can't yet confirm whether its positions are wrong
                              # or just a different (possibly valid) static-snapshot convention
        title="Artemis I -- COMPLETE real trajectory (NASA AROW OEM, "
              "Nov 16 - Dec 11 2022)",
        save_path=str(OUT_DIR / "artemis1_full_mission.jpg"),
    )
    print(f"Saved -> {OUT_DIR / 'artemis1_full_mission.jpg'}")

    # NOTE: a separate DRO-only close-up window was tried previously and
    # removed. Zooming in tightly enough to clearly resolve the DRO loops
    # necessarily excludes Earth and the departure/arrival legs from the
    # frame -- those two goals are in direct tension with a single static
    # view. If a closer look at just the DRO phase is wanted later, it
    # should be a clearly-labeled separate "DRO detail, no Earth/transit
    # context" plot rather than presented as equivalent to this one.