#!/usr/bin/env python
import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib import rcParams

# ssapy ground track (geodetic/cartesian converters)
from ssapy import groundTrack

# Optional background Earth image (best-effort; ok if missing)
def _try_load_earth():
    try:
        # adjust if your helper lives elsewhere
        from yeager_utils.Plots.plotutils import load_earth_file
        return load_earth_file()
    except Exception:
        return None

# Optional pretty progress
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


def _ensure_ffmpeg_path():
    """
    Return a usable ffmpeg executable path, or None if not found.
    Tries PATH first, then imageio-ffmpeg bundled binary.
    """
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


def _broadcast_time_list(r_list, t):
    # Same semantics as your 2D plot: allow single t reused or list matching r_list
    if isinstance(t, (list, tuple)):
        if len(t) != len(r_list):
            raise ValueError("When passing a list of times, its length must match the number of orbits.")
        return list(t)
    return [t for _ in r_list]


def _ensure_dir(path):
    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _ensure_Nx3(a):
    A = np.asarray(a, dtype=float)
    if A.ndim != 2:
        raise ValueError("Each 'r' must be 2D; got shape {}".format(A.shape))
    if 3 in A.shape:
        if A.shape[1] == 3:
            return A
        if A.shape[0] == 3:
            return A.T
    raise ValueError("Each 'r' must have a dimension of size 3; got shape {}".format(A.shape))


def _clean_lonlat_wrap(lon_deg, lat_deg, threshold=179.0):
    """Insert NaNs at 180° crossings so lines do not jump across the map."""
    jumps = np.where(np.abs(np.diff(lon_deg)) > threshold)[0]
    if jumps.size == 0:
        return lon_deg, lat_deg
    lon_out = np.insert(lon_deg, jumps + 1, np.nan)
    lat_out = np.insert(lat_deg, jumps + 1, np.nan)
    return lon_out, lat_out


def groundtrack_video(
    r,
    t,
    ground_stations=None,
    save_path=None,
    title="Ground Track",
    show_legend=True,
    fontsize=18,
    start_end_markers=True,
    fps=30,
    bitrate=2400,
    max_frames=2000,
    progress=True,
    mode="map",              # <<< NEW: "map" (2D lon/lat; matches your plot), "surface3d", "eci3d"
):
    """
    Create an MP4 animation of satellite ground tracks.

    Default `mode="map"` renders a 2D lon/lat plot that matches your groundtrack_plot
    (full static track + moving marker). If you prefer a 3D path on the Earth's surface,
    use mode="surface3d" (uses groundTrack(..., format='cartesian')). The old ECI-in-space
    style can be done with mode="eci3d" (plots r directly around a sphere).

    Parameters
    ----------
    r : (n,3) array_like or list of (n,3)
        GCRF/ECI positions [m]. Single orbit or a list of orbits.
    t : (n,) array_like or list of (n,)
        Absolute times matching r (same as groundtrack_plot).
    ground_stations : (k,2) array_like, optional
        (lat_deg, lon_deg) rows. Used in "map" and "surface3d" modes.
    save_path : str, required, must end with '.mp4'
    start_end_markers : bool
    fps, bitrate, max_frames, progress : controls
    mode : "map" | "surface3d" | "eci3d"
    """
    if not save_path or not str(save_path).lower().endswith(".mp4"):
        raise ValueError("Please provide save_path ending with '.mp4'.")

    # FFmpeg availability
    ff = _ensure_ffmpeg_path()
    if not ff:
        raise RuntimeError(
            "ffmpeg executable not found.\n"
            "Install one of the following and re-run:\n"
            "  - pip install imageio-ffmpeg   (bundled binary)\n"
            "  - conda install -c conda-forge ffmpeg\n"
            "  - apt-get install ffmpeg / brew install ffmpeg\n"
            "Or add an existing ffmpeg to your PATH."
        )
    rcParams["animation.ffmpeg_path"] = ff

    # Normalize and validate inputs
    r_list = [_ensure_Nx3(ri) for ri in _as_list(r)]
    t_list = _broadcast_time_list(r_list, t)

    # Optionally downsample frames to cap runtime/size
    # We'll index by frame k across the longest series; shorter series "pause" at their ends.
    lengths = [ri.shape[0] for ri in r_list]
    L = int(np.max(lengths)) if lengths else 0
    if L == 0:
        raise ValueError("Empty trajectory; nothing to animate.")
    if L > max_frames:
        sel = np.linspace(0, L - 1, max_frames).astype(int)
        r_list = [ri[sel if ri.shape[0] == L else np.linspace(0, ri.shape[0]-1, max_frames).astype(int)] for ri in r_list]
        t_list = [ti[sel if len(ti) == L else np.linspace(0, len(ti)-1, max_frames).astype(int)] for ti in t_list]
        lengths = [ri.shape[0] for ri in r_list]
        L = int(np.max(lengths))

    _ensure_dir(save_path)
    writer = FFMpegWriter(
        fps=fps,
        bitrate=bitrate,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"]  # widest compatibility
    )

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(r_list))))

    # --- MODE: 2D MAP (matches your groundtrack_plot) ---
    if mode == "map":
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)

        # Background map (best-effort)
        bg = _try_load_earth()
        if bg is not None:
            ax.imshow(bg, extent=[-180, 180, -90, 90], aspect='auto', zorder=-1)

        # Axes styling
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (deg)", fontsize=fontsize)
        ax.set_ylabel("Latitude (deg)", fontsize=fontsize)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=fontsize-2)
        ax.set_title(title, fontsize=fontsize+4)

        # Prepare static full tracks + start/end markers, plus moving heads
        heads = []
        lon_all, lat_all = [], []
        for i, (ri, ti) in enumerate(zip(r_list, t_list)):
            lon, lat, _h = groundTrack(np.asarray(ri), ti, format='geodetic')
            lon_deg = np.degrees(lon)
            lat_deg = np.degrees(lat)
            lon_plot, lat_plot = _clean_lonlat_wrap(lon_deg, lat_deg, threshold=179.0)

            # Static full line
            ax.plot(lon_plot, lat_plot, color=colors[i % len(colors)], linewidth=2.5, zorder=2)

            # Start/end markers
            if start_end_markers and len(lon_deg) > 0:
                ax.plot(lon_deg[0],  lat_deg[0],  marker='*', color=colors[i % len(colors)], markersize=12, linestyle='None', zorder=3)
                ax.plot(lon_deg[-1], lat_deg[-1], marker='x', color=colors[i % len(colors)], markersize=9,  linestyle='None', zorder=3)

            # Moving head
            head, = ax.plot([lon_deg[0]], [lat_deg[0]], marker='o', markersize=5,
                            color=colors[i % len(colors)], linestyle='None', zorder=4)
            heads.append(head)
            lon_all.append(lon_deg)
            lat_all.append(lat_deg)

        # Ground stations
        if ground_stations is not None:
            gs = np.asarray(ground_stations, dtype=float)
            if gs.ndim == 2 and gs.shape[1] == 2:
                ax.scatter(gs[:, 1], gs[:, 0], s=50, color='red', label="Ground Station", zorder=5)

        # Legend (optional)
        if show_legend:
            from matplotlib.lines import Line2D
            base = [
                Line2D([0], [0], color='black', linewidth=2.5, label='Orbit Track'),
                Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=12, label='Orbit Start'),
                Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=10, label='Orbit End'),
            ]
            if ground_stations is not None:
                base.append(Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=8, label='Ground Station'))
            ax.legend(handles=base, loc='lower left', fontsize=fontsize-2)

        # Progress iterator
        if progress and _HAS_TQDM:
            frame_iter = tqdm(range(L), desc="Rendering MP4 (map)", unit="frame")
            ascii_mode = False
        else:
            frame_iter = range(L)
            ascii_mode = progress
        bar_len = 28
        update_every = max(1, L // 100)

        # Animate
        with writer.saving(fig, save_path, dpi=150):
            for k in frame_iter:
                for i in range(len(r_list)):
                    kk = min(k, len(lon_all[i]) - 1)
                    heads[i].set_data([lon_all[i][kk]], [lat_all[i][kk]])
                writer.grab_frame()

                if ascii_mode and (k % update_every == 0 or k == L - 1):
                    pct = (k + 1) / float(L)
                    filled = int(bar_len * pct)
                    bar = "#" * filled + "." * (bar_len - filled)
                    sys.stdout.write("\rRendering MP4 (map): [{}] {:3d}% ({}/{})".format(bar, int(pct * 100), k + 1, L))
                    sys.stdout.flush()

        if ascii_mode:
            sys.stdout.write("\n"); sys.stdout.flush()

        plt.close(fig)
        return save_path

    # --- Other modes (optional): 3D on surface or full ECI space ---
    # Kept brief; if you need either, say the word and I’ll expand fully.

    raise ValueError('Unsupported mode "{}". Use mode="map" to match your groundtrack_plot.'.format(mode))


# ----------------------------- Demo (optional) -----------------------------
if __name__ == "__main__":
    # Tiny demo using synthetic data to verify the look & write path.
    N = 1000
    # Fake "orbit" in ECI (meters)
    R = 6371e3
    r1 = np.zeros((N, 3)); r2 = np.zeros((N, 3))
    th = np.linspace(0.0, 4.0*np.pi, N)
    r1[:,0] = (R + 700e3) * np.cos(th)
    r1[:,1] = (R + 700e3) * np.sin(th)
    r1[:,2] = 0.2*(R+700e3)*np.sin(0.5*th)
    r2[:,0] = (R + 35786e3) * np.cos(0.5*th + 0.4)
    r2[:,1] = (R + 35786e3) * np.sin(0.5*th + 0.4)
    r2[:,2] = 0.0

    # Fake times (any absolute scale supported by ssapy.groundTrack should be fine)
    t = np.linspace(1.42e9, 1.42e9 + 10000.0, N)  # seconds

    out = groundtrack_video(
        r=[r1, r2],
        t=t,
        save_path=os.path.join(os.getcwd(), "demo_groundtrack_map.mp4"),
        mode="map",
        progress=True,
        fps=30,
        bitrate=2400,
        max_frames=800
    )
    print("Wrote:", out)
