import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from matplotlib.animation import FFMpegWriter
from yeager_utils import EARTH_RADIUS


def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


def _broadcast_time_list(r_list, t):
    # Accepted for API compatibility; not used for rendering
    if isinstance(t, (list, tuple)):
        if len(t) != len(r_list):
            raise ValueError("When passing a list of times, its length must match the number of orbits.")
        return list(t)
    return [t for _ in r_list]


def _ensure_dir(path):
    d = os.path.dirname(path)
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


def groundtrack_video(
    r,
    t,
    ground_stations=None,
    save_path=None,
    title="Ground Track",
    show_legend=True,   # kept for API compatibility; unused here
    fontsize=18,        # kept for API compatibility; unused here
    start_end_markers=True,
):
    """
    Create a 3D MP4 animation of orbits around a semi-transparent Earth sphere.

    Parameters
    ----------
    r : (n,3) array_like or list of (n,3)
        ECI/GCRF positions [m]. Single orbit or list of orbits.
    t : (n,) array_like or list of (n,)
        Accepted for API compatibility; not used for rendering timing.
    ground_stations : (k,2) array_like, optional
        (lat_deg, lon_deg) rows.
    save_path : str
        Must end with '.mp4'. The video will be written here.
    title, show_legend, fontsize : kept for compatibility; not used.
    start_end_markers : bool
        Draw start '*' and end 'x' markers.

    Returns
    -------
    str
        The output video path.
    """
    if not save_path or not str(save_path).lower().endswith(".mp4"):
        raise ValueError("Please provide save_path ending with '.mp4'.")

    # normalize inputs
    r_list = [_ensure_Nx3(ri) for ri in _as_list(r)]
    _ = _broadcast_time_list(r_list, t)  # not used, but validates shape

    # optional downsample to cap runtime/size
    max_frames = 1500  # tweak as needed
    r_list_ds = []
    lengths = [ri.shape[0] for ri in r_list]
    for ri in r_list:
        n = ri.shape[0]
        if n > max_frames:
            idx = np.linspace(0, n - 1, max_frames).astype(int)
            r_list_ds.append(ri[idx])
        else:
            r_list_ds.append(ri)
    r_list = r_list_ds
    L = int(np.max([ri.shape[0] for ri in r_list])) if r_list else 0
    if L == 0:
        raise ValueError("Empty trajectory; nothing to animate.")

    # limits
    traj_max = EARTH_RADIUS
    for R in r_list:
        if R.size:
            traj_max = max(traj_max, float(np.max(np.linalg.norm(R, axis=1))))
    lim_m = 1.05 * max(traj_max, EARTH_RADIUS)
    lim_km = lim_m / 1e3

    # figure/axes
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.tight_layout()

    # Earth sphere (semi-transparent), in km
    n_theta, n_phi = 48, 96
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=True)
    TT, PP = np.meshgrid(theta, phi, indexing="ij")
    Xs = EARTH_RADIUS * np.sin(TT) * np.cos(PP) / 1e3
    Ys = EARTH_RADIUS * np.sin(TT) * np.sin(PP) / 1e3
    Zs = EARTH_RADIUS * np.cos(TT) / 1e3
    ax.plot_surface(Xs, Ys, Zs, linewidth=0, alpha=0.25)

    # colors, lines, and moving heads
    colors = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    lines, heads = [], []
    for i, R in enumerate(r_list):
        line, = ax.plot([], [], [], lw=1.5, color=colors[i % 10])
        head, = ax.plot([], [], [], marker="o", markersize=4, color=colors[i % 10])
        lines.append(line)
        heads.append(head)

        if start_end_markers and R.shape[0] > 0:
            ax.scatter(R[0,0]/1e3,  R[0,1]/1e3,  R[0,2]/1e3,  marker="*", s=80, color=colors[i % 10])
            ax.scatter(R[-1,0]/1e3, R[-1,1]/1e3, R[-1,2]/1e3, marker="x", s=60, color=colors[i % 10])

    # ground stations (lat, lon deg)
    if ground_stations is not None:
        gs = np.asarray(ground_stations, dtype=float)
        if gs.ndim == 2 and gs.shape[1] == 2 and gs.size > 0:
            lat = np.radians(gs[:, 0])
            lon = np.radians(gs[:, 1])
            gx = EARTH_RADIUS * np.cos(lat) * np.cos(lon) / 1e3
            gy = EARTH_RADIUS * np.cos(lat) * np.sin(lon) / 1e3
            gz = EARTH_RADIUS * np.sin(lat) / 1e3
            ax.scatter(gx, gy, gz, s=12, color="red")

    # axes limits/labels/camera
    ax.set_xlim(-lim_km, lim_km)
    ax.set_ylim(-lim_km, lim_km)
    ax.set_zlim(-lim_km, lim_km)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.view_init(elev=20.0, azim=-60.0)

    # prepare output
    _ensure_dir(save_path)
    writer = FFMpegWriter(fps=30, bitrate=2400)

    # animate: 1 frame per sample; shorter orbits pause when they finish
    with writer.saving(fig, save_path, dpi=150):
        for k in range(L):
            for i, R in enumerate(r_list):
                kk = min(k, R.shape[0] - 1)
                xkm = R[:kk+1, 0] / 1e3
                ykm = R[:kk+1, 1] / 1e3
                zkm = R[:kk+1, 2] / 1e3
                lines[i].set_data(xkm, ykm)
                lines[i].set_3d_properties(zkm)
                heads[i].set_data([xkm[-1]], [ykm[-1]])
                heads[i].set_3d_properties([zkm[-1]])
            writer.grab_frame()

    plt.close(fig)
    return save_path
