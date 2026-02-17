from yeager_utils import get_lunar_rv, Time, figpath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

import ssapy
from ssapy.propagator import default_numerical


# ---------- helpers ----------
def _set_axes_equal_3d(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))


def solve_collinear_l1_x(mu, tol=1e-12, max_iter=50):
    """
    Solve for the x-location of the Earth-Moon L1 point in the rotating CR3BP
    barycentric frame.

    Conventions:
      Earth at x = -mu
      Moon  at x = 1 - mu
      L1 is between them: x in (0, 1 - mu)

    Returns:
      x_l1 (float): L1 x-coordinate in nondimensional units where Earth-Moon
                    distance = 1.
    """
    x = 1.0 - mu - (mu / 3.0) ** (1.0 / 3.0)

    for _ in range(max_iter):
        r1 = x + mu
        r2 = x - (1.0 - mu)

        a1 = (1.0 - mu) / np.abs(r1) ** 3
        a2 = mu / np.abs(r2) ** 3

        f = x - a1 * r1 - a2 * r2

        da1 = -3.0 * (1.0 - mu) / np.abs(r1) ** 5
        da2 = -3.0 * mu / np.abs(r2) ** 5
        fp = 1.0 - (da1 * r1 * r1 + a1) - (da2 * r2 * r2 + a2)

        step = f / fp
        x_new = x - step

        if np.abs(step) < tol:
            return float(x_new)

        x = x_new

    return float(x)


def _propagate_with_fallback(orbit, t_frames):
    """
    Try to propagate across all requested frame times.
    If propagation fails (ends early), retry with shorter spans and return the
    longest successful prefix.
    """
    prop = default_numerical()

    n = len(t_frames)
    min_n = max(2, int(0.05 * n))
    k = n

    while k >= min_n:
        try:
            times = Time(t_frames[:k], format="gps")
            r_sc_f, v_sc_f = ssapy.rv(orbit=orbit, time=times, propagator=prop)
            r_sc_f = np.array(r_sc_f, dtype=float).reshape((-1, 3))
            return r_sc_f, t_frames[:k]
        except Exception:
            k = int(0.9 * k)

    times = Time(t_frames[:2], format="gps")
    r_sc_f, v_sc_f = ssapy.rv(orbit=orbit, time=times, propagator=prop)
    r_sc_f = np.array(r_sc_f, dtype=float).reshape((-1, 3))
    return r_sc_f, t_frames[:2]


# ---------- demo video ----------
def orbit_moon_video_demo(
    t0="2024-01-01",
    duration_days=365.0,
    fps=30,
    seconds_per_frame=3 * 3600,  # 3 hours of sim time per frame
    trail_len=200,  # points of trail to show
    out_name="tests/demo_orbit_moon_video.mp4",
    save_gif=False,
):
    # ---- time grid for frames ----
    t0_gps = float(np.array(Time(t0).gps, dtype=float).ravel()[0])
    dt_frame = float(seconds_per_frame)
    n_frames = int(np.ceil((duration_days * 86400.0) / dt_frame)) + 1
    t_frames = t0_gps + dt_frame * np.arange(n_frames, dtype=float)

    # ---- Moon ephemeris at frame times ----
    r_moon, v_moon = get_lunar_rv(t_frames)  # expected (N,3) arrays
    r_moon = np.array(r_moon, dtype=float).reshape((-1, 3))
    v_moon = np.array(v_moon, dtype=float).reshape((-1, 3))

    # ---- spacecraft initial state: near Earth-Moon L1 with DRO-like velocity ----
    rm0 = r_moon[0]
    vm0 = v_moon[0]

    r_em = np.linalg.norm(rm0)
    x_hat = rm0 / r_em

    h_vec = np.cross(rm0, vm0)
    h_norm = np.linalg.norm(h_vec)
    if h_norm == 0.0:
        k_hat = np.array([0.0, 0.0, 1.0])
    else:
        k_hat = h_vec / h_norm

    y_hat = np.cross(k_hat, x_hat)
    y_norm = np.linalg.norm(y_hat)
    if y_norm == 0.0:
        y_hat = np.array([0.0, 1.0, 0.0])
    else:
        y_hat = y_hat / y_norm

    mu = 0.0121505856
    x_l1 = solve_collinear_l1_x(mu)
    d_l1_from_moon = (1.0 - mu) - x_l1  # nondim distance Moon->L1

    r_l1 = rm0 - (d_l1_from_moon * r_em) * x_hat

    y_offset = 70_000e3
    z_offset = 2_000e3
    r0 = r_l1 + y_offset * y_hat + z_offset * k_hat

    omega = h_norm / (r_em * r_em)
    omega_vec = omega * k_hat

    r_rel_m = r0 - rm0
    v_rel = -np.cross(omega_vec, r_rel_m)

    dro_speed_scale = 0.95
    v_rel = dro_speed_scale * v_rel

    radial_nudge = 5.0
    v0 = vm0 + v_rel + radial_nudge * x_hat

    # ---- propagate spacecraft orbit using SSAPy SciPyPropagator (with fallback) ----
    orbit = ssapy.Orbit(r=r0, v=v0, t=t0_gps)
    r_sc_f, t_frames_ok = _propagate_with_fallback(orbit, t_frames)

    # Trim Moon series to match the successful propagation length
    n_ok = len(t_frames_ok)
    r_moon = r_moon[:n_ok, :]

    # ---- scene scale ----
    max_norm = np.max(np.linalg.norm(np.vstack([r_moon, r_sc_f]), axis=1))
    lim = 1.10 * max_norm

    # ---- matplotlib setup ----
    fig = plt.figure(figsize=(10, 10), dpi=160)
    fig.patch.set_facecolor("black")
    fig.suptitle("Initialize L1 Lunar Orbit", color="white")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor((0, 0, 0, 0))
        axis.pane.set_facecolor((0, 0, 0, 0))
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # ---- Earth sphere ----
    R_E = 6378.1363e3
    u = np.linspace(0.0, 2.0 * np.pi, 140)
    v = np.linspace(0.0, np.pi, 70)
    x = R_E * np.outer(np.cos(u), np.sin(v))
    y = R_E * np.outer(np.sin(u), np.sin(v))
    z = R_E * np.outer(np.ones_like(u), np.cos(v))

    shade = (z - z.min()) / (z.max() - z.min() + 1e-12)
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        facecolors=plt.cm.Blues(0.25 + 0.65 * shade),
        alpha=0.95,
    )

    th = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.plot(
        R_E * np.cos(th),
        R_E * np.sin(th),
        0.0 * th,
        linewidth=0.8,
        alpha=0.25,
    )

    # ---- artists ----
    moon_pt, = ax.plot([], [], [], marker="o", markersize=6, linewidth=0)
    sc_pt, = ax.plot([], [], [], marker="o", markersize=4, linewidth=0)

    moon_trail, = ax.plot([], [], [], linewidth=1.2, alpha=0.35)
    sc_trail, = ax.plot([], [], [], linewidth=1.6, alpha=0.55)

    moon_glow, = ax.plot(
        [],
        [],
        [],
        marker="o",
        markersize=16,
        linewidth=0,
        alpha=0.10,
    )
    sc_glow, = ax.plot(
        [],
        [],
        [],
        marker="o",
        markersize=12,
        linewidth=0,
        alpha=0.10,
    )

    title = ax.text2D(0.03, 0.92, "", transform=ax.transAxes)
    title.set_color("white")

    moon_pt.set_color("0.85")
    moon_glow.set_color("0.85")
    moon_trail.set_color("0.75")

    sc_pt.set_color("cyan")
    sc_glow.set_color("cyan")
    sc_trail.set_color("cyan")

    _set_axes_equal_3d(ax, lim)

    # ---- animation ----
    def init():
        moon_pt.set_data([], [])
        moon_pt.set_3d_properties([])
        sc_pt.set_data([], [])
        sc_pt.set_3d_properties([])
        moon_trail.set_data([], [])
        moon_trail.set_3d_properties([])
        sc_trail.set_data([], [])
        sc_trail.set_3d_properties([])
        moon_glow.set_data([], [])
        moon_glow.set_3d_properties([])
        sc_glow.set_data([], [])
        sc_glow.set_3d_properties([])
        title.set_text("")
        return (
            moon_pt,
            sc_pt,
            moon_trail,
            sc_trail,
            moon_glow,
            sc_glow,
            title,
        )

    def update(i):
        rm = r_moon[i]
        rs = r_sc_f[i]

        moon_pt.set_data([rm[0]], [rm[1]])
        moon_pt.set_3d_properties([rm[2]])
        sc_pt.set_data([rs[0]], [rs[1]])
        sc_pt.set_3d_properties([rs[2]])

        moon_glow.set_data([rm[0]], [rm[1]])
        moon_glow.set_3d_properties([rm[2]])
        sc_glow.set_data([rs[0]], [rs[1]])
        sc_glow.set_3d_properties([rs[2]])

        j0 = max(0, i - trail_len)
        rm_tr = r_moon[j0: i + 1]
        rs_tr = r_sc_f[j0: i + 1]

        moon_trail.set_data(rm_tr[:, 0], rm_tr[:, 1])
        moon_trail.set_3d_properties(rm_tr[:, 2])
        sc_trail.set_data(rs_tr[:, 0], rs_tr[:, 1])
        sc_trail.set_3d_properties(rs_tr[:, 2])

        az = (0.35 * i) % 360.0
        denom = max(1, (len(t_frames_ok) - 1))
        el = 18.0 + 8.0 * np.sin(2.0 * np.pi * i / denom)
        ax.view_init(elev=float(el), azim=float(az))

        days = (t_frames_ok[i] - t_frames_ok[0]) / 86400.0
        title.set_text(f"t0 = {t0}   +{days:6.2f} days")

        return (
            moon_pt,
            sc_pt,
            moon_trail,
            sc_trail,
            moon_glow,
            sc_glow,
            title,
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=len(t_frames_ok),
        init_func=init,
        blit=False,
        interval=1000.0 / fps,
    )

    # ---- save ----
    out_mp4 = figpath(out_name)
    writer = FFMpegWriter(fps=fps, bitrate=8000)
    ani.save(out_mp4, writer=writer)
    print(f"Saved MP4: {out_mp4}")

    if save_gif:
        out_gif = figpath(out_name.replace(".mp4", ".gif"))
        ani.save(out_gif, writer=PillowWriter(fps=fps))
        print(f"Saved GIF: {out_gif}")

    plt.close(fig)


if __name__ == "__main__":
    orbit_moon_video_demo(
        t0="2024-01-01",
        duration_days=365.0,
        fps=30,
        seconds_per_frame=3 * 3600,
        trail_len=220,
        out_name="tests/demo_orbit_moon_video.mp4",
        save_gif=True,
    )
    print("VIDEO DEMO DONE.")
