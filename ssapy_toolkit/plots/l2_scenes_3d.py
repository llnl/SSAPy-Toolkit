"""
Two full-bleed 3D cislunar scenes, no side panels.

  1. hero      Earth + Moon in the synodic frame, all selected orbits
  2. closeup   the Moon alone, a handful of Moon-bound orbits

Both use the toolkit's own pieces: ``starfield.add_starfield`` for the sky
(precession-corrected, real HYG photometry), ``bodies3d`` for texture-mapped
Sun-lit bodies, and the baked LOLA albedo when it is on the texture path.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cislunar_families_plot as C
import bodies3d as b3

BG, INK, GRAY, SLATE = C.BG, C.INK, C.GRAY, C.SLATE
YELLOW, TURQ = C.YELLOW, C.TURQ
LD, R_EARTH, R_MOON = C.LD, C.R_EARTH, C.R_MOON


def _sky(ax, half, elev, azim, mag_limit=6.2, milky_way=True):
    """Toolkit starfield behind the scene; degrade quietly if unavailable."""
    try:
        from ssapy_toolkit.plots.starfield import add_starfield
        add_starfield(ax, half, elev=elev, azim=azim, mag_limit=mag_limit,
                      show_milky_way=milky_way, frame="gcrf")
        return True
    except Exception as ex:
        print(f"[scenes] starfield unavailable ({ex})")
        return False


def _axes(fig, rect, elev, azim, lim, aspect):
    ax = fig.add_axes(rect, projection="3d", facecolor=BG)
    ax.set_xlim(lim[0]); ax.set_ylim(lim[1]); ax.set_zlim(lim[2])
    ax.set_box_aspect(aspect)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()                      # full-bleed: no panes, no ticks
    return ax


def hero(tracks, fams, out="l2_hero_3d.png", frame_index=720, trail_days=24.0,
         stride_hours=6, body_scale=8.0, elev=22.0, azim=-58.0, dpi=180,
         envelope_alpha=0.16):
    """Earth-Moon synodic scene, single 3D view."""
    C.body_textures()
    rm = C.moon_gcrf_hourly()
    sun = C.sun_synodic(rm, C.sun_hour_facing(rm, elev=elev, azim=azim))
    L = C.lagrange_points()

    m = tracks[0].shape[0]
    i1 = min(int(frame_index), m - 1)
    tail = max(4, int(trail_days * 24 / stride_hours))
    i0, ihot = max(0, i1 - tail), max(0, i1 - tail // 4)

    fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)
    ax = _axes(fig, [0.0, 0.0, 1.0, 1.0], elev, azim,
               [(-1.30, 1.55), (-1.35, 1.35), (-0.85, 0.85)], (2.85, 2.85, 1.75))
    _sky(ax, 1.6, elev, azim)

    for S, f in zip(tracks, fams):
        col = C.FAMILY_COLOR[f]
        b = 1.7 if f == "near-Moon" else 1.0
        ax.plot(S[:, 0], S[:, 1], S[:, 2], lw=0.35 * b, alpha=envelope_alpha,
                color=col)
        ax.plot(S[i0:i1, 0], S[i0:i1, 1], S[i0:i1, 2], lw=1.1 * b, alpha=0.55,
                color=col)
        ax.plot(S[ihot:i1, 0], S[ihot:i1, 1], S[ihot:i1, 2], lw=2.0 * b,
                alpha=0.98, color=col)
        ax.scatter([S[i1 - 1, 0]], [S[i1 - 1, 1]], [S[i1 - 1, 2]],
                   s=14 * b, c=col, depthshade=False, zorder=9)

    C._draw_bodies_3d(ax, L, sun=sun, body_scale=body_scale,
                      body_n=(120, 96), labels=False)
    for nm, dx in (("L1", -0.11), ("L2", 0.11)):
        p = L[nm]
        ax.scatter([p[0]], [0], [0], s=95, marker="+", c=YELLOW, lw=2.0,
                   depthshade=False, zorder=10)
        ax.text(p[0] + dx, 0.0, 0.10, nm, color=YELLOW, fontsize=13,
                fontweight="bold")
    ax.text(0.0, 0.13, 0.19, "Earth", color=INK, fontsize=14)
    ax.text(1.0, 0.08, 0.24, "Moon", color=GRAY, fontsize=13)

    _caption(fig, tracks, fams, i1, tail, stride_hours, m, body_scale)
    fig.savefig(out, dpi=dpi, facecolor=BG)
    plt.close(fig)
    print("wrote", out)
    return out


def closeup(tracks, fams, ids, out="l2_moon_closeup_3d.png", n_orbits=3,
            days=30.0, stride_hours=1, body_scale=4.0, elev=16.0, azim=-64.0,
            dpi=180):
    """The Moon alone, with a few Moon-bound orbits, in Moon-centred 10^3 km."""
    C.body_textures()
    rm = C.moon_gcrf_hourly()
    sun = C.sun_synodic(rm, C.sun_hour_facing(rm, elev=elev, azim=azim))

    # tightest orbits first: a smaller scene lets the lunar disc read
    cand = [j for j, f in enumerate(fams) if f == "near-Moon"] or list(range(len(tracks)))
    MOON0 = np.array([1.0, 0.0, 0.0])
    cand.sort(key=lambda j: float(np.median(
        np.linalg.norm(tracks[j] - MOON0, axis=1))))
    pick = cand[:n_orbits]
    # Re-read the picked orbits at full hourly cadence: at this zoom the
    # 6-hourly tracks used for the wide scene render as visible polygons.
    MOON = np.array([1.0, 0.0, 0.0])
    nh = int(days * 24)
    M = []
    for j in pick:
        r = C.fetch_orbit(int(ids[j]), "cache")[:nh]
        S = C.to_synodic(r, rm[:r.shape[0]])
        M.append((S - MOON) * LD / 1e6)                        # 10^3 km

    lim = 1.12 * max(np.abs(np.concatenate(M)).max(), 12.0)
    rm_km = R_MOON / 1e6 * body_scale

    fig = plt.figure(figsize=(14.4, 10.8), facecolor=BG)
    ax = _axes(fig, [0.0, 0.0, 1.0, 1.0], elev, azim,
               [(-lim, lim)] * 3, (1, 1, 1))
    _sky(ax, lim, elev, azim, mag_limit=6.6)

    T = C.body_textures()
    b3.add_body(ax, [0, 0, 0], rm_km, T.get("moon"), sun_dir=sun, n=200,
                zorder=6)

    cyc = [YELLOW, "#ffda00", C.ORANGE, TURQ, C.GREEN, "#b06ecb"]
    for k, S in enumerate(M):
        col = cyc[k % len(cyc)]
        ax.plot(S[:, 0], S[:, 1], S[:, 2], lw=1.15, alpha=0.92, color=col,
                zorder=8)
        ax.scatter([S[-1, 0]], [S[-1, 1]], [S[-1, 2]], s=26, c=col,
                   depthshade=False, zorder=9)

    fig.text(0.012, 0.972, "Moon-bound cislunar orbits",
             color=INK, fontsize=27, fontweight="bold", va="top")
    rng = [np.linalg.norm(S, axis=1) for S in M]
    fig.text(0.012, 0.928,
             f"{len(M)} orbits from LLNL's One Million Cislunar Orbits catalogue "
             f"(Yeager et al. 2025)  ·  first {days:.0f} days  ·  selenocentric "
             f"range {min(r.min() for r in rng)*1e3:,.0f}–"
             f"{max(r.max() for r in rng)*1e3:,.0f} km",
             color=GRAY, fontsize=13, va="top")
    h = [Line2D([], [], color=cyc[k % len(cyc)], lw=2.6,
                label=f"orb_id {int(ids[j])}") for k, j in enumerate(pick)]
    leg = fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.012, 0.048),
                     ncol=len(h), frameon=False, fontsize=12)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.988, 0.014,
             f"Moon-centred synodic axes  ·  lunar disc drawn {body_scale:g}$\\times$ "
             f"actual size, LOLA albedo  ·  starfield from SSAPy-Toolkit (HYG)",
             color=SLATE, fontsize=10, ha="right")

    fig.savefig(out, dpi=dpi, facecolor=BG)
    plt.close(fig)
    print("wrote", out)
    return out


def _caption(fig, tracks, fams, i1, tail, stride_hours, m, body_scale):
    from collections import Counter
    c = Counter(fams)
    fig.text(0.012, 0.972,
             "Cislunar orbits near the Moon and Earth$-$Moon L2",
             color=INK, fontsize=28, fontweight="bold", va="top")
    fig.text(0.012, 0.926,
             f"{len(tracks)} orbits from LLNL's One Million Cislunar Orbits "
             f"catalogue (Yeager et al. 2025)  ·  Earth$-$Moon synodic frame  ·  "
             f"day {i1*stride_hours/24:.0f} of {m*stride_hours/24:,.0f}, "
             f"{tail*stride_hours/24:.0f}-day trail over the full 6-year path",
             color=GRAY, fontsize=13, va="top")
    h = [Line2D([], [], color=C.FAMILY_COLOR[f], lw=2.8,
                label=f"{C.FAMILY_LABEL[f]}  ({c[f]})")
         for f in C.FAMILY_ORDER if c.get(f)]
    leg = fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.012, 0.012),
                     ncol=len(h), frameon=False, fontsize=12.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.988, 0.014,
             f"Earth and Moon textured and Sun-lit, drawn {body_scale:g}$\\times$ "
             f"actual size  ·  starfield from SSAPy-Toolkit (HYG, precessed)",
             color=SLATE, fontsize=10, ha="right")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="l2_selection_20.csv")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--outdir", default="/mnt/user-data/outputs")
    ap.add_argument("--closeup-orbits", type=int, default=3)
    a = ap.parse_args()

    sel = C.load_selection(a.selection, a.n)
    tracks, fams, ids = C.load_tracks(sel, 6, "cache", verbose=False)
    hero(tracks, fams, f"{a.outdir}/l2_hero_3d.png")
    closeup(tracks, fams, ids, f"{a.outdir}/l2_moon_closeup_3d.png",
            n_orbits=a.closeup_orbits)


def animate_scene(tracks, fams, out="l2_scene_6month.mp4", frames=720,
                  days_per_frame=0.25, trail_days=24.0, stride_hours=6,
                  fps=30, dpi=110, body_scale=8.0, elev=22.0, azim=-58.0,
                  sweep_deg=10.0, envelope_alpha=0.10, stars=True, env=10,
                  star_mag=4.8, body_n=(40, 32)):
    """Single-panel synodic animation: no side panels, starfield behind.

    Bodies, sky and the faint 6-year envelope are drawn once; only the comet
    trails are updated, grouped into one polyline per family so the 3D
    backend re-projects a handful of artists instead of hundreds.
    """
    from matplotlib.animation import FFMpegWriter

    C.body_textures()
    rm = C.moon_gcrf_hourly()
    sun = C.sun_synodic(rm, C.sun_hour_facing(rm, elev=elev, azim=azim))
    L = C.lagrange_points()

    m = tracks[0].shape[0]
    spf = max(1, int(round(days_per_frame * 24 / stride_hours)))
    tail = max(4, int(trail_days * 24 / stride_hours))
    frames = int(min(frames, (m - 1) / spf))

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax = _axes(fig, [0.0, 0.0, 1.0, 1.0], elev, azim,
               [(-1.30, 1.55), (-1.35, 1.35), (-0.85, 0.85)], (2.85, 2.85, 1.75))
    if stars:
        _sky(ax, 1.6, elev, azim, mag_limit=star_mag, milky_way=False)

    Tall = np.stack(tracks)
    fam_idx = {}
    for j, f in enumerate(fams):
        fam_idx.setdefault(f, []).append(j)
    fam_idx = {f: np.asarray(v) for f, v in fam_idx.items()}

    for f, idx in fam_idx.items():
        # every artist is re-projected each frame, so the static envelope is
        # decimated hard: at alpha 0.10 it only conveys the swept volume
        E = np.concatenate([np.vstack([tracks[j][::env], np.full((1, 3), np.nan)])
                            for j in idx])
        ax.plot(E[:, 0], E[:, 1], E[:, 2], lw=0.28, alpha=envelope_alpha,
                color=C.FAMILY_COLOR[f])

    C._draw_bodies_3d(ax, L, sun=sun, body_scale=body_scale,
                      body_n=body_n, labels=False)
    for nm, dx in (("L1", -0.11), ("L2", 0.11)):
        p = L[nm]
        ax.scatter([p[0]], [0], [0], s=80, marker="+", c=YELLOW, lw=1.9,
                   depthshade=False, zorder=10)
        ax.text(p[0] + dx, 0.0, 0.10, nm, color=YELLOW, fontsize=12,
                fontweight="bold")
    ax.text(0.0, 0.13, 0.19, "Earth", color=INK, fontsize=13)
    ax.text(1.0, 0.08, 0.24, "Moon", color=GRAY, fontsize=12)

    art = {}
    for f, idx in fam_idx.items():
        col = C.FAMILY_COLOR[f]
        b = 1.7 if f == "near-Moon" else 1.0
        art[f] = dict(
            dim=ax.plot([], [], [], lw=1.0 * b, alpha=0.50, color=col)[0],
            hot=ax.plot([], [], [], lw=1.9 * b, alpha=0.98, color=col)[0],
            head=ax.plot([], [], [], "o", ms=3.0 * b, color=col, ls="none")[0])

    def joined(idx, a, b):
        seg = Tall[idx, a:b]
        pad = np.full((seg.shape[0], 1, 3), np.nan)
        return np.concatenate([seg, pad], axis=1).reshape(-1, 3)

    from collections import Counter
    c = Counter(fams)
    fig.text(0.012, 0.972, "Cislunar orbits near the Moon and Earth$-$Moon L2",
             color=INK, fontsize=24, fontweight="bold", va="top")
    fig.text(0.012, 0.926,
             f"{len(tracks)} orbits from LLNL's One Million Cislunar Orbits "
             f"catalogue  ·  Earth$-$Moon synodic frame  ·  SSAPy",
             color=GRAY, fontsize=12, va="top")
    h = [Line2D([], [], color=C.FAMILY_COLOR[f], lw=2.8,
                label=f"{C.FAMILY_LABEL[f]}  ({c[f]})")
         for f in C.FAMILY_ORDER if c.get(f)]
    leg = fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.012, 0.016),
                     ncol=len(h), frameon=False, fontsize=11.5)
    for t in leg.get_texts():
        t.set_color(INK)
    clock = fig.text(0.988, 0.966, "", color=INK, fontsize=16,
                     family="monospace", ha="right", va="top")

    # Star points and thin trails on black are close to worst case for H.264
    # rate control; a fixed bitrate spends it on noise.  Use constant-quality
    # instead, and yuv444p so the 1-2 px stars keep their chroma.
    writer = FFMpegWriter(
        fps=fps, codec="libx264",
        # yuv420p + High profile: Windows Media Foundation (PowerPoint,
        # Films & TV) cannot decode 4:4:4, so 444 renders will not open there.
        extra_args=["-crf", "16", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-profile:v", "high",
                    "-level", "4.2", "-movflags", "+faststart",
                    "-x264-params", "aq-mode=3:aq-strength=1.1"],
        metadata=dict(title="Cislunar L2 corridor",
                      artist="LLNL SSAPy-Toolkit"))
    print(f"rendering {frames} frames -> {out}")
    with writer.saving(fig, out, dpi):
        for k in range(frames):
            i1 = min(m - 1, 1 + k * spf)
            i0, ih = max(0, i1 - tail), max(0, i1 - tail // 4)
            for f, idx in fam_idx.items():
                A = art[f]
                d, hh = joined(idx, i0, i1), joined(idx, ih, i1)
                p = Tall[idx, i1 - 1]
                A["dim"].set_data_3d(d[:, 0], d[:, 1], d[:, 2])
                A["hot"].set_data_3d(hh[:, 0], hh[:, 1], hh[:, 2])
                A["head"].set_data_3d(p[:, 0], p[:, 1], p[:, 2])
            ax.view_init(elev=elev,
                         azim=azim + sweep_deg * np.sin(2 * np.pi * k / frames))
            days = i1 * stride_hours / 24.0
            clock.set_text(f"T + {days:6.1f} d   {days/30.4375:5.2f} mo")
            writer.grab_frame()
            if (k + 1) % 120 == 0:
                print(f"  {k+1}/{frames}", flush=True)
    plt.close(fig)
    print("wrote", out)
    return out