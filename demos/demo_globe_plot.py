#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from astropy.time import Time
from ssapy import Orbit, rv

from ssapy_toolkit.constants import RGEO
from ssapy_toolkit.plots.figpath import figpath
from ssapy_toolkit.plots.globe_plot import globe_plot

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, make_video=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if make_video is None:
        make_video = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    t0 = Time("2025-01-01T00:00:00", scale="utc")

    orbit1 = Orbit.fromKeplerianElements(
        a=RGEO, e=0.15, i=np.radians(20.0),
        pa=np.radians(20.0), raan=np.radians(15.0), trueAnomaly=0.0, t=t0
    )
    orbit2 = Orbit.fromKeplerianElements(
        a=1.5 * RGEO, e=0.35, i=np.radians(63.4),
        pa=np.radians(120.0), raan=np.radians(210.0), trueAnomaly=np.radians(45.0), t=t0
    )

    if fast:
        duration_hr = 6.0
        dt_s = 300.0
        video_frames = 36
        fps = 10
    else:
        duration_hr = 24.0
        dt_s = 120.0
        video_frames = 120
        fps = 20

    times_gps = t0.gps + np.arange(0.0, duration_hr * 3600.0 + dt_s, dt_s)

    r1, v1 = rv(orbit1, Time(times_gps, format="gps"))
    r2, v2 = rv(orbit2, Time(times_gps, format="gps"))
    r1 = np.asarray(r1, dtype=float).reshape((-1, 3))
    r2 = np.asarray(r2, dtype=float).reshape((-1, 3))

    outputs = {
        "r1": r1,
        "r2": r2,
        "times_gps": times_gps,
    }

    if make_figures:
        out_static = Path(figpath("figures/demo_globe_plot_two_orbits"))
        if out_static.suffix == "":
            out_static = out_static.with_suffix(".png")
        out_static.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = globe_plot(
            [r1, r2],
            t=[times_gps, times_gps],
            title="Globe Plot Demo: Two Orbits",
            c="black",
            labels=["Orbit 1", "Orbit 2"],
            orbit_colors=["cyan", "magenta"],
            globe_time=Time(times_gps[0], format="gps"),
            save_path=str(out_static),
        )
        plt.close(fig)
        print("Saved:", out_static)
        outputs["static_plot"] = str(out_static)

    if make_video:
        out_mp4 = Path(figpath("figures/demo_globe_plot_animation"))
        if out_mp4.suffix == "":
            out_mp4 = out_mp4.with_suffix(".mp4")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        frame_idxs = np.linspace(0, len(times_gps) - 1, video_frames).astype(int)

        fig = plt.figure(dpi=100, figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        def update(k):
            ax.cla()
            idx = frame_idxs[k]
            globe_plot(
                [r1[:idx + 1], r2[:idx + 1]],
                t=[times_gps[:idx + 1], times_gps[:idx + 1]],
                title=f"Globe Plot Demo: t + {(times_gps[idx] - times_gps[0]) / 3600.0:.1f} hr",
                c="black",
                labels=["Orbit 1", "Orbit 2"],
                orbit_colors=["cyan", "magenta"],
                globe_time=Time(times_gps[idx], format="gps"),
            )
            fig.axes[:] = [ax]
            return ax,

        plt.close(fig)

        fig = plt.figure(dpi=100, figsize=(8, 8))
        ani_ax = fig.add_subplot(111, projection="3d")

        def update2(k):
            ani_ax.cla()
            idx = frame_idxs[k]
            globe_plot(
                [r1[:idx + 1], r2[:idx + 1]],
                t=[times_gps[:idx + 1], times_gps[:idx + 1]],
                title=f"Globe Plot Demo: t + {(times_gps[idx] - times_gps[0]) / 3600.0:.1f} hr",
                c="black",
                labels=["Orbit 1", "Orbit 2"],
                orbit_colors=["cyan", "magenta"],
                globe_time=Time(times_gps[idx], format="gps"),
            )
            # keep only latest axes visible
            for extra_ax in fig.axes[:-1]:
                extra_ax.remove()
            return ani_ax,

        ani = FuncAnimation(fig, update2, frames=len(frame_idxs), interval=1000 / fps, blit=False)
        writer = FFMpegWriter(fps=fps, bitrate=4000)
        ani.save(str(out_mp4), writer=writer)
        plt.close(fig)
        print("Saved:", out_mp4)
        outputs["video"] = str(out_mp4)

    return outputs


if __name__ == "__main__":
    main(make_figures=True, make_video=True, fast=False)
