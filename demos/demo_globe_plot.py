#!/usr/bin/env python3
import os
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from astropy.time import Time
from ssapy import Orbit, rv

from ssapy_toolkit.constants import EARTH_RADIUS
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

    # ISS-like LEO orbit
    a_iss = EARTH_RADIUS + 420e3
    orbit1 = Orbit.fromKeplerianElements(
        a=a_iss,
        e=0.001,
        i=np.radians(51.6),
        pa=np.radians(40.0),
        raan=np.radians(20.0),
        trueAnomaly=0.0,
        t=t0,
    )

    # GPS-like MEO orbit
    a_gps = EARTH_RADIUS + 20200e3
    orbit2 = Orbit.fromKeplerianElements(
        a=a_gps,
        e=0.01,
        i=np.radians(55.0),
        pa=np.radians(120.0),
        raan=np.radians(240.0),
        trueAnomaly=np.radians(45.0),
        t=t0,
    )

    if fast:
        duration_hr = 4.0
        dt_s = 300.0
        video_frames = 24
        fps = 8
    else:
        duration_hr = 12.0
        dt_s = 180.0
        video_frames = 60
        fps = 16

    times_gps = t0.gps + np.arange(0.0, duration_hr * 3600.0 + dt_s, dt_s)

    r1, _ = rv(orbit1, Time(times_gps, format="gps"))
    r2, _ = rv(orbit2, Time(times_gps, format="gps"))
    r1 = np.asarray(r1, dtype=float).reshape((-1, 3))
    r2 = np.asarray(r2, dtype=float).reshape((-1, 3))

    outputs = {
        "r1": r1,
        "r2": r2,
        "times_gps": times_gps,
    }

    if make_figures:
        out_static = Path(figpath("demo_gallery/figures/demo_globe_plot_two_orbits"))
        if out_static.suffix == "":
            out_static = out_static.with_suffix(".png")
        out_static.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = globe_plot(
            [r1, r2],
            t=[times_gps, times_gps],
            title="Globe Plot Demo: ISS-like LEO and GPS-like MEO",
            c="black",
            labels=["ISS-like LEO", "GPS-like MEO"],
            orbit_colors=["cyan", "magenta"],
            globe_time=Time(times_gps[0], format="gps"),
            save_path=str(out_static),
        )
        plt.close(fig)
        print("Saved:", out_static)
        outputs["static_plot"] = str(out_static)

    if make_video:
        out_mp4 = Path(figpath("demo_gallery/figures/demo_globe_plot_animation"))
        if out_mp4.suffix == "":
            out_mp4 = out_mp4.with_suffix(".mp4")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        frame_idxs = np.linspace(0, len(times_gps) - 1, video_frames).astype(int)

        temp_dir = Path(tempfile.mkdtemp(prefix="demo_globe_plot_frames_"))
        frame_files = []

        try:
            for k, idx in enumerate(frame_idxs):
                fig, ax = globe_plot(
                    [r1[:idx + 1], r2[:idx + 1]],
                    t=[times_gps[:idx + 1], times_gps[:idx + 1]],
                    title=f"Globe Plot Demo: t + {(times_gps[idx] - times_gps[0]) / 3600.0:.1f} hr",
                    c="black",
                    labels=["ISS-like LEO", "GPS-like MEO"],
                    orbit_colors=["cyan", "magenta"],
                    globe_time=Time(times_gps[idx], format="gps"),
                )

                frame_path = temp_dir / f"frame_{k:04d}.png"
                fig.savefig(frame_path, dpi=100, bbox_inches="tight")
                plt.close(fig)

                frame_files.append(frame_path)
                print(f"Rendered frame {k+1}/{len(frame_idxs)}")

            with imageio.get_writer(out_mp4, fps=fps) as writer:
                for frame_path in frame_files:
                    writer.append_data(imageio.imread(frame_path))

            print("Saved:", out_mp4)
            outputs["video"] = str(out_mp4)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return outputs


if __name__ == "__main__":
    main(make_figures=True, make_video=True, fast=False)
