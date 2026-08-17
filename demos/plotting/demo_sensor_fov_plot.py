#!/usr/bin/env python3
"""
Demo: satellite field-of-view cone in a 3D Plotly scene.

The demo propagates a single SSAPy orbit, builds a nadir-pointing field-of-view
cone, and saves an interactive HTML file. The same structure can be copied and
edited for custom sensor geometry.
"""

import os
import sys
from pathlib import Path

from ssapy_toolkit.plots.figpath import figpath
from ssapy_toolkit.plots.sensor_fov_plot import DEFAULT_CFG, build_figure, propagate_orbit

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    cfg = DEFAULT_CFG.copy()
    cfg["a_km"] = 7078.0
    cfg["e"] = 0.001
    cfg["inc_deg"] = 98.2
    cfg["raan_deg"] = 35.0
    cfg["nu_deg"] = 35.0
    cfg["n_orbits"] = 0.45 if fast else 1.0
    cfg["dt_s"] = 90.0 if fast else 30.0
    cfg["fov_half_angle_deg"] = 28.0
    cfg["fov_cone_length_km"] = 1800.0
    cfg["fov_pointing_mode"] = "nadir"
    cfg["fov_time_index"] = 12
    cfg["fov_animate"] = True
    cfg["fov_anim_step"] = 3 if fast else 8
    cfg["axis_range_km"] = 9000.0

    r_km, v_kms = propagate_orbit(cfg)
    fig = build_figure(cfg, r_km, v_kms)

    html_path = None
    if make_figures:
        html_path = Path(figpath("demo_gallery/figures/demo_sensor_fov_plot.html"))
        fig.write_html(str(html_path))
        print(f"Saved: {html_path}")

    return {
        "figure": fig,
        "r_km": r_km,
        "v_kms": v_kms,
        "n_steps": len(r_km),
        "html": str(html_path) if html_path else None,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
