#!/usr/bin/env python3
"""
Demo: eclipse search and space-view plotting.

The Matplotlib strip is a quick visual check of eclipse appearance. The Plotly
space view uses the same search/refinement code as the eclipse plot module and
saves an interactive HTML scene.
"""

import os
import sys
from pathlib import Path

from ssapy_toolkit.plots.eclipse_appearance_strip import make_strip
from ssapy_toolkit.plots.eclipse_space_view_plotly import find_and_plot_eclipse, plot_space_view_plotly
from ssapy_toolkit.plots.figpath import figpath

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    outputs = {
        "strip_fig": None,
        "search_fig": None,
        "space_fig": None,
        "stats": None,
        "files": [],
    }
    search_days = 730.0 if make_figures else 120.0

    strip_path = figpath("demo_gallery/figures/demo_eclipse_lunar_strip.png") if make_figures else None
    strip_fig = make_strip(kind="lunar", n_panels=5 if fast else 9, save_path=strip_path)
    outputs["strip_fig"] = strip_fig
    if strip_path:
        outputs["files"].append(strip_path)
        print(f"Saved: {strip_path}")

    search_path = figpath("demo_gallery/figures/demo_eclipse_lunar_search.png") if make_figures else None
    search_fig, stats = find_and_plot_eclipse(
        mode="lunar",
        search_days=search_days,
        save_path=search_path,
        verbose=not fast,
    )
    outputs["search_fig"] = search_fig
    outputs["stats"] = stats
    if search_path:
        outputs["files"].append(search_path)

    if make_figures:
        html_path = Path(figpath("demo_gallery/figures/demo_eclipse_space_lunar.html"))
        space_fig = plot_space_view_plotly(
            mode="lunar",
            search_days=search_days,
            save_path=str(html_path),
            verbose=not fast,
        )
        outputs["space_fig"] = space_fig
        outputs["files"].append(str(html_path))
    else:
        outputs["space_fig"] = None

    return outputs


if __name__ == "__main__":
    main(make_figures=True, fast=False)
