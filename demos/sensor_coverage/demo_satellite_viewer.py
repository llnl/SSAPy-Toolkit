#!/usr/bin/env python3
"""
Demo: self-contained Three.js satellite viewer export.

This builds the browser-based satellite viewer HTML artifact from the packaged
JavaScript sources and texture helper. No source data files are written into the
repository; the output goes under the standard SSATK figure directory.
"""

GALLERY_CATEGORY = "sensor_coverage"

import os
import sys
from pathlib import Path

from ssapy_toolkit.plots.build_satellite_viewer import build
from ssapy_toolkit.plots.figpath import figpath

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST

    if not make_figures:
        return {"html": None, "skipped": True, "reason": "figures_disabled"}

    html_path = Path(figpath("demo_gallery/figures/demo_satellite_viewer.html"))
    written = build(out_path=str(html_path), verbose=True)
    return {"html": str(written), "skipped": False}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
