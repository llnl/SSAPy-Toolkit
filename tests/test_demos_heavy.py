from importlib import import_module
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def demo_attr(module_name, attr_name="main"):
    module_paths = {
        "demo_fancy_video": "demos.video_animation.demo_fancy_video",
        "demo_gifify": "demos.video_animation.demo_gifify",
    }
    return getattr(import_module(module_paths[module_name]), attr_name)


@pytest.mark.slow
def test_demo_fancy_video_smoke():
    orbit_moon_video_demo = demo_attr("demo_fancy_video", "orbit_moon_video_demo")
    out = orbit_moon_video_demo(
        duration_days=1.0,
        fps=4,
        seconds_per_frame=12 * 3600,
        make_figures=False,
        fast=True,
        save_gif=False,
    )
    assert "r_sc_f" in out


@pytest.mark.slow
def test_demo_gifify_smoke():
    demo_gifify = demo_attr("demo_gifify")
    out = demo_gifify(make_artifacts=False, fast=True, verbose=False)
    assert isinstance(out, dict)
