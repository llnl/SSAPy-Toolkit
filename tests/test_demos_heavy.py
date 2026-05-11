import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demos.demo_fancy_video import orbit_moon_video_demo
from demos.demo_gifify import main as demo_gifify
from demos.demo_write_gif import main as demo_write_gif
from demos.demo_write_video import main as demo_write_video


@pytest.mark.slow
def test_demo_fancy_video_smoke():
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
    out = demo_gifify(make_artifacts=False, fast=True, verbose=False)
    assert isinstance(out, dict)


@pytest.mark.slow
def test_demo_write_gif_smoke():
    out = demo_write_gif(make_artifacts=False, fast=True)
    assert isinstance(out, dict)


@pytest.mark.slow
def test_demo_write_video_smoke():
    out = demo_write_video(make_artifacts=False, fast=True)
    assert isinstance(out, dict)