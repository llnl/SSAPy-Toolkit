from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from ssapy_toolkit.plots.write_gifs import write_gif
from ssapy_toolkit.plots.write_videos import write_video


def _make_frame(path: Path, size=(80, 60), text="0") -> Path:
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), text, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def test_write_gif_accepts_glob_and_normalizes_frame_sizes(tmp_path):
    frames = tmp_path / "frames"
    _make_frame(frames / "frame_10.png", size=(80, 60), text="10")
    _make_frame(frames / "frame_2.png", size=(60, 80), text="2")
    _make_frame(frames / "frame_1.png", size=(80, 60), text="1")
    output = tmp_path / "out" / "animation.gif"

    write_gif(
        str(output),
        str(frames / "frame_*.png"),
        fps=5,
        target_size=(80, 60),
        warn_on_ambiguous=False,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    with imageio.get_reader(output) as reader:
        assert reader.get_length() == 3


def test_write_video_accepts_explicit_frames_and_resizes(tmp_path):
    frames = [
        _make_frame(tmp_path / "frames" / "frame_0.png", size=(80, 60), text="0"),
        _make_frame(tmp_path / "frames" / "frame_1.png", size=(60, 80), text="1"),
        _make_frame(tmp_path / "frames" / "frame_2.png", size=(80, 60), text="2"),
    ]
    output = tmp_path / "out" / "animation.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    write_video(str(output), [str(path) for path in frames], fps=5, target_size=(80, 60))

    assert output.exists()
    assert output.stat().st_size > 0
