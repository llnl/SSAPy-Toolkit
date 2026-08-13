from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image, ImageDraw

from ssapy_toolkit.plots import write_gifs, write_videos
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


def test_gif_sorting_resolution_and_validation_branches(tmp_path, monkeypatch):
    assert write_gifs._sort_frames([]) == []
    frame = _make_frame(tmp_path / "only.png")
    with pytest.raises(ValueError, match="At least 2"):
        write_gifs._resolve_frames([frame])
    with pytest.raises(ValueError, match="No frames"):
        write_gifs._resolve_frames(str(tmp_path / "missing_*.png"))
    with pytest.raises(ValueError, match="target size"):
        write_gifs._fit_to_canvas(Image.new("RGB", (1, 1)), (0, 10))
    with pytest.warns(UserWarning, match="Multiple files"):
        assert len(write_gifs._sort_frames([str(frame), str(frame)])) == 2

    monkeypatch.setattr(write_gifs, "natural_key", lambda path: (_ for _ in ()).throw(RuntimeError("bad key")))
    with pytest.warns(UserWarning, match="Could not build"):
        assert write_gifs._sort_frames(["b.png", "a.png"]) == ["a.png", "b.png"]

    monkeypatch.setattr(write_gifs, "natural_key", lambda path: [object()])
    with pytest.warns(UserWarning, match="Natural sort failed"):
        assert write_gifs._sort_frames(["b.png", "a.png"]) == ["a.png", "b.png"]


def test_write_gif_validation_and_nonuniform_writer(monkeypatch, tmp_path, capsys):
    frames = [tmp_path / "frame_1.png", tmp_path / "frame_2.png"]
    for path in frames:
        _make_frame(path)

    with pytest.raises(ValueError, match="must end"):
        write_gif(str(tmp_path / "bad.mp4"), frames)
    with pytest.raises(ValueError, match="duration"):
        write_gif(str(tmp_path / "bad.gif"), frames, duration=0)
    with pytest.raises(ValueError, match="fps"):
        write_gif(str(tmp_path / "bad.gif"), frames, fps=0)

    appended = []

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def append_data(self, arr):
            appended.append(np.asarray(arr).shape)

    monkeypatch.setattr(write_gifs.imageio, "get_writer", lambda *args, **kwargs: FakeWriter())
    monkeypatch.setattr(write_gifs.imageio, "imread", lambda path: np.zeros((4, 5, 3), dtype=np.uint8))
    write_gif(str(tmp_path / "ok.gif"), frames, duration=0.1, sort_frames=False, uniform_size=False, verbose=True)
    assert appended == [(4, 5, 3), (4, 5, 3)]
    assert "Writing gif" in capsys.readouterr().out


def test_write_video_validation_and_fake_cv2_branches(monkeypatch, tmp_path, capsys):
    folder = tmp_path / "frames"
    folder.mkdir()
    for name in ["frame_2.png", "frame_1.png", "bad.png"]:
        (folder / name).write_text("x", encoding="utf-8")
    clean_folder = tmp_path / "clean_frames"
    clean_folder.mkdir()
    for name in ["frame_1.png", "frame_2.png"]:
        (clean_folder / name).write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        write_video(str(tmp_path / "empty.mp4"), [])

    monkeypatch.setattr(write_videos.cv2, "imread", lambda path: None)
    with pytest.raises(ValueError, match="first frame"):
        write_video(str(tmp_path / "unreadable.mp4"), [str(folder / "frame_1.png")])

    arrays = {
        str(folder / "frame_1.png"): np.zeros((3, 4), dtype=np.uint8),
        str(folder / "frame_2.png"): np.zeros((5, 6, 3), dtype=np.uint8),
        str(folder / "bad.png"): None,
        str(clean_folder / "frame_1.png"): np.zeros((3, 4, 3), dtype=np.uint8),
        str(clean_folder / "frame_2.png"): np.zeros((3, 4, 3), dtype=np.uint8),
    }
    writes = []

    class FakeWriter:
        def __init__(self, opened=True):
            self.opened = opened

        def isOpened(self):
            return self.opened

        def write(self, frame):
            writes.append(np.asarray(frame).shape)

        def release(self):
            writes.append("released")

    monkeypatch.setattr(write_videos.cv2, "imread", lambda path: arrays[str(path)])
    monkeypatch.setattr(write_videos.cv2, "resize", lambda img, size: np.zeros((size[1], size[0]) + (() if img.ndim == 2 else (img.shape[2],)), dtype=img.dtype))
    monkeypatch.setattr(write_videos.cv2, "cvtColor", lambda img, code: np.repeat(img[:, :, None], 3, axis=2))
    monkeypatch.setattr(write_videos.cv2, "COLOR_GRAY2BGR", 1, raising=False)
    monkeypatch.setattr(write_videos.cv2, "VideoWriter_fourcc", lambda *args: 123)
    monkeypatch.setattr(write_videos.cv2, "VideoWriter", lambda *args: FakeWriter(opened=True))

    with pytest.warns(UserWarning, match="skipping"):
        write_video(
            str(tmp_path / "ok.mp4"),
            [str(folder / "frame_1.png"), str(folder / "bad.png"), str(folder / "frame_2.png")],
            fps=2,
            sort_frames=False,
            target_size=(4, 3),
            freeze_last_seconds=1.0,
        )
    assert writes.count((3, 4, 3)) >= 3
    assert writes[-1] == "released"

    writes.clear()
    write_video(str(tmp_path / "folder.mp4"), clean_folder, fps=2, verbose=True)
    assert writes[-1] == "released"
    assert "Writing video" in capsys.readouterr().out

    monkeypatch.setattr(write_videos.cv2, "VideoWriter", lambda *args: FakeWriter(opened=False))
    with pytest.raises(RuntimeError, match="Failed to open"):
        write_video(str(tmp_path / "bad_writer.mp4"), [str(folder / "frame_1.png")], sort_frames=False)
