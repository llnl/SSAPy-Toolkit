import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

from ssapy_toolkit.plots.figpath import figpath
from ssapy_toolkit.plots.write_gifs import write_gif  # inferred from repo structure [41]

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def make_frame(size, i, n):
    img = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"{i}/{n}", fill=(0, 0, 0, 255))
    return img


def create_varying_frames(n, base_size=(320, 240), subdir="demo_frames_a"):
    paths = []
    bw, bh = base_size
    for i in range(n):
        phase = 2.0 * np.pi * i / max(1, n)
        s = 0.25 * (1.0 + np.sin(phase))
        w = int(bw * (0.75 + s))
        h = int(bh * (0.75 + 0.5 * s))
        img = make_frame((max(40, w), max(40, h)), i, n)
        p = Path(figpath("figures/%s/frame_%d.png" % (subdir, i)))
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        paths.append(str(p))
    return paths


def create_constant_frames(n, size=(320, 240), subdir="demo_frames_b"):
    paths = []
    for i in range(n):
        img = make_frame(size, i, n)
        p = Path(figpath("figures/%s/frame_%02d.png" % (subdir, i)))
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        paths.append(str(p))
    return paths


def cleanup_frames(paths_a, paths_b):
    all_paths = [Path(p) for p in (paths_a + paths_b)]
    deleted = 0
    for p in all_paths:
        try:
            p.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            print("Warning: could not delete %s: %s" % (p, e))

    parent_dirs = sorted({p.parent for p in all_paths})
    for d in parent_dirs:
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except Exception:
            pass

    print("\nFrame cleanup complete. Deleted %d PNG(s)." % deleted)


def main(make_artifacts=None, fast=None):
    if make_artifacts is None:
        make_artifacts = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    n = 4 if fast else 16
    paths_a = create_varying_frames(n=n, base_size=(320, 240), subdir="demo_frames_a")
    paths_b = create_constant_frames(n=n, size=(320, 240), subdir="demo_frames_b")

    outputs = {}
    if make_artifacts:
        frames_a_pattern = str(Path(paths_a[0]).parent / "frame_*.png")
        outputs["gif_glob"] = write_gif(
            gif_name=str(figpath("figures/out/gif_glob_unpadded.gif")),
            frames=frames_a_pattern,
            fps=10,
            sort_frames=True,
            warn_on_ambiguous=True,
            uniform_size=True,
            target_size=(320, 240),
            bg_color=(240, 240, 240, 255),
        )

    cleanup_frames(paths_a, paths_b)
    return outputs


if __name__ == "__main__":
    main(make_artifacts=True, fast=False)