#!/usr/bin/env python3
"""
Demo for write_gif() using yeager_utils.figpath for all file outputs.

Requirements:
  - Pillow, imageio, numpy
  - yeager_utils providing:
        figpath(name: str) -> str
        write_gif(...)
Run:
  python demo_write_gif.py
"""

from pathlib import Path
import math
import random
import sys
import shutil

from PIL import Image, ImageDraw, ImageFont
from yeager_utils import figpath, write_gif


def _measure_text(draw, text, font=None):
    """
    Robust text measurement:
      - Pillow 10+: use textbbox
      - Older Pillow: fall back to textsize
    Returns (width, height).
    """
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    except Exception:
        return draw.textsize(text, font=font)


def make_frame(size, frame_idx, total_frames, bg=(255, 255, 255, 255)):
    """
    Create a simple RGBA frame with a moving stripe, circles, and a label.
    size is a tuple (w, h).
    """
    w, h = size
    img = Image.new("RGBA", (w, h), bg)
    dr = ImageDraw.Draw(img)

    # Moving stripe
    t = frame_idx / max(1, total_frames - 1)
    x0 = int((0.1 + 0.8 * t) * w)
    dr.rectangle([x0 - max(2, w // 20), 0, x0 + max(2, w // 20), h], fill=(0, 0, 0, 40))

    # A few circles with deterministic randomness
    rng = random.Random(frame_idx)
    for _ in range(6):
        r = rng.randint(max(2, h // 12), max(3, h // 5))
        cx = rng.randint(r, max(r + 1, w - r))
        cy = rng.randint(r, max(r + 1, h - r))
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 0, 0, 90), width=max(1, r // 6))

    # Label
    label = "Frame %02d" % frame_idx
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    tw, th = _measure_text(dr, label, font)
    dr.rectangle([5, 5, 5 + tw + 6, 5 + th + 6], fill=(255, 255, 255, 200))
    dr.text((8, 8), label, fill=(0, 0, 0, 255), font=font)

    return img


def create_varying_frames(n, base_size=(320, 240), subdir="demo_frames_a"):
    """
    Build frames whose sizes vary to exercise uniform_size behavior.
    Filenames are unpadded (frame_0.png ...) to exercise natural sorting.
    Returns list of string paths.
    """
    paths = []
    bw, bh = base_size
    for i in range(n):
        s = 0.25 * (1.0 + math.sin(2.0 * math.pi * i / max(1, n)))
        w = int(bw * (0.75 + s))
        h = int(bh * (0.75 + 0.5 * s))
        img = make_frame((max(40, w), max(40, h)), i, n)

        p = Path(figpath("tests/%s/frame_%d.png" % (subdir, i)))
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        paths.append(str(p))
    return paths


def create_constant_frames(n, size=(320, 240), subdir="demo_frames_b"):
    """
    Build frames that all share the same size.
    Filenames are zero-padded (frame_00.png ...), so lexicographic equals natural.
    Returns list of string paths.
    """
    paths = []
    for i in range(n):
        img = make_frame(size, i, n)
        p = Path(figpath("tests/%s/frame_%02d.png" % (subdir, i)))
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        paths.append(str(p))
    return paths


def cleanup_frames(paths_a, paths_b):
    """
    Delete exactly the PNGs we created (regardless of where figpath put them),
    then remove their parent folders if they become empty. Leaves GIFs intact.
    """
    all_paths = [Path(p) for p in (paths_a + paths_b)]
    # Delete files
    deleted = 0
    for p in all_paths:
        try:
            p.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
        except IsADirectoryError:
            # Shouldn't happen, but ignore just in case
            pass
        except Exception as e:
            print("Warning: could not delete %s: %s" % (p, e))

    # Try to remove now-empty parent dirs for the frame sets
    parent_dirs = sorted({Path(p).parent for p in all_paths})
    for d in parent_dirs:
        try:
            # Only remove if empty
            next(d.rglob("*")).__class__  # will raise StopIteration if empty
            # if not empty, skip
        except StopIteration:
            try:
                d.rmdir()
            except Exception:
                # maybe race or perms; ignore
                pass

    print("\nFrame cleanup complete. Deleted %d PNG(s)." % deleted)


def main():
    print("Generating demo frames...")
    paths_a = create_varying_frames(n=16, base_size=(320, 240), subdir="demo_frames_a")
    paths_b = create_constant_frames(n=16, size=(320, 240), subdir="demo_frames_b")

    print("Writing GIFs...")

    # 1) Basic usage: natural sort on unpadded names; 10 fps; loop forever
    write_gif(
        gif_name=str(figpath("tests/gif_basic.gif")),
        frames=paths_a,
        fps=10,
        sort_frames=True,
        warn_on_ambiguous=True,
        uniform_size=False,
    )

    # 2) Uniform canvas: normalize all frames to a fixed size with padding
    write_gif(
        gif_name=str(figpath("tests/gif_uniform_400x300.gif")),
        frames=paths_a,
        fps=12,
        sort_frames=True,
        warn_on_ambiguous=True,
        uniform_size=True,
        target_size=(400, 300),
        bg_color=(30, 30, 30, 255),
    )

    # 3) Duration override: fixed 0.20 s per frame, loop once, use constant-size set
    write_gif(
        gif_name=str(figpath("tests/gif_duration_loop1.gif")),
        frames=paths_b,
        duration=0.20,  # overrides fps
        loop=1,         # play twice total: initial + 1 repeat
        sort_frames=True,
        warn_on_ambiguous=False,
        uniform_size=True,
        target_size=None,
        bg_color=(255, 255, 255, 0),
    )

    # 4) Using a generator with Path.glob; derive the directory from saved frames
    frames_b_dir = Path(paths_b[0]).parent
    frame_matches = sorted(frames_b_dir.glob("frame_*.png"))
    if not frame_matches:
        print("Warning: no frames found for gif_glob.gif under:", frames_b_dir)
    else:
        write_gif(
            gif_name=str(figpath("tests/out/gif_glob.gif")),
            frames=(str(p) for p in frame_matches),
            fps=15,
            sort_frames=True,
            uniform_size=True,
        )

    # Clean up the temporary frames now that GIFs are written
    cleanup_frames(paths_a, paths_b)

    print("\nDone.")
    print("Check the output GIFs under:", figpath("tests/"))


if __name__ == "__main__":
    main()
