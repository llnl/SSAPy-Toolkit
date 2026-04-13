#!/usr/bin/env python3
"""
Demo for write_video() using ssapy_toolkit.figpath for all file outputs.

Requirements:
  - Pillow, numpy, OpenCV (cv2)
  - ssapy_toolkit providing:
        figpath(name: str) -> str
        write_video(...)
Run:
  python demo_write_video.py
"""

from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ssapy_toolkit import figpath, write_video


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
    stripe_half = max(2, w // 20)
    dr.rectangle([x0 - stripe_half, 0, x0 + stripe_half, h], fill=(0, 0, 0, 40))

    # A few circles with deterministic randomness
    rng = random.Random(frame_idx)
    for _ in range(6):
        r = rng.randint(max(2, h // 12), max(3, h // 5))
        cx = rng.randint(r, max(r + 1, w - r))
        cy = rng.randint(r, max(r + 1, h - r))
        dr.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=(0, 0, 0, 90),
            width=max(1, r // 6),
        )

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


def create_varying_frames(n, base_size=(320, 240), subdir="demo_frames_video_a"):
    """
    Build frames whose sizes vary to exercise uniform_size behavior.
    Filenames are unpadded (frame_0.png ...) to exercise natural sorting.
    Returns list of string paths.
    """
    paths = []
    bw, bh = base_size
    for i in range(n):
        # Use numpy for trig
        phase = 2.0 * np.pi * i / max(1, n)
        s = 0.25 * (1.0 + np.sin(phase))
        w = int(bw * (0.75 + s))
        h = int(bh * (0.75 + 0.5 * s))
        img = make_frame((max(40, w), max(40, h)), i, n)

        p = Path(figpath(f"tests/{subdir}/frame_{i}.png"))
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        paths.append(str(p))
    return paths


def create_constant_frames(n, size=(320, 240), subdir="demo_frames_video_b"):
    """
    Build frames that all share the same size.
    Filenames are zero-padded (frame_00.png ...), so lexicographic equals natural.
    Returns list of string paths.
    """
    paths = []
    for i in range(n):
        img = make_frame(size, i, n)
        p = Path(figpath(f"tests/{subdir}/frame_{i:02d}.png"))
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        paths.append(str(p))
    return paths


def cleanup_frames(paths_a, paths_b):
    """
    Delete exactly the PNGs we created (regardless of where figpath put them),
    then remove their parent folders if they become empty. Leaves videos intact.
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
            next(d.rglob("*")).__class__  # raises StopIteration if empty
        except StopIteration:
            try:
                d.rmdir()
            except Exception:
                # perms or races; ignore
                pass

    print("\nFrame cleanup complete. Deleted %d PNG(s)." % deleted)


def main():
    print("Generating demo frames for video...")
    paths_a = create_varying_frames(
        n=16, base_size=(320, 240), subdir="demo_frames_video_a"
    )
    paths_b = create_constant_frames(
        n=16, size=(320, 240), subdir="demo_frames_video_b"
    )

    print("Writing videos...")

    # 1) Basic usage: natural sort on unpadded names; 15 fps
    write_video(
        video_name=str(figpath("tests/video_basic.mp4")),
        frames=paths_a,
        fps=3,
        sort_frames=True,
        warn_on_ambiguous=True,
    )

    # 2) Uniform canvas: normalize all frames to a fixed size with resizing
    write_video(
        video_name=str(figpath("tests/video_uniform_640x480.mp4")),
        frames=paths_a,
        fps=2,
        sort_frames=True,
        warn_on_ambiguous=True,
        target_size=(640, 480),
        freeze_last_seconds=0.0,
    )

    # 3) Constant-size frames, sort on zero-padded filenames, freeze last frame
    write_video(
        video_name=str(figpath("tests/video_freeze_last.mp4")),
        frames=paths_b,
        fps=3,
        sort_frames=True,
        warn_on_ambiguous=False,
        target_size=None,          # use first frame's size
        freeze_last_seconds=2.0,   # hold last frame ~2 seconds
    )

    # 4) Using a generator with Path.glob; derive the directory from saved frames
    frames_b_dir = Path(paths_b[0]).parent
    frame_matches = sorted(frames_b_dir.glob("frame_*.png"))
    if not frame_matches:
        print("Warning: no frames found for video_glob.mp4 under:", frames_b_dir)
    else:
        write_video(
            video_name=str(figpath("tests/out/video_glob.mp4")),
            frames=(str(p) for p in frame_matches),
            fps=3,
            sort_frames=True,
        )

    # Clean up the temporary frames now that videos are written
    cleanup_frames(paths_a, paths_b)

    print("\nDone.")
    print("Check the output videos under:", figpath("tests/"))


if __name__ == "__main__":
    main()
