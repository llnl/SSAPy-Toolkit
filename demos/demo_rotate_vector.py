#!/usr/bin/env python3
"""
Generate a dense set of rotated unit-vector plots and optionally stitch them
into a GIF.

Frames are saved under figpath("tests/rotate_vector_frames/")
and the GIF is saved under figpath("tests/"), using only ssapy_toolkit:
  - ut.figpath(...) to resolve paths
  - ut.write_gif(...) to stitch the GIF

Pytest-safe behavior:
- does not generate all frames by default under pytest
- does not write GIF by default under pytest
- uses a much smaller grid under pytest
"""

import os
import sys
import numpy as np

try:
    from IPython.display import clear_output
except Exception:
    def clear_output(wait=True):
        return None

import ssapy_toolkit as ut

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, make_gif=None, fast=None):
    """
    Run the rotate-vector demo.

    Parameters
    ----------
    make_figures : bool or None
        If None, defaults to False under pytest and True otherwise.
    make_gif : bool or None
        If None, defaults to False under pytest and True otherwise.
    fast : bool or None
        If None, defaults to True under pytest and False otherwise.

    Returns
    -------
    dict
        Frame paths and GIF path.
    """
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if make_gif is None:
        make_gif = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    # -------------------------- configuration --------------------------
    v_unit = np.array([1, 0, 0])

    if fast:
        theta_step = 60
        phi_step = 90
        fps = 6
    else:
        theta_step = 10
        phi_step = 10
        fps = 20

    num_theta = int(180 / theta_step) + 1
    num_phi = int(360 / phi_step) + 1
    total = num_theta * num_phi

    frames = []

    # ----------------------------- generate frames -----------------------------
    i = 0
    for theta in range(0, 181, theta_step):
        for phi in range(0, 361, phi_step):
            clear_output(wait=True)

            frame_base = ut.figpath(f"tests/rotate_vector_frames/{i:06d}")

            if make_figures:
                ut.rotate_vector(v_unit, theta, phi, save_path=frame_base)
                if frame_base.lower().endswith(".jpg"):
                    frame_path = frame_base
                else:
                    frame_path = frame_base + ".jpg"
                frames.append(frame_path)
            else:
                # In pytest-safe mode, still touch the code path if possible
                try:
                    ut.rotate_vector(v_unit, theta, phi)
                except TypeError:
                    # Fallback if rotate_vector requires save_path in this install
                    pass

            i += 1
            print(f"Rendered {i}/{total} frames")

    # ----------------------------- write the GIF -----------------------------
    gif_path = None
    if make_gif and frames:
        gif_base_jpg_like = ut.figpath(
            f"tests/rotate_vectors_{v_unit[0]:.0f}_{v_unit[1]:.0f}_{v_unit[2]:.0f}"
        )
        if gif_base_jpg_like.lower().endswith(".jpg"):
            gif_path = gif_base_jpg_like[:-4] + ".gif"
        else:
            gif_path = gif_base_jpg_like + ".gif"

        ut.write_gif(gif_name=gif_path, frames=frames, fps=fps)

        print("\nGIF written to:", gif_path)
        print("First frame:", frames[0])
        print("Last frame:", frames[-1])
        print("Total frames:", len(frames))

    return {
        "frames": frames,
        "gif_path": gif_path,
        "total_frames": total,
        "theta_step": theta_step,
        "phi_step": phi_step,
    }


if __name__ == "__main__":
    main(make_figures=True, make_gif=True, fast=False)