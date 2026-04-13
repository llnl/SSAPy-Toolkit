#!/usr/bin/env python3
# Purpose: generate a dense set of rotated unit-vector plots and stitch them
# into a GIF. Frames are saved under figpath("tests/rotate_vector_frames/")
# and the GIF is saved under figpath("tests/"), using only ssapy_toolkit:
#   - ut.figpath(...) to resolve paths
#   - ut.save_plot(...) to save each frame
#   - ut.write_gif(...) to stitch the GIF
#
# Requirements honored:
# - Use numpy (no math, no typing).
# - Do not manually create directories; rely on ssapy_toolkit.

import numpy as np
from IPython.display import clear_output
import ssapy_toolkit as ut

# -------------------------- configuration --------------------------
v_unit = np.array([1, 0, 0])   # base unit vector to rotate (x-axis by default)
theta_step = 10                # polar step (deg): 0..180
phi_step   = 10                # azimuth step (deg): 0..360
fps        = 20                # gif frame rate

# Frame count for 0..180 and 0..360 inclusive with given steps
num_theta = int(180 / theta_step) + 1
num_phi   = int(360 / phi_step) + 1
total     = num_theta * num_phi

# ----------------------------- generate frames -----------------------------
i = 0
for theta in range(0, 181, theta_step):
    for phi in range(0, 361, phi_step):
        clear_output(wait=True)

        frame_path = ut.figpath(f"tests/rotate_vector_frames/{i:06d}")
        ut.rotate_vector(v_unit, theta, phi, save_path=frame_path)

        i += 1
        print(f"Rendered {i}/{total} frames")

# ----------------------------- write the GIF -----------------------------
# Build the exact list of frame paths we just saved, in numeric order.
frames = [ut.figpath(f"tests/rotate_vector_frames/{j:06d}.jpg") for j in range(total)]

# Place the GIF itself under figpath('tests/'). Convert the default .jpg to .gif.
gif_base_jpg_like = ut.figpath(f"tests/rotate_vectors_{v_unit[0]:.0f}_{v_unit[1]:.0f}_{v_unit[2]:.0f}")
if gif_base_jpg_like.lower().endswith(".jpg"):
    gif_path = gif_base_jpg_like[:-4] + ".gif"
else:
    # If your figpath doesn't add ".jpg", still ensure we end with ".gif"
    gif_path = gif_base_jpg_like + ".gif"

ut.write_gif(gif_name=gif_path, frames=frames, fps=fps)

print("\nGIF written to:", gif_path)
print("First frame:", frames[0])
print("Last frame:", frames[-1])
print("Total frames:", len(frames))
