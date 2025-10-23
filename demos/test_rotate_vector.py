# %%
import yeager_utils as ut
import numpy as np
from IPython.display import clear_output

# Example usage:
v_unit = np.array([1, 0, 0])  # Replace this with your actual unit vector

figs = []
gif_frame_path = ut.figpath('rotate_vector_frames/')
ut.mkdir(gif_frame_path)
i = 0
for theta in range(0, 181, 10):
    for phi in range(0, 361, 10):
        clear_output(wait=True)
        new_unit_vector = ut.rotate_vector(v_unit, theta, phi, plot=True, save_idx=i)
        i += 1

gif_path = ut.figpath(f'rotate_vectors_{v_unit[0]:.0f}_{v_unit[1]:.0f}_{v_unit[2]:.0f}.gif')
ut.write_gif(gif_name=gif_path, frames=ut.sortbynum(ut.listdir(gif_frame_path)), fps=20)
