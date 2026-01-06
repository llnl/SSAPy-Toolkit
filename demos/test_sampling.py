"""
Demo for perturb_state_3d.

This script:
  * Defines a nominal 3D position + velocity
  * Applies perturb_state_3d many times with different distributions:
        - uniform   (in a solid ball)
        - normal    (component-wise Gaussian)
        - shell     (fixed-radius sphere surface)
        - laplace   (component-wise Laplace)
  * Prints basic stats of the perturbations
  * Plots a 2x2 figure per case:
        [0,0] position offsets (dx, dy)
        [0,1] velocity offsets (dvx, dvy)
        [1,0] position samples  (x,  y)  centered around r_nom
        [1,1] velocity samples  (vx, vy) centered around v_nom

Run:
    python demo_perturb_state_3d.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from yeager_utils import perturb_state_3d, yufig  # adjust import path if needed


def demo_single_distribution(
    r_nom,
    v_nom,
    pos_scale,
    vel_scale,
    pos_distribution,
    vel_distribution,
    n_samples=1000,
    rng_seed=42,
):
    """
    Draw many perturbed states and return offsets for analysis / plotting.
    """
    rng = np.random.default_rng(rng_seed)

    r_offsets = np.zeros((n_samples, 3))
    v_offsets = np.zeros((n_samples, 3))

    for i in range(n_samples):
        r_p, v_p = perturb_state_3d(
            r_nom,
            v_nom,
            pos_scale=pos_scale,
            vel_scale=vel_scale,
            pos_distribution=pos_distribution,
            vel_distribution=vel_distribution,
            rng=rng,
        )
        r_offsets[i] = r_p - r_nom
        v_offsets[i] = v_p - v_nom

    return r_offsets, v_offsets


def summarize_offsets(name, r_offsets, v_offsets):
    """
    Print basic statistics for position & velocity perturbations.
    """
    # radial norms
    r_norm = np.linalg.norm(r_offsets, axis=1)
    v_norm = np.linalg.norm(v_offsets, axis=1)

    print(f"\n=== {name} ===")
    print(
        "Position offsets | "
        f"mean radius = {np.mean(r_norm):.3f}, "
        f"std radius  = {np.std(r_norm):.3f}, "
        f"max radius  = {np.max(r_norm):.3f}"
    )

    print(
        "Velocity offsets | "
        f"mean radius = {np.mean(v_norm):.3f}, "
        f"std radius  = {np.std(v_norm):.3f}, "
        f"max radius  = {np.max(v_norm):.3f}"
    )


def plot_offsets_2d(name, r_offsets, v_offsets, r_nom, v_nom):
    """
    2x2 plots:
      [0,0] position offsets (dx, dy)
      [0,1] velocity offsets (dvx, dvy)
      [1,0] position samples  (x,  y)  centered around r_nom
      [1,1] velocity samples  (vx, vy) centered around v_nom
    """
    # Actual samples (nominal + offsets)
    r_samples = r_nom[np.newaxis, :] + r_offsets
    v_samples = v_nom[np.newaxis, :] + v_offsets

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(name)

    # --- Top-left: position offsets ---
    ax = axes[0, 0]
    ax.scatter(r_offsets[:, 0], r_offsets[:, 1], s=5, alpha=0.5)
    ax.set_title("Position offsets (dx, dy)")
    ax.set_xlabel("dx")
    ax.set_ylabel("dy")
    ax.axis("equal")
    ax.grid(True)

    # --- Top-right: velocity offsets ---
    ax = axes[0, 1]
    ax.scatter(v_offsets[:, 0], v_offsets[:, 1], s=5, alpha=0.5)
    ax.set_title("Velocity offsets (dvx, dvy)")
    ax.set_xlabel("dvx")
    ax.set_ylabel("dvy")
    ax.axis("equal")
    ax.grid(True)

    # --- Bottom-left: absolute positions (blob around r_nom) ---
    ax = axes[1, 0]
    ax.scatter(r_samples[:, 0], r_samples[:, 1], s=5, alpha=0.5)
    ax.scatter([r_nom[0]], [r_nom[1]], marker="+", s=80)  # mark nominal
    ax.set_title("Position samples (x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)

    # --- Bottom-right: absolute velocities (blob around v_nom) ---
    ax = axes[1, 1]
    ax.scatter(v_samples[:, 0], v_samples[:, 1], s=5, alpha=0.5)
    ax.scatter([v_nom[0]], [v_nom[1]], marker="+", s=80)  # mark nominal
    ax.set_title("Velocity samples (vx, vy)")
    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.axis("equal")
    ax.grid(True)

    plt.tight_layout()
    return fig  # let caller decide whether to show/save


def main():
    # Nominal state: put it somewhere obvious in 3D
    r_nom = np.array([100.0, 100.0, 100.0])   # meters
    v_nom = np.array([0.0, 1.0, 0.0])         # m/s

    # Scales for perturbations
    pos_scale = 1.0    # ~ 1 m positional "radius"/sigma depending on distribution
    vel_scale = 0.1    # ~ 0.1 m/s velocity spread

    distributions = [
        ("Uniform ball", "uniform", "uniform"),
        ("Gaussian", "normal", "normal"),
        ("Shell (fixed radius)", "shell", "shell"),
        ("Laplace", "laplace", "laplace"),
    ]

    os.makedirs("tests", exist_ok=True)

    for label, pos_dist, vel_dist in distributions:
        r_offsets, v_offsets = demo_single_distribution(
            r_nom,
            v_nom,
            pos_scale=pos_scale,
            vel_scale=vel_scale,
            pos_distribution=pos_dist,
            vel_distribution=vel_dist,
            n_samples=2000,
            rng_seed=123,
        )
        summarize_offsets(label, r_offsets, v_offsets)
        fig = plot_offsets_2d(label, r_offsets, v_offsets, r_nom, v_nom)

        # sanitize label for filename and save figure
        safe_label = label.replace(" ", "_")
        yufig(fig, f"tests/sampling_tests_{safe_label}")  # no extension -> yufig adds .jpg

        # If you still want to pop the figure interactively:
        plt.show()


if __name__ == "__main__":
    main()
