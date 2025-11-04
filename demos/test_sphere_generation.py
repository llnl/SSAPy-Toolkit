import numpy as np
import matplotlib.pyplot as plt

from yeager_utils import generate_sphere_vectors, yufig

def main():
    # Match your earlier example
    n = 10_000
    mag = 3.5
    seed = 0

    A_uniform = generate_sphere_vectors(n, mag, seed=seed, distribution="uniform")
    A_random  = generate_sphere_vectors(n, mag, seed=seed, distribution="random")

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(A_uniform[:, 0], A_uniform[:, 1], A_uniform[:, 2], s=2, alpha=0.8)
    ax1.set_title("Uniform on S^2 (Gaussian-normalized)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_xlim(-mag, mag)
    ax1.set_ylim(-mag, mag)
    ax1.set_zlim(-mag, mag)
    try:
        ax1.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(A_random[:, 0], A_random[:, 1], A_random[:, 2], s=2, alpha=0.8)
    ax2.set_title("Area-uniform via z~U[-1,1], phi~U[0,2*pi)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_xlim(-mag, mag)
    ax2.set_ylim(-mag, mag)
    ax2.set_zlim(-mag, mag)
    try:
        ax2.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass

    fig.tight_layout()
    yufig(fig, "tests/spheres_subplots.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
