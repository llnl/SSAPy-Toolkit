# generate_sphere_of_vectors.py
# Purpose: generate N 3D vectors that point in uniformly random directions on the
#          sphere and all have the same magnitude. Returns a NumPy array (N, 3).
#
# Module use:
#   from yeager_utils.compute import generate_sphere_vectors
#   V = generate_sphere_vectors(1000, 13.0)  # example call
#
# Script use (hard-coded example at bottom):
#   python3 -m yeager_utils.compute.generate_sphere_of_vectors
#
# Notes:
#   - Uses NumPy only (no math, no typing).
#   - Sampling method: draw standard normals in R^3 and normalize each row.

import numpy as np


def generate_sphere_vectors(n, magnitude, seed=None):
    """
    Generate N three-dimensional vectors with identical magnitude that are
    uniformly distributed in direction over the unit sphere.

    Parameters
    ----------
    n : int
        Number of vectors to generate. If n <= 0, returns an empty (0, 3) array.
    magnitude : float
        Desired magnitude for every vector.
    seed : int or None
        Optional seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Array of shape (n, 3). Each row has norm approximately equal to `magnitude`.
    """
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3), dtype=float)

    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, 3))
    norms = np.linalg.norm(V, axis=1, keepdims=True)

    # Guard (vanishingly unlikely, but avoid division by zero):
    norms = np.where(norms == 0.0, 1.0, norms)

    U = V / norms
    return U * float(magnitude)


def _summarize(V, label="vectors"):
    norms = np.linalg.norm(V, axis=1)
    print(f"{label}: shape={V.shape}")
    if V.size == 0:
        return
    print(f"norm range: [{norms.min():.6f}, {norms.max():.6f}]  mean={norms.mean():.6f}")
    m = min(5, V.shape[0])
    if m > 0:
        print("first few rows:")
        for i in range(m):
            v = V[i]
            print(f"  {i:03d}: [{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}]  |v|={np.linalg.norm(v):.6f}")


if __name__ == "__main__":
    # ----------------------------
    # Hard-coded example parameters
    # ----------------------------
    N = 1000         # number of vectors to generate
    MAGNITUDE = 13.0 # target magnitude for each vector
    SEED = 42        # set to None for non-deterministic sampling
    SAVE_PATH = ""   # set to e.g. "vectors.npy" to save the array

    V = generate_sphere_vectors(N, MAGNITUDE, seed=SEED)
    _summarize(V, label=f"vectors (mag={MAGNITUDE})")

    if SAVE_PATH:
        np.save(SAVE_PATH, V)
        print(f"Saved to: {SAVE_PATH}")
