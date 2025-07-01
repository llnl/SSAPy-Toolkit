import numpy as np


def ellipse_3d(p1, p2, eccentricity, inclination=0.0, num_points=200):
    """Return coordinates of a full 3‑D ellipse whose major axis joins *p1* and *p2*.

    Parameters
    ----------
    p1, p2 : array-like of shape (3,)
        End-points of the major axis.
    eccentricity : float
        Geometric eccentricity *e*, with 0 ≤ e < 1.
    inclination : float, optional
        Rotation of the ellipse's plane about its major axis (radians, default 0).
        *θ = 0* reproduces the default orientation; positive angles obey the
        right-hand rule around the *p1 → p2* vector.
    num_points : int, optional
        Number of sample points along the curve (default 200).

    Notes
    -----
    * Semi-major axis *a* = ½|p₂-p₁|.
    * Semi-minor axis *b* = *a*·√(1-e²).
    * *e = 0* → circle; *e → 1* → highly flattened ellipse.
    """
    # Validate inputs
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if not (0.0 <= eccentricity < 1.0):
        raise ValueError("eccentricity must satisfy 0 ≤ e < 1")

    # Major‑axis properties
    delta = p2 - p1
    distance = np.linalg.norm(delta)
    if distance == 0:
        raise ValueError("p1 and p2 must be distinct points")
    a = distance / 2.0                     # semi‑major length
    midpoint = (p1 + p2) / 2.0
    u = delta / distance                  # unit vector along major axis

    # Semi‑minor length from eccentricity
    b = a * np.sqrt(1.0 - eccentricity ** 2)

    # Build an orthonormal basis {u, v0, w0}
    helper = np.array([0, 0, 1]) if not np.allclose(u, [0, 0, 1]) else np.array([0, 1, 0])
    v0 = np.cross(u, helper)
    v0 /= np.linalg.norm(v0)
    w0 = np.cross(u, v0)

    # Rotate v0 by *inclination* around u to get final minor‑axis direction v
    cos_t = np.cos(inclination)
    sin_t = np.sin(inclination)
    v = cos_t * v0 + sin_t * w0            # rotated minor‑axis unit vector

    # Parametric angle
    t = np.linspace(0.0, 2 * np.pi, num_points)

    # Ellipse equation in 3‑D
    ellipse = midpoint[:, None] + u[:, None] * (a * np.cos(t)) + v[:, None] * (b * np.sin(t))
    return ellipse