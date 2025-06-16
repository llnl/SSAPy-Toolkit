import numpy as np
from ..constants import EARTH_MU
from ..time import to_gps


def _accel_point_earth(r):
    """Two-body Earth gravity (μ / r²) toward Earth centre."""
    x, y, z = r
    r_mag = np.sqrt(x**2 + y**2 + z**2)
    factor = -EARTH_MU / r_mag**3
    return np.array([factor * x, factor * y, factor * z])


def _accel_velocity(v, thrust_mag):
    """Thrust vector along spacecraft velocity."""
    v = np.asarray(v)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return np.zeros(3)
    return thrust_mag * v / norm


def _accel_radial(r, magnitude):
    """Thrust vector along spacecraft radius (outward if magnitude > 0)."""
    r = np.asarray(r)
    norm = np.linalg.norm(r)
    if norm == 0.0:
        return np.zeros(3)
    return magnitude * r / norm


def _accel_inclination(r, v, magnitude):
    """
    Thrust perpendicular to orbital plane (changes inclination).
    Positive magnitude ≈ push toward geocentric north.
    """
    r = np.asarray(r, float)
    norm_r = np.linalg.norm(r)
    if norm_r == 0.0 or magnitude == 0.0:
        return np.zeros(3)

    r_hat = r / norm_r
    z_hat = np.array([0.0, 0.0, 1.0])
    north_dir = z_hat - np.dot(z_hat, r_hat) * r_hat
    norm_nd = np.linalg.norm(north_dir)
    if norm_nd == 0.0:
        return np.zeros(3)
    return magnitude * north_dir / norm_nd


# ---------------------------------------------------------------------------
# Thrust-profile builder
# ---------------------------------------------------------------------------
def _build_profile(profile, t_arr):
    """
    Expand a thrust description into an array of magnitudes
    aligned with t_arr.

    Accepts:
      • None              → all zeros
      • scalar            → constant
      • iterable length N → already per-step
      • tuple/list of segments:
          (start, end, thrust) or (start, thrust),
        where start/end are indices *or* time values (same units as t_arr)
    """
    n = len(t_arr)
    out = np.zeros(n, float)

    if profile is None:
        return out

    if np.isscalar(profile):
        out[:] = float(profile)
        return out

    if isinstance(profile, (list, tuple, np.ndarray)) and len(profile) == n:
        return np.asarray(profile, float)

    # Normalize to list of segments, handling a single tuple
    if isinstance(profile, tuple) and (len(profile) == 2 or len(profile) == 3):
        segments = [profile]
    elif isinstance(profile, (list, tuple)):
        segments = profile
    else:
        raise TypeError("Unsupported profile format")

    for seg in segments:
        if len(seg) == 2:          # (start, thrust) – continuous to end
            start, thrust = seg
            end = None
        elif len(seg) == 3:        # (start, end, thrust)
            start, end, thrust = seg
        else:
            raise ValueError("Segment must be (start, thrust) or (start, end, thrust)")

        # Convert to indices
        start_idx = (
            int(start)
            if isinstance(start, (int, np.integer))
            else int(np.searchsorted(t_arr, start))
        )
        end_idx = (
            n
            if end is None
            else (
                int(end)
                if isinstance(end, (int, np.integer))
                else int(np.searchsorted(t_arr, end))
            )
        )
        if start_idx >= n:
            continue
        out[start_idx:end_idx] += float(thrust)

    return out


# ---------------------------------------------------------------------------
# Leapfrog propagator (velocity-Verlet)
# ---------------------------------------------------------------------------

def leapfrog(
    r0,
    v0,
    t,
    radial=None,
    velocity=None,
    inclination=None,
    accel_gravity=_accel_point_earth,
):
    """
    Symplectic leapfrog integrator with optional thrust along:
      • radial (out/in)         via `radial`
      • velocity (along-track)  via `velocity`
      • normal to plane         via `inclination`

    Parameters
    ----------
    r0, v0 : array_like (3,)
        Initial Cartesian state [m, m/s].
    t : array_like
        Time axis (s or datetime-like).  Uniform Δt required.
    accel_gravity : callable
        Function f(r) → gravitational acceleration (default: 2-body Earth).
    radial, velocity, inclination :
        Thrust profiles (see _build_profile).

    Returns
    -------
    r, v : ndarray
        Arrays of shape (N, 3) with propagated positions and velocities.
    """
    # -- Time array in seconds from epoch -----------------------------------
    t_arr = to_gps(t)
    t_arr = t_arr - t_arr[0]
    n_steps = len(t_arr)

    dt_vals = np.diff(t_arr)
    if not np.allclose(dt_vals, dt_vals[0]):
        raise ValueError("Non-uniform Δt not supported")
    dt = dt_vals[0]

    # -- Expand thrust schedules -------------------------------------------
    r_th = _build_profile(radial,       t_arr)
    v_th = _build_profile(velocity,     t_arr)
    i_th = _build_profile(inclination,  t_arr)

    # -- Allocate state arrays ---------------------------------------------
    r = np.empty((n_steps, 3))
    v = np.empty((n_steps, 3))
    r[0] = np.asarray(r0, float)
    v[0] = np.asarray(v0, float)

    # -- Leapfrog integration ----------------------------------------------
    for i in range(n_steps - 1):
        a0 = (
            accel_gravity(r[i])
            + _accel_radial(r[i],            r_th[i])
            + _accel_velocity(v[i],          v_th[i])
            + _accel_inclination(r[i], v[i], i_th[i])
        )

        v_half = v[i] + 0.5 * dt * a0
        r[i + 1] = r[i] + dt * v_half

        a1 = (
            accel_gravity(r[i + 1])
            + _accel_radial(r[i + 1],            r_th[i + 1])
            + _accel_velocity(v_half,            v_th[i + 1])
            + _accel_inclination(r[i + 1], v_half, i_th[i + 1])
        )
        v[i + 1] = v_half + 0.5 * dt * a1

    return r, v
