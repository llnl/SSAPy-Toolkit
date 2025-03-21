import numpy as np
from ..constants import EARTH_MU, EARTH_RADIUS
from ..integrators import leapfrog


def state_vector_to_conic(r1, r2, v1_t, mu=EARTH_MU):
    """Convert transfer orbit to conic coefficients (ellipse or hyperbola).

    Args:
        r1: Initial position vector [x1, y1] (m)
        r2: Final position vector [x2, y2] (m)
        v1_t: Transfer velocity at r1 [vx, vy] (m/s)
        mu: Gravitational parameter (m^3/s^2), default Earth

    Returns:
        List of [A, B, C, D, E, F] coefficients for the conic equation
    """
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    h = r1[0] * v1_t[1] - r1[1] * v1_t[0]  # Angular momentum (z-component)
    p = h**2 / mu  # Semi-latus rectum

    v1_t_mag = np.linalg.norm(v1_t)
    epsilon = v1_t_mag**2 / 2 - mu / r1_mag  # Specific energy

    a = -mu / (2 * epsilon)  # Semi-major axis (negative for hyperbola)
    e = np.sqrt(1 - p / a) if epsilon < 0 else np.sqrt(1 + p / abs(a))  # Eccentricity

    cos_theta1 = r1[0] / r1_mag
    sin_theta1 = r1[1] / r1_mag
    center_x = -a * e * cos_theta1
    center_y = -a * e * sin_theta1

    if epsilon < 0:  # Ellipse
        b = a * np.sqrt(1 - e**2)  # Semi-minor axis
    else:  # Hyperbola
        b = abs(a) * np.sqrt(e**2 - 1)  # Semi-transverse axis

    A = b**2
    B = 0  # No tilt assumed
    C = a**2
    D = -2 * A * center_x
    E = -2 * C * center_y
    F = A * center_x**2 + C * center_y**2 - a**2 * b**2

    F_adjusted = -(A * 0**2 + B * 0 * 0 + C * 0**2 + D * 0 + E * 0)
    coeffs = [A, B, C, D, E, F_adjusted]

    norm_factor = abs(F_adjusted) if F_adjusted != 0 else 1
    return [c / norm_factor for c in coeffs]


def find_intersection_time(r_start, v_start, r_ref, t_max):
    """Find time when orbit from (r_start, v_start) nearly intersects r_ref."""
    r_orbit, _ = leapfrog(r_start, v_start, t=np.arange(0, t_max, 1))
    distances = np.linalg.norm(r_orbit - r_ref, axis=1)
    idx = np.argmin(distances)
    return idx  # Time index of closest approach in seconds


def transfer_lambertian(r1, v1, r2, v2, MIN_PERIGEE=EARTH_RADIUS + 100000, mu=EARTH_MU):
    """Find transfer conic (ellipse or hyperbola) connecting r1 and r2 with perigee >= 6478 km.

    Args:
        r1: Initial position vector (m)
        v1: Initial velocity vector (m/s)
        r2: Final position vector (m)
        v2: Final velocity vector (m/s)
        mu: Gravitational parameter (m^3/s^2), default Earth

    Returns:
        Tuple (coeffs, v1_t, v2_t, tof, orbit_type):
            - coeffs: [A, B, C, D, E, F] of transfer conic
            - v1_t: Transfer velocity at r1 (m/s)
            - v2_t: Transfer velocity at r2 (m/s)
            - tof: Time of flight from r1 to r2 (s)
            - orbit_type: 'ellipse' or 'hyperbola'
    """
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # Check if either point is below MIN_PERIGEE
    if r1_mag < MIN_PERIGEE or r2_mag < MIN_PERIGEE:
        raise ValueError(f"Cannot satisfy perigee >= 6478 km: r1 = {r1_mag/1000:.0f} km or r2 = {r2_mag/1000:.0f} km below constraint")

    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    dnu = np.arccos(np.clip(cos_dnu, -1.0, 1.0))
    cross_r = np.cross(r1, r2)
    if len(r1) == 3 and cross_r[2] < 0:  # 3D check
        dnu = 2 * np.pi - dnu

    # Minimum energy transfer (Hohmann-like) for initial guess
    s = r1_mag + r2_mag + np.sqrt(np.sum((r1 - r2)**2))
    a_min = s / 4
    tof_min = np.pi * np.sqrt(a_min**3 / mu)

    def lambert_velocity(tof):
        a = (mu * tof**2 / (dnu**2))**(1 / 3)  # Approximate a
        p = r1_mag * r2_mag * np.sin(dnu)**2
        p /= (r1_mag + r2_mag - 2 * np.sqrt(r1_mag * r2_mag) * np.cos(dnu))
        e = np.sqrt(1 - p / a) if p < a else np.sqrt(1 + p / a)
        f = 1 - r2_mag * (1 - np.cos(dnu)) / p
        g = r1_mag * r2_mag * np.sin(dnu) / np.sqrt(mu * p)
        v1_t = (r2 - f * r1) / g
        v2_t = (-r1 + f * r2) / g
        return v1_t, v2_t

    # Initial attempt
    tof = tof_min
    v1_t, v2_t = lambert_velocity(tof)
    h = r1[0] * v1_t[1] - r1[1] * v1_t[0]
    p = h**2 / mu
    v1_t_mag = np.linalg.norm(v1_t)
    epsilon = v1_t_mag**2 / 2 - mu / r1_mag
    a = -mu / (2 * epsilon)
    e = np.sqrt(1 - p / a) if epsilon < 0 else np.sqrt(1 + p / abs(a))
    r_p = a * (1 - e) if epsilon < 0 else abs(a) * (e - 1)

    # Adjust TOF if perigee is too low
    if r_p < MIN_PERIGEE:
        print(f"Initial perigee {r_p/1000:.0f} km < 6478 km, adjusting TOF...")
        tof_values = [tof_min * (1 + 0.1 * i) for i in range(20)] + \
                     [tof_min * i for i in [0.5, 2, 5, 10, 20]]  # Wide range
        for tof in tof_values:
            v1_t, v2_t = lambert_velocity(tof)
            h = r1[0] * v1_t[1] - r1[1] * v1_t[0]
            p = h**2 / mu
            v1_t_mag = np.linalg.norm(v1_t)
            epsilon = v1_t_mag**2 / 2 - mu / r1_mag
            a = -mu / (2 * epsilon)
            e = np.sqrt(1 - p / a) if epsilon < 0 else np.sqrt(1 + p / abs(a))
            r_p = a * (1 - e) if epsilon < 0 else abs(a) * (e - 1)
            if r_p >= MIN_PERIGEE:
                print(f"Adjusted to perigee {r_p/1000:.0f} km >= 6478 km with TOF {tof/60:.0f} min")
                break
        else:
            raise ValueError("No transfer orbit found with perigee >= 6478 km")

    coeffs = state_vector_to_conic(r1, r2, v1_t, mu)
    orbit_type = 'ellipse' if epsilon < 0 else 'hyperbola'
    t_max = 2 * np.pi * np.sqrt(a**3 / mu) if epsilon < 0 else 2 * tof
    tof = find_intersection_time(r1, v1_t, r2, t_max)

    return coeffs, v1_t, v2_t, tof, orbit_type


if __name__ == "__main__":
    from yeager_utils import kepler_to_state, hkoe, RGEO, transfer_plot

    # Initial orbit (LEO)
    r1, v1 = kepler_to_state(*hkoe(0.5 * RGEO, 0.1, 0, 0, 0, 0))
    # Final orbit (another LEO or beyond)
    r2, v2 = kepler_to_state(*hkoe(1.0 * RGEO, 0.5, 0, 0, 0, 10))

    coeffs, v1_t, v2_t, tof, orbit_type = transfer_ellipse(r1, v1, r2, v2)
    state_vectors = [(r1, v1), (r1, v1_t), (r2, v2)]

    print(f"Transfer time: {tof/60:.0f} minutes")
    print(f"Orbit type: {orbit_type}")
    transfer_plot(state_vectors, show=True, c='black',
                  title=f'Transfer time: {tof/60:.0f} minutes ({orbit_type})')
