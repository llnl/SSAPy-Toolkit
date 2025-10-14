import numpy as np
from yeager_utils import EARTH_MU

def delta_v_semi_major_change_circular(a_initial, a_final):
    """
    Delta-v for Hohmann transfer between two circular orbits (in meters).
    Assumes impulsive burns at perigee and apogee of the transfer ellipse.
    """
    r1 = np.asarray(a_initial, dtype=float)
    r2 = np.asarray(a_final, dtype=float)

    v1 = np.sqrt(EARTH_MU / r1)
    v_trans_p = np.sqrt(2 * EARTH_MU * r2 / (r1 * (r1 + r2)))
    delta_v1 = np.abs(v_trans_p - v1)

    v2 = np.sqrt(EARTH_MU / r2)
    v_trans_a = np.sqrt(2 * EARTH_MU * r1 / (r2 * (r1 + r2)))
    delta_v2 = np.abs(v2 - v_trans_a)

    total_delta_v = delta_v1 + delta_v2

    print(f"\n[Semi-major axis change] Transfer from {r1:.0f} m to {r2:.0f} m:")
    print(f" Burn 1 (at r1): v_initial = {v1:.2f} m/s, v_transfer = {v_trans_p:.2f} m/s")
    print(f" Burn 2 (at r2): v_final = {v2:.2f} m/s, v_transfer = {v_trans_a:.2f} m/s")
    print(f"Total Δv: {total_delta_v:.2f} m/s\n")
    return delta_v1, delta_v2, total_delta_v


def delta_v_inclination_change(r, delta_i_rad):
    """
    Delta-v for inclination change at radius r (meters) by angle delta_i_rad (radians).
    Assumes circular orbit with single impulsive maneuver.
    """
    v = np.sqrt(EARTH_MU / r)
    delta_v = 2 * v * np.abs(np.sin(delta_i_rad / 2))

    print(f"[Inclination change] At radius {r:.0f} m, inclination change = {np.degrees(delta_i_rad):.2f}°:")
    print(f" Orbital speed = {v:.2f} m/s, Δv = {delta_v:.2f} m/s\n")

    return delta_v


def delta_v_periapsis_change_from_apoapsis(r_apoapsis, current_a, target_periapsis):
    """
    Delta-v at apoapsis to change periapsis to a new value.
    Inputs in meters. Assumes elliptical orbits and impulsive change.
    """
    a_new = 0.5 * (r_apoapsis + target_periapsis)
    v_before = np.sqrt(EARTH_MU * (2 / r_apoapsis - 1 / current_a))
    v_after = np.sqrt(EARTH_MU * (2 / r_apoapsis - 1 / a_new))
    delta_v = np.abs(v_after - v_before)

    print(f"\n[Periapsis change] At apoapsis {r_apoapsis:.0f} m, raise periapsis to {target_periapsis:.0f} m:")
    print(f" v_before = {v_before:.2f} m/s, v_after = {v_after:.2f} m/s\n")

    return delta_v


def delta_v_apogee_change_from_periapsis(r_periapsis, current_a, target_apogee):
    """
    Delta-v at periapsis to change apogee to a new value.
    Inputs in meters. Assumes elliptical orbits and impulsive change.
    """
    a_new = 0.5 * (r_periapsis + target_apogee)
    v_before = np.sqrt(EARTH_MU * (2 / r_periapsis - 1 / current_a))
    v_after = np.sqrt(EARTH_MU * (2 / r_periapsis - 1 / a_new))
    delta_v = np.abs(v_after - v_before)

    print(f"\n[Apogee change] At periapsis {r_periapsis:.0f} m, raise apogee to {target_apogee:.0f} m:")
    print(f" v_before = {v_before:.2f} m/s, v_after = {v_after:.2f} m/s\n")

    return delta_v


def delta_v_raan_change(r, inclination_rad, delta_raan_rad):
    """
    Delta-v for RAAN (Ω) change in a circular orbit of radius r (meters).
    RAAN change requires plane rotation: ΔΩ * sin(i).
    """
    v = np.sqrt(EARTH_MU / r)
    plane_angle = np.abs(delta_raan_rad * np.sin(inclination_rad))
    delta_v = 2 * v * np.abs(np.sin(plane_angle / 2))

    print(f"\n[RAAN change] At radius {r:.0f} m, inclination = {np.degrees(inclination_rad):.2f}°, ΔRAAN = {np.degrees(delta_raan_rad):.2f}°:")
    print(f" Plane angle = {np.degrees(plane_angle):.2f}°, Δv = {delta_v:.2f} m/s\n")

    return delta_v


if __name__ == "__main__":
    earth_radius = 6378e3

    # 1. Semi-major axis change
    a1 = 7000e3
    a2 = 12000e3
    dv1, dv2, total = delta_v_semi_major_change_circular(a1, a2)

    # 2. Inclination change of 10° at 700 km altitude
    r_orbit = earth_radius + 700e3
    dv_inc = delta_v_inclination_change(r_orbit, np.radians(10.0))

    # 3. Raise periapsis from 500 km to 700 km at apoapsis = 2000 km
    r_apo = earth_radius + 2000e3
    current_a = 0.5 * (r_apo + (earth_radius + 500e3))
    dv_peri = delta_v_periapsis_change_from_apoapsis(r_apo, current_a, earth_radius + 700e3)

    # 4. Raise apogee from 2000 km to 3000 km at periapsis = 500 km
    r_peri = earth_radius + 500e3
    current_a2 = 0.5 * (r_peri + (earth_radius + 2000e3))
    dv_apo = delta_v_apogee_change_from_periapsis(r_peri, current_a2, earth_radius + 3000e3)

    # 5. RAAN change of 20° at 700 km altitude with 45° inclination
    delta_raan = np.radians(20.0)
    inclination = np.radians(45.0)
    dv_raan = delta_v_raan_change(r_orbit, inclination, delta_raan)
