import numpy as np
from ssapy_toolkit.constants import EARTH_MU  # [35]


def delta_v_semi_major_change_circular(a_initial, a_final):
    r1 = np.asarray(a_initial, dtype=float)
    r2 = np.asarray(a_final, dtype=float)

    v1 = np.sqrt(EARTH_MU / r1)
    v_trans_p = np.sqrt(2 * EARTH_MU * r2 / (r1 * (r1 + r2)))
    delta_v1 = np.abs(v_trans_p - v1)

    v2 = np.sqrt(EARTH_MU / r2)
    v_trans_a = np.sqrt(2 * EARTH_MU * r1 / (r2 * (r1 + r2)))
    delta_v2 = np.abs(v2 - v_trans_a)

    total_delta_v = delta_v1 + delta_v2
    return delta_v1, delta_v2, total_delta_v


def delta_v_inclination_change(r, delta_i_rad):
    v = np.sqrt(EARTH_MU / r)
    delta_v = 2 * v * np.abs(np.sin(delta_i_rad / 2))
    return delta_v


def delta_v_periapsis_change_from_apoapsis(r_apoapsis, current_a, target_periapsis):
    a_new = 0.5 * (r_apoapsis + target_periapsis)
    v_before = np.sqrt(EARTH_MU * (2 / r_apoapsis - 1 / current_a))
    v_after = np.sqrt(EARTH_MU * (2 / r_apoapsis - 1 / a_new))
    return np.abs(v_after - v_before)


def delta_v_apogee_change_from_periapsis(r_periapsis, current_a, target_apogee):
    a_new = 0.5 * (r_periapsis + target_apogee)
    v_before = np.sqrt(EARTH_MU * (2 / r_periapsis - 1 / current_a))
    v_after = np.sqrt(EARTH_MU * (2 / r_periapsis - 1 / a_new))
    return np.abs(v_after - v_before)


def main():
    return {
        "hohmann": delta_v_semi_major_change_circular(7000e3, 12000e3),
        "inclination": delta_v_inclination_change(7000e3, np.radians(10.0)),
        "peri_change": delta_v_periapsis_change_from_apoapsis(12000e3, 9500e3, 7000e3),
        "apo_change": delta_v_apogee_change_from_periapsis(7000e3, 9500e3, 12000e3),
    }


if __name__ == "__main__":
    out = main()
    for k, v in out.items():
        print(k, "->", v)