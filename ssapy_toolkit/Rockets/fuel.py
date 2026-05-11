import numpy as np
from astropy.time import Time

from ssapy.accel import AccelConstNTW

G0 = 9.80665  # m/s^2


def _to_gps_seconds(t):
    if isinstance(t, Time):
        return float(t.gps)
    return float(t)


def _finite_on_intervals(time_breakpoints, t_min=None, t_max=None):
    """
    AccelConstNTW time_breakpoints are [on0, off0, on1, off1, ...] in GPS seconds.
    Returns finite (on, off) intervals, optionally clipped to [t_min, t_max].
    """
    bp = np.asarray(time_breakpoints, dtype=float)

    if bp.size == 0:
        return []

    if np.any(np.diff(bp) < 0):
        raise ValueError("time_breakpoints must be sorted.")

    # If odd length, assume final "off" at +inf
    if (bp.size % 2) == 1:
        bp = np.concatenate([bp, [np.inf]])

    if t_min is not None:
        t_min = float(t_min)
    if t_max is not None:
        t_max = float(t_max)

    intervals = []
    for k in range(0, bp.size, 2):
        on = bp[k]
        off = bp[k + 1]

        if t_min is not None:
            on = max(on, t_min)
        if t_max is not None:
            off = min(off, t_max)

        if np.isfinite(on) and np.isfinite(off) and (off > on):
            intervals.append((on, off))

    return intervals


def estimate_fuel_for_accel_ntw_burn(
    burn: AccelConstNTW,
    m0_kg: float,
    isp_s: float,
    t_min=None,
    t_max=None,
    g0=G0,
    mode="constant_accel",
):
    """
    Estimate propellant usage for an SSAPy AccelConstNTW maneuver.

    mode:
      - "constant_accel" (recommended for AccelConstNTW): commanded acceleration -> exponential mass decay
      - "constant_thrust": constant thrust inferred from initial mass and |a| -> linear mass decay
    """
    m0_kg = float(m0_kg)
    isp_s = float(isp_s)
    g0 = float(g0)

    a_vec = np.asarray(burn.accelntw, dtype=float).reshape(3)
    a_mag = float(np.sqrt(np.einsum("i,i", a_vec, a_vec)))

    t_min_gps = None if t_min is None else _to_gps_seconds(t_min)
    t_max_gps = None if t_max is None else _to_gps_seconds(t_max)

    intervals = _finite_on_intervals(burn.time_breakpoints, t_min=t_min_gps, t_max=t_max_gps)
    burn_time = float(np.sum([off - on for on, off in intervals]))  # seconds

    a_int = a_mag * burn_time  # m/s

    out = {
        "burn_time_s": burn_time,
        "a_mag_mps2": a_mag,
        "a_int_mps": a_int,
        "m0_kg": m0_kg,
    }

    if mode == "constant_accel":
        m_final = m0_kg * np.exp(-a_int / (isp_s * g0))
        out["m_final_kg"] = float(m_final)
        out["m_prop_kg"] = float(m0_kg - m_final)
        return out

    if mode == "constant_thrust":
        thrust_N = m0_kg * a_mag
        mdot = thrust_N / (isp_s * g0)  # kg/s
        m_final = max(m0_kg - mdot * burn_time, 0.0)

        out["thrust_N"] = float(thrust_N)
        out["mdot_kgps"] = float(mdot)
        out["m_final_kg"] = float(m_final)
        out["m_prop_kg"] = float(m0_kg - m_final)
        return out

    raise ValueError("mode must be 'constant_accel' or 'constant_thrust'")


def mass_profile_for_accel_ntw_burn(burn: AccelConstNTW, t_gps, m0_kg: float, isp_s: float, g0=G0):
    """
    Mass vs time on a provided time grid (GPS seconds), using 'constant_accel' interpretation.
    """
    tgps = np.asarray(t_gps, dtype=float)
    if tgps.ndim != 1 or tgps.size < 2:
        raise ValueError("t_gps must be a 1D array with length >= 2")

    a_vec = np.asarray(burn.accelntw, dtype=float).reshape(3)
    a_mag = float(np.sqrt(np.einsum("i,i", a_vec, a_vec)))

    dt = np.diff(tgps)
    t_mid = 0.5 * (tgps[:-1] + tgps[1:])
    ind = np.searchsorted(np.asarray(burn.time_breakpoints, dtype=float), t_mid)
    on = (ind % 2) == 1

    a_int_cum = np.concatenate([[0.0], np.cumsum(a_mag * dt * on)])
    m = float(m0_kg) * np.exp(-a_int_cum / (float(isp_s) * float(g0)))
    return m


if __name__ == "__main__":
    # Example: 0.1 mm/s^2 along-track for 120 s, starting at GPS=1.0e9
    burn = AccelConstNTW(
        accelntw=[0.0, 1e-4, 0.0],
        time_breakpoints=[1.0e9, 1.0e9 + 120.0],
    )

    est_accel = estimate_fuel_for_accel_ntw_burn(
        burn,
        m0_kg=250.0,
        isp_s=220.0,
        mode="constant_accel",
    )
    print("Constant-accel interpretation:")
    for k, v in est_accel.items():
        print(f"  {k}: {v}")

    est_thrust = estimate_fuel_for_accel_ntw_burn(
        burn,
        m0_kg=250.0,
        isp_s=220.0,
        mode="constant_thrust",
    )
    print("\nConstant-thrust interpretation (thrust inferred from m0*|a|):")
    for k, v in est_thrust.items():
        print(f"  {k}: {v}")

    # Optional: mass history on a simple time grid
    t_grid = np.linspace(1.0e9, 1.0e9 + 300.0, 301)  # 1 Hz for 300 s
    m_hist = mass_profile_for_accel_ntw_burn(burn, t_grid, m0_kg=250.0, isp_s=220.0)
    print(f"\nMass after 300 s grid: {m_hist[-1]} kg (burn only affects first 120 s)")
