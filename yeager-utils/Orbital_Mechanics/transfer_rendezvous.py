import numpy as np
from scipy.optimize import differential_evolution
from ssapy import rv, Orbit, SciPyPropagator, AccelKepler
from ..Time_Functions import get_times, Time
from ..Plots import rendezvous_plot


def transfer_rendezvous(orbit1, orbit2, tol=1000, max_iter=1000, plot=False, status=False):
    """
    Finds the delta-v that leads to a transfer rendezvous with a moving target orbit.
    Uses a global gradient-free search to minimize rendezvous position error.

    Parameters
    ----------
    orbit1 : ssapy.Orbit
        Initial orbit of the chaser at departure epoch.
    orbit2 : ssapy.Orbit
        Target orbit (any epoch). Will be propagated to rendezvous epoch.
    tol : float, optional
        Position tolerance (m). Default is 1000 m.
    max_iter : int, optional
        Maximum optimization iterations. Default is 1000.
    plot : bool, optional
        If True, plots the final transfer. Default is False.
    status : bool, optional
        If True, prints optimization info. Default is False.

    Returns
    -------
    dict
        Output including delta-v vectors, orbits, trajectory arrays, and optional plot.
    """
    r1 = orbit1.r
    v1 = orbit1.v
    t0 = Time(orbit1.t, format='gps')

    def propagate(delta_v_in):
        v_transfer0 = v1 + delta_v_in
        orb_tr = Orbit(r=r1, v=v_transfer0, t=t0)
        try:
            period = orb_tr.period
            if np.isinf(period) or period > 1e7:
                period = 48 * 3600
        except OverflowError:
            period = 48 * 3600

        times = get_times(duration=(float(period), 'sec'), freq=(1, 'sec'), t0=t0)
        try:
            r_traj, v_traj = rv(orb_tr, time=times)
        except RuntimeError:
            r_traj, v_traj = rv(orb_tr, time=times, propagator=SciPyPropagator(AccelKepler()))

        try:
            r2_traj, v2_traj = rv(orbit2, time=times)
        except RuntimeError:
            r2_traj, v2_traj = rv(orbit2, time=times, propagator=SciPyPropagator(AccelKepler()))

        distances = np.linalg.norm(r_traj - r2_traj, axis=1)
        idx_min = np.argmin(distances)
        return (r_traj[idx_min], v_traj[idx_min],
                r2_traj[idx_min], v2_traj[idx_min],
                times[idx_min], r_traj[:idx_min+1], v_traj[:idx_min+1])

    def error_func(delta_v_try):
        r_arr, _, r2_arr, _, _, *_ = propagate(delta_v_try)
        return np.linalg.norm(r_arr - r2_arr)

    bounds = [(-2000, 2000)] * 3
    result_opt = differential_evolution(error_func, bounds, tol=1e-3, maxiter=max_iter, disp=status)
    delta_v = result_opt.x

    # Final propagation using optimal delta_v
    r_arr, v_arr, r2_arr, v2_arr, t_arr, r_seq, v_seq = propagate(delta_v)
    error = np.linalg.norm(r_arr - r2_arr)

    # Build final transfer orbit
    v_transfer0 = v1 + delta_v
    orb_tr = Orbit(r=r1, v=v_transfer0, t=t0)
    try:
        period = orb_tr.period
        if np.isinf(period) or period > 1e7:
            period = 2 * 3600
    except OverflowError:
        period = 2 * 3600

    times_full = get_times(duration=(float(period), 'sec'), freq=(1, 'sec'), t0=t0)
    try:
        r_full, v_full = rv(orb_tr, time=times_full)
    except RuntimeError:
        r_full, v_full = rv(orb_tr, time=times_full, propagator=SciPyPropagator(AccelKepler()))

    try:
        r2_full, v2_full = rv(orbit2, time=times_full)
    except RuntimeError:
        r2_full, v2_full = rv(orbit2, time=times_full, propagator=SciPyPropagator(AccelKepler()))

    dist_full = np.linalg.norm(r_full - r2_full, axis=1)
    idx_closest = np.argmin(dist_full)
    r_transfer = r_full[: idx_closest + 1]
    v_transfer = v_full[: idx_closest + 1]
    tof = times_full[idx_closest].gps - t0.gps

    delta_v2 = v2_arr - v_arr
    delta_v2_mag = np.linalg.norm(delta_v2)
    delta_v1_mag = np.linalg.norm(delta_v)

    orb_initial = Orbit(r=r1, v=v1, t=t0)
    orb_final = Orbit(r=r2_arr, v=v2_arr, t=t_arr)

    result = {
        'initial': orb_initial,
        'final': orb_final,
        'transfer': orb_tr,
        '|delta_v1|': delta_v1_mag,
        '|delta_v2|': delta_v2_mag,
        'delta_v1': delta_v,
        'delta_v2': delta_v2,
        'r_transfer': r_transfer,
        'v_transfer': v_transfer,
        'tof': tof,
        'error': error
    }

    if status:
        print(f"Δv₁ = {delta_v1_mag:.6f} m/s, Δv₂ = {delta_v2_mag:.6f} m/s, TOF = {tof/60:.2f} min, Final error = {error:.3f} m")

    if plot:
        fig = rendezvous_plot(
            orbit1.r, orbit1.v,
            r_transfer, v_transfer,
            orbit2.r, orbit2.v,
            title=(
                f"Transfer Rendezvous\n"
                f"TOF: {tof/60:.1f} min | "
                f"|Δv₁| = {delta_v1_mag/1e3:.3f} km/s, "
                f"|Δv₂| = {delta_v2_mag/1e3:.3f} km/s"
            ),
        )
        result['fig'] = fig

    return result
