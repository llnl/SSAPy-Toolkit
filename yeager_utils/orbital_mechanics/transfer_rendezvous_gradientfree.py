import numpy as np
from scipy.optimize import differential_evolution
from ssapy import rv, Orbit, SciPyPropagator, AccelKepler
from ..constants import EARTH_RADIUS
from ..time import get_times, Time
from ..plots import rendezvous_plot


def transfer_rendezvous_gradientfree(
    orbit1,
    orbit2,
    tol=1,
    max_iter=1000,
    plot=False,
    status=False,
    MIN_ALTITUDE=EARTH_RADIUS,
    accel=AccelKepler()
):
    """
    Finds the delta-v that leads to a transfer rendezvous with a moving target.

    This function uses a gradient-free optimization (differential evolution)
    to find a single impulsive maneuver (Δv) that brings a chaser (orbit1) into 
    position to rendezvous with a moving target (orbit2) with minimal distance error.

    The target (orbit2) can be specified as:
      1. An `ssapy.Orbit` instance (i.e. fixed Keplerian orbit).
      2. A callable function `f(t)` that returns `(r, v)` in GCRF at time `t`,
         where `t` is an instance of your Time class (or convertible to what `f` needs).

    To pass a target function that requires additional arguments, use:

        from functools import partial
        def target_fn(t, arg1, arg2):
            ...
            return r, v

        target_partial = partial(target_fn, arg1=..., arg2=...)
        transfer_rendezvous_gradientfree(orbit1, target_partial)

    Parameters
    ----------
    orbit1 : ssapy.Orbit
        The chaser's initial orbit at the departure epoch.
    orbit2 : Union[ssapy.Orbit, Callable[[Time], Tuple[np.ndarray, np.ndarray]]]
        The target orbit or a time-dependent function of the target's (r, v).
    tol : float, optional
        Desired position tolerance for rendezvous in meters. Default is 1 meter.
    max_iter : int, optional
        Maximum number of iterations for the optimizer. Default is 1000.
    plot : bool, optional
        If True, include a rendezvous trajectory plot in the result.
    status : bool, optional
        If True, print optimization progress and final result summary.

    Returns
    -------
    dict
        Dictionary with results including:
        - delta_v1, delta_v2: velocity vectors (initial burn, final error)
        - |delta_v1|, |delta_v2|: norms of those vectors
        - initial, final, transfer: ssapy.Orbit instances
        - r_transfer, v_transfer: arrays of chaser trajectory to rendezvous
        - tof: time-of-flight in seconds
        - error: position error at closest approach (in meters)
        - fig: plot (if plot=True)
    """

    r1 = orbit1.r
    v1 = orbit1.v
    t0 = Time(orbit1.t, format='gps')

    def get_target_rv(target, times):
        if isinstance(target, Orbit):
            try:
                return rv(target, time=times)
            except RuntimeError:
                return rv(target, time=times, propagator=SciPyPropagator(accel))
        if callable(target):
            r_list, v_list = [], []
            for t in times:
                r2, v2 = target(t)
                r_list.append(r2)
                v_list.append(v2)
            return np.vstack(r_list), np.vstack(v_list)
        raise ValueError("orbit2 must be an ssapy.Orbit or a callable returning (r, v)")

    def propagate(delta_v_in):
        v_transfer0 = v1 + delta_v_in
        orb_tr = Orbit(r=r1, v=v_transfer0, t=t0)

        try:
            period = orb_tr.period
            if np.isinf(period) or period > 1e7:
                period = 48 * 3600
        except (OverflowError, ValueError):
            period = 48 * 3600

        times = get_times(duration=(float(period), 'sec'), freq=(1, 'sec'), t0=t0)

        try:
            r_traj, v_traj = rv(orb_tr, time=times)
        except RuntimeError:
            r_traj, v_traj = rv(orb_tr, time=times, propagator=SciPyPropagator(accel))

        r2_traj, v2_traj = get_target_rv(orbit2, times)
        distances = np.linalg.norm(r_traj - r2_traj, axis=1)
        idx_min = np.argmin(distances)

        return (
            r_traj[idx_min],
            v_traj[idx_min],
            r2_traj[idx_min],
            v2_traj[idx_min],
            times[idx_min],
            r_traj[:idx_min + 1],
            v_traj[:idx_min + 1],
        )

    def error_func(delta_v_try):
        try:
            _, _, _, _, _, r_path, _ = propagate(delta_v_try)
        except Exception:
            return 1e9  # fail-safe penalty

        radii = np.linalg.norm(r_path, axis=1)
        if np.any(radii < MIN_ALTITUDE):
            return 1e6 + 100 * np.sum(np.clip(MIN_ALTITUDE - radii, 0, None))

        r_arr, _, r2_arr, _, _, *_ = propagate(delta_v_try)
        return np.linalg.norm(r_arr - r2_arr)

    bounds = [(-2000, 2000)] * 3
    result_opt = differential_evolution(
        error_func,
        bounds,
        tol=1e-3,
        maxiter=max_iter,
        disp=status,
    )
    delta_v = result_opt.x

    (
        r_arr,
        v_arr,
        r2_arr,
        v2_arr,
        t_arr,
        r_seq,
        v_seq,
    ) = propagate(delta_v)
    error = np.linalg.norm(r_arr - r2_arr)

    v_transfer0 = v1 + delta_v
    orb_tr = Orbit(r=r1, v=v_transfer0, t=t0)

    try:
        period = orb_tr.period
        if np.isinf(period) or period > 1e7:
            period = 2 * 3600
    except (OverflowError, ValueError):
        period = 2 * 3600

    times_full = get_times(duration=(float(period), 'sec'), freq=(1, 'sec'), t0=t0)
    try:
        r_full, v_full = rv(orb_tr, time=times_full)
    except RuntimeError:
        r_full, v_full = rv(orb_tr, time=times_full, propagator=SciPyPropagator(accel))

    r2_full, v2_full = get_target_rv(orbit2, times_full)
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
        'error': error,
    }

    if status:
        print(
            f"Δv₁ = {delta_v1_mag:.6f} m/s, "
            f"Δv₂ = {delta_v2_mag:.6f} m/s, "
            f"TOF = {tof/60:.2f} min, "
            f"Final error = {error:.3f} m"
        )

    if plot:
        fig = rendezvous_plot(
            orbit1.r,
            orbit1.v,
            r_transfer,
            v_transfer,
            orbit2.r if isinstance(orbit2, Orbit) else None,
            orbit2.v if isinstance(orbit2, Orbit) else None,
            title=(
                f"Transfer Rendezvous\n"
                f"TOF: {tof/60:.1f} min | "
                f"|Δv₁| = {delta_v1_mag/1e3:.3f} km/s, "
                f"|Δv₂| = {delta_v2_mag/1e3:.3f} km/s"
            ),
        )
        result['fig'] = fig

    return result
