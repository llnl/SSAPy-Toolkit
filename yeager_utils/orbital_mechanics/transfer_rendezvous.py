import numpy as np
from ssapy import rv, Orbit, SciPyPropagator, AccelKepler
from ..constants import EARTH_MU
from ..time import get_times, Time

def transfer_rendezvous(orbit1, orbit2, tol=1, max_iter=50, plot=False, status=False):
    """
    Finds the delta-v that leads to a transfer rendezvous with a moving target orbit.
    Uses a finite-difference Newton shooting method to match spacecraft and target positions at the same epoch.

    Parameters
    ----------
    orbit1 : ssapy.Orbit
        Initial orbit of the chaser at departure epoch.
    orbit2 : ssapy.Orbit
        Target orbit (any epoch). Will be propagated to rendezvous epoch.
    tol : float, optional
        Position tolerance (m). Default is 1 m.
    max_iter : int, optional
        Maximum Newton iterations. Default is 50.
    plot : bool, optional
        If True, plots the final transfer. Default is False.
    status : bool, optional
        If True, prints iteration info. Default is False.

    Returns
    -------
    dict
        - 'initial': Orbit object for the chaser at departure
        - 'final': Orbit object for the target at rendezvous
        - 'transfer': Orbit object for the transfer trajectory
        - '|delta_v1|': Magnitude of initial Δv (m/s)
        - '|delta_v2|': Magnitude of terminal Δv (m/s)
        - 'delta_v1': Initial Δv vector (m/s)
        - 'delta_v2': Terminal Δv vector (m/s)
        - 'r_transfer': Array of spacecraft positions up to intercept (m)
        - 'v_transfer': Array of spacecraft velocities up to intercept (m/s)
        - 'tof': Time of flight (s)
        - 'error': Final position error (m)
        - 'fig': Figure object if plot=True
    """
    # Extract initial state and epoch
    r1 = orbit1.r
    v1 = orbit1.v
    t0 = Time(orbit1.t, format='gps')

    # Prepare target orbit for propagation
    # We'll sample it at the same times as the transfer orbit
    delta_v = np.zeros(3)
    eps = 1e-6  # finite-difference step

    def propagate(delta_v_in):
        """Propagate transfer orbit and compute rendezvous error with moving target."""
        # Build transfer orbit from r1, v1 + Δv at t0
        v_transfer0 = v1 + delta_v_in
        orb_tr = Orbit(r=r1, v=v_transfer0, t=t0)

        # Estimate a propagation window (use transfer period or fallback)
        try:
            period = orb_tr.period
            if np.isinf(period) or period > 1e7:
                period = 2 * 3600  # fallback to 2 h
        except OverflowError:
            period = 2 * 3600

        # Sample at 1 s intervals
        times = get_times(duration=(float(period), 'sec'), freq=(1, 'sec'), t0=t0)

        # Propagate transfer and target orbits
        try:
            r_traj, v_traj = rv(orb_tr, time=times)
        except RuntimeError:
            r_traj, v_traj = rv(orb_tr, time=times, propagator=SciPyPropagator(AccelKepler()))

        # Propagate target to the same times
        try:
            r2_traj, v2_traj = rv(orbit2, time=times)
        except RuntimeError:
            r2_traj, v2_traj = rv(orbit2, time=times, propagator=SciPyPropagator(AccelKepler()))

        # Find minimum separation index
        distances = np.linalg.norm(r_traj - r2_traj, axis=1)
        idx_min = np.argmin(distances)
        return (r_traj[idx_min], v_traj[idx_min],
                r2_traj[idx_min], v2_traj[idx_min],
                times[idx_min], r_traj[:idx_min+1], v_traj[:idx_min+1])

    # Initial propagation and error
    r_arr, v_arr, r2_arr, v2_arr, t_arr, r_seq, v_seq = propagate(delta_v)
    error = r_arr - r2_arr

    for it in range(max_iter):
        err_norm = np.linalg.norm(error)
        if status:
            print(f"Iter {it}: error = {err_norm:.6f} m")
        if err_norm < tol:
            break

        # Build Jacobian via finite differences
        J = np.zeros((3, 3))
        for i in range(3):
            d = np.zeros(3)
            d[i] = eps
            r_p, v_p, r2_p, _, t_p, *_ = propagate(delta_v + d)
            Fp = r_p - r2_p
            J[:, i] = (Fp - error) / eps

        # Solve J Δ = –error
        try:
            delta_update = np.linalg.solve(J, -error)
        except np.linalg.LinAlgError:
            if status:
                print("Jacobian singular; stopping.")
            break

        delta_v += delta_update
        r_arr, v_arr, r2_arr, v2_arr, t_arr, r_seq, v_seq = propagate(delta_v)
        error = r_arr - r2_arr
    else:
        if status:
            print("Reached max iterations without convergence.")

    # Build final transfer orbit for plotting and Δv₂
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
    # Locate intercept in the full trajectory
    try:
        r2_full, v2_full = rv(orbit2, time=times_full)
    except RuntimeError:
        r2_full, v2_full = rv(orbit2, time=times_full, propagator=SciPyPropagator(AccelKepler()))
    dist_full = np.linalg.norm(r_full - r2_full, axis=1)
    idx_closest = np.argmin(dist_full)
    r_transfer = r_full[: idx_closest + 1]
    v_transfer = v_full[: idx_closest + 1]
    tof = times_full[idx_closest].gps - t0.gps

    # Compute Δv₂ at rendezvous
    delta_v2 = v2_arr - v_arr
    delta_v2_mag = np.linalg.norm(delta_v2)
    delta_v1_mag = np.linalg.norm(delta_v)

    # Build final orbit objects
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
        'error': np.linalg.norm(error)
    }

    if status:
        print(f"Converged in {it+1} steps: Δv₁ = {delta_v1_mag:.6f} m/s, Δv₂ = {delta_v2_mag:.6f} m/s, TOF = {tof/60:.2f} min")

    if plot:
        from ..plots import transfer_plot
        fig = transfer_plot(r1, v1, r_transfer, v_transfer, r2_arr, v2_arr,
                            title=f"Transfer TOF: {tof/60:.1f} min\n|Δv₁| {delta_v1_mag/1e3:.3f} km/s, |Δv₂| {delta_v2_mag/1e3:.3f} km/s",
                            show=False)
        result['fig'] = fig

    return result
