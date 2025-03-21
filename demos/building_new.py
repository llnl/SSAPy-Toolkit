import numpy as np
from ssapy import rv, Orbit, SciPyPropagator, AccelKepler
from yeager_utils import hkoe, RGEO, get_times, Time, transfer_plot


def transfer_shooter(r1, v1, r2, v2, tol=100, max_iter=50, show=False, status=False):
    """
    Finds an initial delta-v that will lead to a transfer orbit with an arrival position
    within tol (in meters) of the target position r2.

    This function uses a finite-difference Newton shooting method. At each iteration, the error
    vector is defined as F(delta_v) = r_arrival(delta_v) - r2, where r_arrival is the position
    at the time of closest approach. The Jacobian dF/d(delta_v) is approximated via finite differences,
    and a Newton update is computed.

    Parameters
    ----------
    r1, v1 : array_like
        Initial position and velocity vectors.
    r2, v2 : array_like
        Target position and velocity vectors.
    tol : float, optional
        Tolerance (in meters) for the arrival position error. Default is 100.
    max_iter : int, optional
        Maximum number of iterations for the Newton method.
    status : bool, optional
        If True, prints iteration details.

    Returns
    -------
    r_transfer, v_transfer : array_like
        Propagated transfer orbit states (position and velocity arrays) for the final guess.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    t0 = Time("2025-1-1")
    delta_v = np.zeros(3)
    eps = 1e-6  # finite difference step

    def propagate(delta_v_in):
        """Propagate orbit with current delta_v and return arrival position and velocity."""
        v_transfer = v1 + delta_v_in
        orbit_transfer = Orbit(r=r1, v=v_transfer, t=t0)
        # Use the period of orbit1 to set a propagation duration.
        try:
            period = orbit_transfer.period
            if np.isinf(period) or period > 1e7:
                # fall back on a reasonable period
                period = 2 * 3600  # 2 hours, for example
            times = get_times(duration=(period, 's'), freq=(1, 's'), t0=t0)
        except OverflowError:
            period = 2 * 3600
            times = get_times(duration=(period, 's'), freq=(1, 's'), t0=t0)
        try:
            r_traj, v_traj = rv(orbit_transfer, time=times)
        except RuntimeError:
            r_traj, v_traj = rv(orbit_transfer, time=times, propagator=SciPyPropagator(AccelKepler()))
        # Find the time of closest approach to r2
        distances = np.linalg.norm(r_traj - r2, axis=1)
        idx_min = np.argmin(distances)
        return r_traj[idx_min], v_traj[idx_min], times[idx_min]

    # Initial propagation and error
    r_arrival, v_arrival, t_arrival = propagate(delta_v)
    error = r_arrival - r2

    for it in range(max_iter):
        error_norm = np.linalg.norm(error)
        if status:
            print(f"Iteration {it}: Error norm = {error_norm:.6f} m")
        if error_norm < tol:
            break

        # Compute Jacobian J ~ dF/d(delta_v) using finite differences.
        J = np.zeros((3, 3))
        for i in range(3):
            delta = np.zeros(3)
            delta[i] = eps
            r_arrival_pert, _, _ = propagate(delta_v + delta)
            F_pert = r_arrival_pert - r2
            J[:, i] = (F_pert - error) / eps

        # Solve for the update: J * delta = -error
        try:
            delta_update = np.linalg.solve(J, -error)
        except np.linalg.LinAlgError:
            if status:
                print("Jacobian is singular. Exiting iteration.")
            break

        # Update delta_v
        delta_v = delta_v + delta_update
        # Recompute error
        r_arrival, v_arrival, t_arrival = propagate(delta_v)
        error = r_arrival - r2

    else:
        if status:
            print("Maximum iterations reached without meeting tolerance.")

    # Compute final transfer orbit using the converged delta_v
    v_transfer_initial = v1 + delta_v
    orbit_transfer = Orbit(r=r1, v=v_transfer_initial, t=t0)
    try:
        period = orbit_transfer.period
        if np.isinf(period) or period > 1e7:
            period = 2 * 3600
        times = get_times(duration=(period, 's'), freq=(1, 's'), t0=t0)
    except OverflowError:
        period = 2 * 3600
        times = get_times(duration=(period, 's'), freq=(1, 's'), t0=t0)
    try:
        r_transfer, v_transfer = rv(orbit_transfer, time=times)
    except RuntimeError:
        r_transfer, v_transfer = rv(orbit_transfer, time=times, propagator=SciPyPropagator(AccelKepler()))

    if status:
        final_error = np.linalg.norm(r_arrival - r2)
        print("Converged:")
        print(f"Final delta_v = {delta_v} m/s")
        print(f"Final arrival error = {final_error:.6f} m at time {t_arrival.gps - times[0].gps:.0f} s after t0")
        print(f"Arrival velocity = {v_arrival} m/s")
        arrival_delta_v = v2 - v_arrival
        print(f"Required correction at arrival = {arrival_delta_v} m/s")

    if show:
        transfer_plot(r1, v1, r_transfer[0], v_transfer[0], r2, v2, show=show)

    return r_transfer, v_transfer


if __name__ == '__main__':
    t0 = Time("2025-1-1")
    orbit1 = Orbit.fromKeplerianElements(*hkoe(1 * RGEO, 0.0, 0, 0, 0, 0), t=t0)
    r1, v1 = orbit1.r, orbit1.v
    orbit2 = Orbit.fromKeplerianElements(*hkoe(2 * RGEO, 0.0, 0, 0, 0, 90), t=t0)
    r2, v2 = orbit2.r, orbit2.v
    print(f"Running Newton shooting method for transfer from\n{r1, v1}\nto\n{r2, v2}\n")
    transfer_shooter(r1, v1, r2, v2, tol=100, max_iter=50, show=True, status=True)
