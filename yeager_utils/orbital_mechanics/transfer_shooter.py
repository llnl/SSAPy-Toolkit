import numpy as np
from ssapy import rv, Orbit, SciPyPropagator, AccelKepler
from ..constants import EARTH_MU
from ..time import get_times, Time


def transfer_shooter(*args, r1=None, v1=None, r2=None, v2=None, orbit1=None, orbit2=None, tol=1, max_iter=50, plot=False, status=False, accel=AccelKepler()):
    """
    Finds an initial delta-v that will lead to a transfer orbit with an arrival position
    within tol (in meters) of the target position r2.

    This function uses a finite-difference Newton shooting method. At each iteration, the error
    vector is defined as F(delta_v) = r_arrival(delta_v) - r2, where r_arrival is the position
    at the time of closest approach. The Jacobian dF/d(delta_v) is approximated via finite differences,
    and a Newton update is computed.

    Parameters
    ----------
    *args : tuple
        Positional arguments: either (orbit1, orbit2) or (r1, v1, r2, v2).
    r1, v1 : array_like, optional
        Initial position and velocity vectors (m, m/s).
    r2 : array_like, optional
        Target position vector (m).
    v2 : array_like, optional
        Target velocity vector (m/s). If not provided and r2 is given, assumes a circular orbit at r2.
    orbit1 : ssapy.Orbit, optional
        Initial orbit object; if provided, r1 and v1 are extracted from it.
    orbit2 : ssapy.Orbit, optional
        Target orbit object; if provided, r2 and v2 are extracted from it.
    tol : float, optional
        Tolerance (in meters) for the arrival position error. Default is 1.
    max_iter : int, optional
        Maximum number of iterations for the Newton method. Default is 50.
    show : bool, optional
        If True, displays a plot of the transfer orbit. Default is False.
    status : bool, optional
        If True, prints iteration details. Default is False.

    Returns
    -------
    dict
        Dictionary containing:
        - 'initial': Orbit object for the initial orbit
        - 'final': Orbit object for the target orbit
        - 'transfer': Orbit object for the transfer orbit
        - '|delta_v1|': Magnitude of initial delta-V (m/s)
        - '|delta_v2|': Magnitude of final delta-V (m/s)
        - 'delta_v1': Initial delta-V vector (m/s)
        - 'delta_v2': Final delta-V vector (m/s)
        - 'r_transfer': Position array of the transfer orbit up to closest approach (m)
        - 'v_transfer': Velocity array of the transfer orbit up to closest approach (m/s)
        - 'tof': Time of flight to closest approach (s)
        - 't_to_transfer': Time to transfer start (s, always 0 here)
        - 'error': Final position error at closest approach (m)
        - 'fig': Matplotlib figure object (if show=True)

    Raises
    ------
    ValueError
        If input arguments are invalid or insufficient.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """

    # Handle positional arguments
    if args:
        if len(args) == 2:
            arg1, arg2 = args
            if isinstance(arg1, Orbit) and isinstance(arg2, Orbit):
                orbit1, orbit2 = arg1, arg2
            else:
                raise ValueError("Two positional arguments must be Orbit objects (orbit1, orbit2)")
        elif len(args) == 3:
            r1, v1, r2 = args
        elif len(args) == 4:
            r1, v1, r2, v2 = args
        else:
            raise ValueError("Positional arguments must be 2 (orbit1, orbit2) or 4 (r1, v1, r2, v2)")

    # Determine t0 and extract state vectors
    t0 = Time("2025-1-1")

    # Handle initial state
    if orbit1 is not None:
        if not isinstance(orbit1, Orbit):
            raise ValueError("orbit1 must be an ssapy.Orbit object")
        r1 = orbit1.r
        v1 = orbit1.v
        t0 = Time(orbit1.t, format='gps')  # Use orbit1's time if provided
    elif r1 is None or v1 is None:
        raise ValueError("Must provide either orbit1 or both r1 and v1")

    # Handle target state
    if orbit2 is not None:
        if not isinstance(orbit2, Orbit):
            raise ValueError("orbit2 must be an ssapy.Orbit object")
        r2 = orbit2.r
        v2 = orbit2.v
    elif v2 is None:
        r2 = np.asarray(r2)
        r2_norm = np.linalg.norm(r2)
        v_circ = np.sqrt(EARTH_MU / r2_norm)

        # 1) orbit‑1 plane tangent
        n1 = np.cross(r1, v1)
        tangent = np.cross(n1, r2)
        if np.linalg.norm(tangent) < 1e-8:
            # 2) fallback: project v1 onto plane ⟂ r2
            proj = (np.dot(v1, r2) / r2_norm**2) * r2
            tangent = v1 - proj
            if np.linalg.norm(tangent) < 1e-8:
                # 3) ultimate fallback: XY‑plane tangent at r2
                # ensure r2 isn’t on z-axis
                if np.linalg.norm(r2[:2]) < 1e-8:
                    raise ValueError("r2 lies on z‑axis; cannot define XY tangent.")
                # tangent = [-y, x, 0]
                tangent = np.array([-r2[1], r2[0], 0.0])

        tangent /= np.linalg.norm(tangent)
        v2 = v_circ * tangent
    elif r2 is None:
        raise ValueError("Must provide either orbit2 or both r2")

    delta_v = np.zeros(3)
    eps = 1e-6  # finite difference step

    def propagate(delta_v_in):
        """Propagate orbit with current delta_v and return arrival position and velocity."""
        v_transfer = v1 + delta_v_in
        orbit_transfer = Orbit(r=r1, v=v_transfer, t=t0)
        try:
            period = orbit_transfer.period
            if np.isinf(period) or period > 1e7:
                period = 2 * 3600  # 2 hours fallback

            times = get_times(duration=(float(period), 'sec'), freq=(1, 'sec'), t0=t0)
        except OverflowError:
            period = 2 * 3600
            times = get_times(duration=(period, 'sec'), freq=(1, 'sec'), t0=t0)
        try:
            r_traj, v_traj = rv(orbit_transfer, time=times)
        except RuntimeError:
            r_traj, v_traj = rv(orbit_transfer, time=times, propagator=SciPyPropagator(accel))
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

        # Compute Jacobian J ~ dF/d(delta_v) using finite differences
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
        r_transfer, v_transfer = rv(orbit_transfer, time=times, propagator=SciPyPropagator(accel))

    distances = np.linalg.norm(r_transfer - r2, axis=1)
    closest_idx = np.argmin(distances)
    r_transfer = r_transfer[:closest_idx + 1]
    v_transfer = v_transfer[:closest_idx + 1]
    tof = times[closest_idx].gps - t0.gps

    if v2 is None:
        delta_v2 = None
        delta_v2_mag = None
    else:
        delta_v2 = v2 - v_transfer[closest_idx]
        delta_v2_mag = np.linalg.norm(delta_v2)

    result = {
        'initial': Orbit(r=r1, v=v1, t=t0),
        'final': Orbit(r=r2, v=v2, t=t0 + tof),
        'transfer': orbit_transfer,
        '|delta_v1|': np.linalg.norm(delta_v),
        '|delta_v2|': delta_v2_mag,
        'delta_v1': delta_v,
        'delta_v2': delta_v2,
        'r_transfer': r_transfer,
        'v_transfer': v_transfer,
        'tof': tof,
        't_to_transfer': 0,  # Immediate transfer assumed
        'error': np.linalg.norm(r_arrival - r2)
    }
    if status:
        print(f"Some Results: tof {tof / 60:.0f} min\n|Δv₁| {np.linalg.norm(delta_v)}\n|Δv₂| {delta_v2_mag}\nv1 - vtransfer[0] {np.linalg.norm(v1 - v_transfer[0])}")

    if plot:
        from ..plots import transfer_plot
        fig = transfer_plot(r1, v1, r_transfer, v_transfer, r2, v2, title=f"Transfer time: {result['tof'] / 60:.0f} min\n|Δv₁| {np.linalg.norm(delta_v) / 1e3:.3f}, |Δv₂| {np.linalg.norm(v2 - v_transfer[closest_idx]) / 1e3:.3f} km/s", show=False)
        result['fig'] = fig

    return result
