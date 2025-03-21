import numpy as np
from ssapy import rv, Orbit, SciPyPropagator, AccelKepler
from yeager_utils import hkoe, RGEO, get_times, Time, transfer_plot
from scipy.optimize import minimize, NonlinearConstraint


def transfer_guesser(r1, v1, r2, v2, proximity=100, show=True, status=False):
    t0 = Time("2025-1-1")
    orbit1 = Orbit(r=r1, v=v1, t=t0)
    orbit2 = Orbit(r=r2, v=v2, t=t0)

    # Iteration counter for status updates
    iteration = [0]

    def objective(delta_v):
        v_transfer_initial = v1 + delta_v
        orbit_transfer = Orbit(r=r1, v=v_transfer_initial, t=t0)
        try:
            period = orbit_transfer.period
            if np.isinf(period) or period > 1e7:
                period = max(orbit1.period, orbit2.period)
            times = get_times(duration=(period, 's'), freq=(1, 's'), t0=t0)
        except OverflowError:
            period = max(orbit1.period, orbit2.period)
            times = get_times(duration=(period, 's'), freq=(1, 's'), t0=t0)

        try:
            r_transfer, _ = rv(orbit_transfer, time=times)
        except RuntimeError:
            if status:
                print("KeplerianPropagator failed, using SciPyPropagator")
            r_transfer, _ = rv(orbit_transfer, time=times, propagator=SciPyPropagator(AccelKepler()))

        distances = np.linalg.norm(r_transfer - r2, axis=1)
        return np.min(distances)

    def callback(xk):
        if status:
            iteration[0] += 1
            current_distance = objective(xk)
            print(f"Iteration {iteration[0]}: Closest guess = {current_distance:.6f} meters")

    constraint = NonlinearConstraint(lambda dv: np.linalg.norm(dv), -10000, 10000)
    delta_v0 = np.zeros(3)

    result = minimize(
        objective,
        delta_v0,
        method='SLSQP',
        constraints=[constraint],
        callback=callback,
        options={'disp': status}  # Set disp to status to control optimization output
    )
    delta_v_initial = result.x

    v_transfer_initial = v1 + delta_v_initial
    orbit_transfer = Orbit(r=r1, v=v_transfer_initial, t=t0)
    times = get_times(duration=(orbit1.period, 's'), freq=(1, 's'), t0=t0)

    try:
        r_transfer, v_transfer = rv(orbit_transfer, time=times)
    except RuntimeError:
        if status:
            print("KeplerianPropagator failed, using another for transfer")
        r_transfer, v_transfer = rv(orbit_transfer, time=times, propagator=SciPyPropagator(AccelKepler()))

    distances = np.linalg.norm(r_transfer - r2, axis=1)
    min_dist_idx = np.argmin(distances)
    min_distance = distances[min_dist_idx]
    t_arrival = times[min_dist_idx]
    v_transfer_arrival = v_transfer[min_dist_idx]

    if min_distance > proximity:
        t_refine = np.linspace(t_arrival - 10, t_arrival + 10, 200)
        try:
            r_refine, v_refine = rv(orbit_transfer, time=t_refine)
        except RuntimeError:
            if status:
                print("KeplerianPropagator failed in refinement, using RK4Propagator")
            r_refine, v_refine = rv(orbit_transfer, time=t_refine, propagator=SciPyPropagator(AccelKepler(), h=1))
        distances_refine = np.linalg.norm(r_refine - r2, axis=1)
        min_dist_idx_refine = np.argmin(distances_refine)
        min_distance = distances_refine[min_dist_idx_refine]
        t_arrival = t_refine[min_dist_idx_refine]
        v_transfer_arrival = v_refine[min_dist_idx_refine]

    delta_v_arrival = v2 - v_transfer_arrival

    if status:
        if result.success and min_distance < proximity:
            print(f"Success! Minimum distance to r2 = {min_distance:.6f} meters (within {proximity} m tolerance)")
            print(f"Initial delta_v = {delta_v_initial} m/s")
            print(f"Transfer initial velocity = {v_transfer_initial} m/s")
            print(f"Transfer velocity at arrival = {v_transfer_arrival} m/s")
            print(f"Target velocity (v2) = {v2} m/s")
            print(f"Arrival delta_v = {delta_v_arrival} m/s")
            print(f"Time of closest approach = {t_arrival - times[0]:.2f} seconds after t0")
        else:
            print(f"Optimization failed or tolerance ({proximity} m) not met.")
            print(f"Best initial delta_v = {delta_v_initial} m/s")
            print(f"Minimum distance achieved = {min_distance:.6f} meters")
            print(f"Best arrival delta_v = {delta_v_arrival} m/s")

    if show:
        transfer_plot(r1, v1, r_transfer[0], v_transfer[0], r2, v2, show=show)
    return r_transfer, v_transfer


if __name__ == '__main__':
    t0 = Time("2025-1-1")
    orbit1 = Orbit.fromKeplerianElements(*hkoe(1 * RGEO, 0.0, 0, 0, 0, 0), t=t0)
    r1, v1 = orbit1.r, orbit1.v
    orbit2 = Orbit.fromKeplerianElements(*hkoe(2 * RGEO, 0.0, 0, 0, 0, 90), t=t0)
    r2, v2 = orbit2.r, orbit2.v
    print(f"Running guess for {r1, v1, r2, v2}")
    transfer_guesser(r1, v1, r2, v2, status=True)  # Pass status=True for test
