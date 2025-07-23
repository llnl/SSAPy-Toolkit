from yeager_utils import RGEO, ellipse_arc
import numpy as np

if __name__ == "__main__":
    P1 = np.array([RGEO, 0, 0])
    P2 = np.array([-RGEO, -1*RGEO, 10 * RGEO])

    for ccw in [False, True]:
        print(f"CCW: {ccw}")
        # 1. least-eccentric ellipse
        arc, vel, times, info = ellipse_arc(
            P1, P2,
            n_pts=400,
            plot=True,
            inc=0,
            ccw=ccw,
            debug=True
        )
        print(info)
        # print out the parameters
        # print("Least-eccentric parameters:", prm)
        # print(f"Total flight time: {times[-1]:.3f} seconds\n")

        def print_samples(label, rs, vs, ts):
            print(f"--- {label} ---")
            for i, (r, v_i, t_i) in enumerate(zip(rs, vs, ts), start=1):
                print(f" Sample {i:1d}:")
                print(f"   r = {r}")
                print(f"   v = {v_i}")
                print(f"   t = {t_i:.6f} s")
            print()

        # # First i
        # i = 1
        # print_samples("First 3 samples",
        #             arc[:i], vel[:i], times[:i])

        # # Last i
        # print_samples("Last 3 samples",
        #             arc[-i:], vel[-i:], times[-i:])

    # Solving a transfer.
    from yeager_utils import transfer_ellipse, VGEO
    result = transfer_ellipse(P1, P2, v=[0, VGEO, 0], try_both_dirs=True)

    # --- SSAPy recreation test for both CCW and CW arcs ---
    from ssapy.simple import ssapy_orbit
    from ssapy.plotUtils import orbit_plot
    import numpy as np

    # Run ellipse_arc twice and collect results
    results = {}
    for ccw in [False, True]:
        print(f"CCW: {ccw}")
        arc, vel, times, info = ellipse_arc(
            P1, P2,
            n_pts=400,
            plot=True,
            inc=0,
            ccw=ccw,
            debug=True
        )
        results[ccw] = (arc, vel, times, info)

    # 1️⃣ Reconstruct using initial state vectors
    ssapy_orbits_sv = []
    for ccw, (arc, vel, times, info) in results.items():
        print(ccw)
        r0, v0 = info['r0'], info['v0']
        from ssapy import Orbit
        from yeager_utils import get_times
        orbit = Orbit(r=r0, v=v0, t=get_times(duration=(times[-1], 's'), freq=(len(times), 's'))[0])
        r_ss, v_ss, t_ss = ssapy_orbit(
            r=r0, v=v0,
            duration=(times[-1], 's'), freq=(len(times), 's')
        )
        ssapy_orbits_sv.append((ccw, r_ss, t_ss))

    fig_sv, axarr_sv = orbit_plot(
        [r for _, r, _ in ssapy_orbits_sv],
        t=[t for _, _, t in ssapy_orbits_sv],
        title="SSAPy Reconstructions via State Vectors",
        show=True
    )

    # 3 Reconstruct using Orbit
    ssapy_orbits_ke = []
    for ccw, (arc, vel, times, info) in results.items():
        print(ccw)
        mod = 0
        if not ccw:
            mod = np.pi
        r_ss, v_ss, t_ss = ssapy_orbit(
            a=orbit.a, e=info['e'], i=orbit.i, raan=orbit.raan, pa=orbit.pa, ta=orbit.trueAnomaly,
            duration=(times[-1], 's'), freq=(len(times), 's')
        )
        ssapy_orbits_ke.append((ccw, r_ss, t_ss))

    # Plot both reconstructed orbits
    fig_ke, axarr_ke = orbit_plot(
        [r for _, r, _ in ssapy_orbits_ke],
        t=[t for _, _, t in ssapy_orbits_ke],
        title="SSAPy Reconstructions via Orbit",
        show=True
    )

    # 2️⃣ Reconstruct using Keplerian elements
    ssapy_orbits_ke = []
    for ccw, (arc, vel, times, info) in results.items():
        print(ccw)
        print(info)
        r_ss, v_ss, t_ss = ssapy_orbit(
            a=info['a'], e=info['e'], i=info['i'], raan=info['raan'], pa=info['ap'], ta=info['ta'],
            duration=(times[-1], 's'), freq=(len(times), 's')
        )
        ssapy_orbits_ke.append((ccw, r_ss, t_ss))

    # Plot both reconstructed orbits
    fig_ke, axarr_ke = orbit_plot(
        [r for _, r, _ in ssapy_orbits_ke],
        t=[t for _, _, t in ssapy_orbits_ke],
        title="SSAPy Reconstructions via Keplerian Elements",
        show=True
    )
