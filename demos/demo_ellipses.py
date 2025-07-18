from yeager_utils import RGEO, ellipse_arc
import numpy as np

if __name__ == "__main__":
    P1 = np.array([-RGEO, 0, 0])
    P2 = np.array([3 * RGEO, 0, 0])

    for ccw in [False, True]:
        print(f"CCW: {ccw}")
        # 1. least-eccentric ellipse
        arc, vel, times, prm = ellipse_arc(
            P1, P2,
            n_pts=400,
            plot=True,
            inc=0,
            ccw=ccw,
            debug=True
        )

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
