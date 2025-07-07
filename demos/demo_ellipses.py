from yeager_utils import RGEO, ellipse_arc, eccentricity_range
import numpy as np


if __name__ == "__main__":
    P1_demo = np.array([2 * RGEO, 0, 0])
    P2_demo = np.array([-RGEO/4, 0, RGEO])

    # 1. least-eccentric ellipse
    arc, prm = ellipse_arc(P1_demo, P2_demo, n_pts=400, plot=True, inc=0)
    print("Least-eccentric:", prm)

    # 2. admissible eccentricity range
    e_lo, e_hi = eccentricity_range(P1_demo, P2_demo)
    print(f"{e_lo:.6f} < e < {e_hi}")

    # 3. user-chosen eccentricity
    e = 0.9
    arc, prm = ellipse_arc(P1_demo, P2_demo, e=e, n_pts=400, plot=True)
    print(f"e = {e}:", prm)
