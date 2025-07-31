import numpy as np
from yeager_utils import rv_to_ellipse, pprint

# ---------- quick sanity check ----------
if __name__ == "__main__":
    r0 = [9000e3, 0.0, 0.0]          # start just outside LEO
    v0 = [0.0, 9.0e3, 2.0e3]         # hyperbolic excess
    result = rv_to_ellipse(r0, v0, num=6)
    pprint(result)
