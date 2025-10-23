import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from yeager_utils import rv_to_ellipse, pprint, figpath

# ---------- quick sanity check ----------
if __name__ == "__main__":
    r0 = [9000e3, 0.0, 0.0]          # start just outside LEO
    v0 = [0.0, 9.0e3, 2.0e3]         # hyperbolic excess
    result = rv_to_ellipse(r0, v0, num=600)
    pprint(result)

    # Extract XY from result["r"] which is (N,3)
    x = None
    y = None

    if isinstance(result, dict):
        if "r" in result:
            r_arr = np.asarray(result["r"])
            if r_arr.ndim == 2 and r_arr.shape[1] >= 2:
                x, y = r_arr[:, 0], r_arr[:, 1]
        # Optional fallbacks if your API ever returns these
        if x is None and "x" in result and "y" in result:
            x_arr = np.asarray(result["x"])
            y_arr = np.asarray(result["y"])
            if x_arr.shape == y_arr.shape:
                x, y = x_arr, y_arr
    else:
        arr = np.asarray(result)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            x, y = arr[:, 0], arr[:, 1]

    # Plot and save via figpath if possible
    if x is not None and y is not None:
        plt.figure()
        plt.plot(x, y, linewidth=1.5)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("rv_to_ellipse: XY projection")

        outpath = Path(figpath("rv_to_ellipse_xy"))
        if outpath.suffix == "":
            outpath = outpath.with_suffix(".png")
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        print("Saved plot to:", outpath)
        plt.close()
    else:
        print("Could not determine x,y from rv_to_ellipse output; skipping plot.")
