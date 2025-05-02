import numpy as np
from ssapy import rv, Orbit, SciPyPropagator, AccelKepler
from ..constants import EARTH_MU
from ..time import get_times, Time


def transfer_coplanar(*args, r1=None, v1=None, r2=None, v2=None,
                              orbit1=None, orbit2=None,
                              tol=1, max_iter=50, plot=False, status=False):
    """
    Coplanar transfer shooter: enforces Δv lies in the plane of the initial orbit.
    """
    # Handle positional args
    if args:
        if len(args) == 2 and isinstance(args[0], Orbit) and isinstance(args[1], Orbit):
            orbit1, orbit2 = args
        elif len(args) == 3:
            r1, v1, r2 = args
        elif len(args) == 4:
            r1, v1, r2, v2 = args
        else:
            raise ValueError("Positional args = (orbit1,orbit2) or (r1,v1,r2[,v2]).")

    # t0 and initial state
    t0 = Time("2025-1-1")
    if orbit1 is not None:
        r1, v1 = orbit1.r, orbit1.v
        t0 = Time(orbit1.t, format="gps")
    elif r1 is None or v1 is None:
        raise ValueError("Must supply orbit1 or both r1 & v1.")

    # target state
    if orbit2 is not None:
        r2, v2 = orbit2.r, orbit2.v
    elif r2 is None:
        raise ValueError("Must supply orbit2 or r2.")
    elif v2 is None:
        r2 = np.asarray(r2)
        v2 = np.array([0, np.sqrt(EARTH_MU/np.linalg.norm(r2)), 0])

    # plane normal
    h = np.cross(r1, v1)
    h_hat = h / np.linalg.norm(h)
    delta_v = np.zeros(3)
    eps = 1e-6

    def propagate(dv):
        v_tr = v1 + dv
        ot = Orbit(r=r1, v=v_tr, t=t0)
        try:
            T = ot.period
            if np.isinf(T) or T > 1e7:
                T = 2*3600
            times = get_times(duration=(float(T), "sec"), freq=(1, "sec"), t0=t0)
        except OverflowError:
            times = get_times(duration=(2*3600, "sec"), freq=(1, "sec"), t0=t0)
        try:
            rs, vs = rv(ot, time=times)
        except RuntimeError:
            rs, vs = rv(ot, time=times, propagator=SciPyPropagator(AccelKepler()))
        dists = np.linalg.norm(rs - r2, axis=1)
        i = np.argmin(dists)
        return rs[i], vs[i], times[i]

    # initial
    r_arr, v_arr, t_arr = propagate(delta_v)
    err = r_arr - r2

    for it in range(max_iter):
        en = np.linalg.norm(err)
        if status:
            print(f"Iter {it}: |err|={en:.3f} m")
        if en < tol:
            break
        J = np.zeros((3, 3))
        for i in range(3):
            d = np.zeros(3); d[i] = eps
            r_p, _, _ = propagate(delta_v + d)
            J[:, i] = (r_p - r2 - err) / eps
        try:
            du = np.linalg.solve(J, -err)
        except np.linalg.LinAlgError:
            if status:
                print("Singular J, stopping.")
            break
        # enforce coplanarity
        du -= np.dot(du, h_hat) * h_hat
        delta_v += du
        r_arr, v_arr, t_arr = propagate(delta_v)
        err = r_arr - r2
    else:
        if status:
            print("Max iter reached.")

    # final transfer arc
    v_tr0 = v1 + delta_v
    otf = Orbit(r=r1, v=v_tr0, t=t0)
    try:
        T = otf.period
        if np.isinf(T) or T > 1e7:
            T = 2*3600
        tm = get_times(duration=(T, "s"), freq=(1, "s"), t0=t0)
    except OverflowError:
        tm = get_times(duration=(2*3600, "s"), freq=(1, "s"), t0=t0)
    try:
        rs_full, vs_full = rv(otf, time=tm)
    except RuntimeError:
        rs_full, vs_full = rv(otf, time=tm, propagator=SciPyPropagator(AccelKepler()))
    ds = np.linalg.norm(rs_full - r2, axis=1)
    idx = np.argmin(ds)
    rs_tr, vs_tr = rs_full[: idx+1], vs_full[: idx+1]
    tof = tm[idx].gps - t0.gps

    dv2 = None; dv2_mag = None
    if v2 is not None:
        dv2 = v2 - vs_tr[-1]
        dv2_mag = np.linalg.norm(dv2)

    result = {
        "initial": Orbit(r=r1, v=v1, t=t0),
        "final": Orbit(r=r2, v=v2, t=t0 + tof),
        "transfer": otf,
        "|delta_v1|": np.linalg.norm(delta_v),
        "|delta_v2|": dv2_mag,
        "delta_v1": delta_v,
        "delta_v2": dv2,
        "r_transfer": rs_tr,
        "v_transfer": vs_tr,
        "tof": tof,
        "t_to_transfer": 0,
        "error": np.linalg.norm(err),
    }

    if plot:
        from ..plots import transfer_plot
        fig = transfer_plot(r1, v1, rs_tr, vs_tr, r2, v2,
                            title=f"ToF {tof/60:.0f} min |Δv₁| {np.linalg.norm(delta_v)/1e3:.3f} km/s",
                            show=False)
        result["fig"] = fig

    if status:
        print(f"Done: |Δv₁|={np.linalg.norm(delta_v):.3f}, |Δv₂|={dv2_mag:.3f}")

    return result
