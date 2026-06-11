"""Optimal transfer search between two orbits for SSAPy.

Where :func:`transfer_ssapy` solves a fixed boundary-value problem (two
states, one time of flight -> the unique connecting two-burn transfer),
:func:`transfer_optimal` searches over the *free* variables of an
orbit-to-orbit transfer -- departure time along orbit 1, time of flight,
arrival phase along orbit 2 (insertion mode), and motion sense -- to find
the transfer that minimizes total delta-v (default) or, given a delta-v
budget, the fastest transfer that fits it.

The search uses a coarse porkchop grid of fast impulsive Lambert solves
(Keplerian boundary ephemerides), filters out infeasible candidates
(no zero-revolution solution, or a transfer conic whose perigee dips
below the Earth plus a safety margin), optionally polishes the winner
with a Nelder-Mead local search, and finally plans/propagates the chosen
transfer with :func:`transfer_ssapy` under the full force model --
including single-burn intercept geometries via ``arrival_burn=False``.

Set ``visualize=True`` for mission-designer curves (porkchop contour and
delta-v vs time-of-flight Pareto front) saved via
``ssapy_toolkit.plots.yufig``.
"""

import warnings

import numpy as np

from ssapy.orbit import Orbit
from ssapy.propagator import KeplerianPropagator
from ssapy.compute import rv
from ssapy.constants import EARTH_MU, EARTH_RADIUS

from ssapy_toolkit.orbital_mechanics.transfer_ssapy_function import (
    transfer_ssapy, solve_lambert)

try:  # astropy is an SSAPy dependency, used only for time conversion
    from astropy.time import Time
except ImportError:  # pragma: no cover
    Time = None


def _to_gps_seconds(t):
    if Time is not None and isinstance(t, Time):
        return float(t.gps)
    return float(t)


def _as_orbit(s, mu):
    if isinstance(s, Orbit):
        return Orbit(np.asarray(s.r, float).ravel(),
                     np.asarray(s.v, float).ravel(),
                     _to_gps_seconds(s.t), mu=mu)
    r, v, t = s
    return Orbit(np.asarray(r, float).ravel(),
                 np.asarray(v, float).ravel(), _to_gps_seconds(t), mu=mu)


def _period(orbit, mu):
    if orbit.a <= 0:
        raise ValueError(
            "transfer_optimal's default search windows require closed "
            "(elliptical) boundary orbits; supply explicit t_window and "
            "tof_range for hyperbolic states.")
    return 2 * np.pi * np.sqrt(orbit.a ** 3 / mu)


def _ephemeris(orbit, times):
    """Keplerian states of ``orbit`` at ``times`` (input order kept)."""
    times = np.asarray(times, dtype=float)
    order = np.argsort(times, kind="stable")
    rr, vv = rv(orbit, times[order], propagator=KeplerianPropagator())
    rr, vv = np.atleast_2d(rr), np.atleast_2d(vv)
    out_r = np.empty_like(rr)
    out_v = np.empty_like(vv)
    out_r[order] = rr
    out_v[order] = vv
    return out_r, out_v


def _conic_perigee(r, v, mu):
    """Perigee radius of the conic through state (r, v)."""
    h = np.linalg.norm(np.cross(r, v))
    energy = 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)
    e = np.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / mu ** 2))
    return h * h / (mu * (1.0 + e))


class OptimalTransferResult:
    """Output of :func:`transfer_optimal`.

    Attributes
    ----------
    transfer : TransferResult
        The fully propagated/refined plan from :func:`transfer_ssapy`
        for the chosen geometry (burn NTW components, trajectory, etc.).
    t_depart, t_arrive, tof : float
        Chosen epochs [GPS s] and time of flight [s].
    dv_total : float
        Objective delta-v of the chosen transfer [m/s] (first burn only
        when ``arrival_burn=False``).
    prograde : bool
        Motion sense of the chosen Lambert geometry.
    arrival_phase : float or None
        Insertion mode only: chosen arrival point, as time-since-epoch
        along orbit 2 [s].
    objective, rendezvous, arrival_burn :
        The search configuration.
    perigee_altitude : float
        Transfer-conic perigee altitude above the Earth's surface [m].
    grid : dict
        The porkchop search grid: ``t_dep`` and ``tof`` axes [s], the
        ``cost`` array [m/s] (minimized over phase/sense; NaN where
        infeasible), and the feasible fraction.
    pareto : dict
        ``{'tof', 'dv'}``: best feasible delta-v per time of flight.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def summary(self):
        mode = ("rendezvous" if self.rendezvous else "insertion")
        burns = "both burns" if self.arrival_burn else "first burn only"
        lines = [
            f"Objective: {self.objective} ({mode}, {burns}, "
            f"{'prograde' if self.prograde else 'retrograde'})",
            f"Departure: t = {self.t_depart:.1f} GPS s "
            f"({(self.t_depart - self.grid['t_dep'][0]) / 3600:.2f} h "
            f"into window)",
            f"Time of flight: {self.tof / 3600:.2f} h "
            f"(arrive t = {self.t_arrive:.1f} GPS s)",
            f"Objective delta-v: {self.dv_total:.2f} m/s",
            f"Transfer-conic perigee altitude: "
            f"{self.perigee_altitude / 1e3:.0f} km",
            f"Search grid: {self.grid['feasible_fraction'] * 100:.0f}% "
            f"of candidates feasible",
        ]
        if self.arrival_phase is not None:
            lines.append(f"Arrival phase along orbit 2: "
                         f"{self.arrival_phase:.1f} s past its epoch")
        lines.append("--- chosen transfer (transfer_ssapy) ---")
        lines.append(self.transfer.summary())
        return "\n".join(lines)


def transfer_optimal(
    orbit1,
    orbit2,
    objective="min_dv",
    rendezvous=True,
    arrival_burn=True,
    t_window=None,
    tof_range=None,
    n_grid=(32, 32),
    n_phase=24,
    dv_budget=None,
    perigee_margin=100e3,
    polish=True,
    visualize=False,
    fig_prefix="demo_gallery/figures/transfer_optimal",
    accel=None,
    propagator=None,
    burn_duration=10.0,
    burn_accel=None,
    thrust=None,
    mass=None,
    isp=None,
    **transfer_kwargs,
):
    """Find the optimal two-burn (or intercept) transfer between orbits.

    Parameters
    ----------
    orbit1, orbit2 : ssapy.orbit.Orbit or (r, v, t) tuple
        Departure and target orbits.  Epochs may be GPS seconds or
        ``astropy.time.Time``.
    objective : {"min_dv", "min_time"}
        ``min_dv`` (default) minimizes the objective delta-v within the
        allowed windows.  ``min_time`` minimizes time of flight among
        candidates whose delta-v fits ``dv_budget`` (required).
    rendezvous : bool
        If True (default), the arrival state is wherever the *object on
        orbit 2* is at ``t_depart + tof`` (its phase is set by its
        epoch).  If False (insertion), the arrival point anywhere along
        orbit 2 is a free search variable -- generally cheaper.
    arrival_burn : bool
        If True (default), the second burn matching the arrival velocity
        is performed and counted.  If False, optimize the *first burn
        only* (intercept/flyby): the spacecraft coasts through the
        target point without matching its velocity.
    t_window : (float, float), optional
        Allowed departure epoch span [GPS s].  Default: one revolution
        of orbit 1 from its epoch.
    tof_range : (float, float), optional
        Allowed time-of-flight span [s].  Default: 2% to 150% of the
        larger orbital period.
    n_grid : (int, int)
        Porkchop grid resolution (departure x time-of-flight).
    n_phase : int
        Arrival-phase samples along orbit 2 (insertion mode only).
    dv_budget : float, optional
        Delta-v constraint [m/s]; required for ``objective='min_time'``,
        recorded/warned for ``min_dv`` (via transfer_ssapy).
    perigee_margin : float
        Candidates whose transfer conic dips below
        ``EARTH_RADIUS + perigee_margin`` are rejected [m].
    polish : bool
        Refine the best grid cell with a Nelder-Mead local search over
        the continuous variables.
    visualize : bool
        Save mission-designer curves (porkchop + delta-v/TOF Pareto
        front) via ``ssapy_toolkit.plots.yufig`` under ``fig_prefix``.
    fig_prefix : str
        yufig path prefix for the visualization.
    burn_accel : float, optional
        Burn acceleration magnitude [m/s^2]: the simple alternative to
        ``thrust``/``mass`` (mutually exclusive with them) when the
        thrust-to-mass analysis was done elsewhere.  Sizes burns and
        drives the same feasibility filtering; no propellant estimates.
    thrust, mass, isp : float, optional
        Engine model, passed through to :func:`transfer_ssapy` (thrust
        [N] and mass [kg] together size each burn's duration; isp [s]
        adds propellant estimates).  When given, the porkchop search
        also rejects candidates whose hardware-sized burns would not
        fit inside the time of flight -- so ``min_time`` answers are
        engine-honest, not just budget-honest.
    accel, propagator, burn_duration, **transfer_kwargs
        Passed through to :func:`transfer_ssapy` for the final
        propagated plan (the search itself uses impulsive Keplerian
        Lambert costs; the finishing differential correction absorbs
        finite-burn and force-model differences).

    Returns
    -------
    OptimalTransferResult

    Notes
    -----
    * The search is zero-revolution Lambert per leg; long windows still
      explore multi-revolution *phasing* implicitly through the
      departure-time axis, but each transfer arc itself spans < 1 rev.
    * Boundary ephemerides during the search are Keplerian even when a
      perturbed ``accel`` is supplied; over windows of a few days the
      resulting epoch error is absorbed by the final refinement, but for
      strongly perturbed, multi-week windows treat the porkchop as
      approximate.
    * Both motion senses are searched automatically when the two orbits
      counter-rotate; co-rotating geometries search prograde only.
    """
    mu = EARTH_MU
    o1 = _as_orbit(orbit1, mu)
    o2 = _as_orbit(orbit2, mu)
    p1, p2 = _period(o1, mu), _period(o2, mu)

    if objective not in ("min_dv", "min_time"):
        raise ValueError("objective must be 'min_dv' or 'min_time'")
    if objective == "min_time" and dv_budget is None:
        raise ValueError("objective='min_time' requires dv_budget")
    if (thrust is None) != (mass is None):
        raise ValueError("thrust and mass must be supplied together.")
    if burn_accel is not None and thrust is not None:
        raise ValueError(
            "Specify either burn_accel or thrust+mass, not both.")
    a_burn = (thrust / mass) if thrust is not None else burn_accel

    t0 = float(o1.t)
    if t_window is None:
        t_window = (t0, t0 + p1)
    t_window = (_to_gps_seconds(t_window[0]), _to_gps_seconds(t_window[1]))
    if tof_range is None:
        tof_range = (0.02 * max(p1, p2), 1.5 * max(p1, p2))

    n_dep, n_tof = n_grid
    t_deps = np.linspace(*t_window, n_dep)
    tofs = np.linspace(*tof_range, n_tof)

    # Both senses only if the orbits counter-rotate.
    h1 = np.cross(np.ravel(o1.r), np.ravel(o1.v))
    h2 = np.cross(np.ravel(o2.r), np.ravel(o2.v))
    senses = (True,) if np.dot(h1, h2) >= 0 else (True, False)

    # --- boundary ephemerides (vectorized Keplerian) -------------------
    dep_r, dep_v = _ephemeris(o1, t_deps)
    if rendezvous:
        t_arr_grid = t_deps[:, None] + tofs[None, :]
        arr_r_flat, arr_v_flat = _ephemeris(o2, t_arr_grid.ravel())
        arr_r = arr_r_flat.reshape(n_dep, n_tof, 3)
        arr_v = arr_v_flat.reshape(n_dep, n_tof, 3)
        phases = None
    else:
        phases = np.linspace(0.0, p2, n_phase, endpoint=False)
        ring_r, ring_v = _ephemeris(o2, float(o2.t) + phases)

    r_min = EARTH_RADIUS + perigee_margin

    def candidate_cost(r1, v1, r2, v2, tof):
        """(cost, prograde) for the cheapest feasible sense, else NaN."""
        best, best_sense = (np.nan, np.nan, np.nan), True
        for sense in senses:
            try:
                v1l, v2l = solve_lambert(r1, r2, tof, mu=mu,
                                         prograde=sense, max_iter=60,
                                         tol=1e-6)
            except RuntimeError:
                continue
            if _conic_perigee(r1, v1l, mu) < r_min:
                continue
            dv1 = np.linalg.norm(v1l - v1)
            dv2 = np.linalg.norm(v2 - v2l) if arrival_burn else 0.0
            # Burn-fit filter with headroom: transfer_ssapy enforces
            # burns <= a third of the TOF on the *refined* delta-v,
            # which exceeds this impulsive estimate by the finite-burn
            # steering losses; 25% here reserves that growth.
            if a_burn is not None and (dv1 + dv2) / a_burn >= 0.25 * tof:
                continue                 # burns don't fit this window
            c = dv1 + (dv2 if arrival_burn else 0.0)
            if not (c >= best[0]):       # also catches best == NaN
                best, best_sense = (c, dv1, dv2), sense
        return best, best_sense

    # --- porkchop grid ---------------------------------------------------
    if rendezvous:
        cost = np.full((n_dep, n_tof), np.nan)
        dv1g = np.full((n_dep, n_tof), np.nan)
        dv2g = np.full((n_dep, n_tof), np.nan)
        sense_grid = np.ones((n_dep, n_tof), dtype=bool)
        for i in range(n_dep):
            for j in range(n_tof):
                (cost[i, j], dv1g[i, j], dv2g[i, j]), sense_grid[i, j] = \
                    candidate_cost(dep_r[i], dep_v[i],
                                   arr_r[i, j], arr_v[i, j], tofs[j])
        cost3 = cost[:, :, None]
        dv1g3 = dv1g[:, :, None]
        dv2g3 = dv2g[:, :, None]
    else:
        cost3 = np.full((n_dep, n_tof, n_phase), np.nan)
        dv1g3 = np.full((n_dep, n_tof, n_phase), np.nan)
        dv2g3 = np.full((n_dep, n_tof, n_phase), np.nan)
        sense3 = np.ones((n_dep, n_tof, n_phase), dtype=bool)
        for i in range(n_dep):
            for j in range(n_tof):
                for k in range(n_phase):
                    (cost3[i, j, k], dv1g3[i, j, k], dv2g3[i, j, k]), \
                        sense3[i, j, k] = candidate_cost(
                            dep_r[i], dep_v[i], ring_r[k], ring_v[k],
                            tofs[j])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cost = (np.nanmin(cost3, axis=2)
                    if not np.all(np.isnan(cost3))
                    else np.full((n_dep, n_tof), np.nan))

    feasible_fraction = float(np.mean(np.isfinite(cost3)))
    if not np.any(np.isfinite(cost3)):
        raise RuntimeError(
            "No feasible transfer found on the search grid: every "
            "candidate either lacked a zero-revolution Lambert solution, "
            "dipped its transfer conic below the perigee margin"
            + (", or could not fit the hardware-sized burns (accel "
               f"{a_burn:.4f} m/s^2) into a third of the time of flight"
               if a_burn is not None else "")
            + ". Widen t_window/tof_range, reduce perigee_margin"
            + (", or use a stronger engine / lighter spacecraft"
               if a_burn is not None else "") + ".")

    # For min_time, the budget constrains the impulsive search estimate,
    # but the hardware-sized finite burns add steering losses on top.
    # Retry with a shrunken effective budget until the *refined* plan
    # actually fits (the porkchop grid is reused, so retries are cheap).
    budget_eff = dv_budget
    for _attempt in range(3):
        # --- objective selection on the grid --------------------------------
        if objective == "min_dv":
            flat = np.nanargmin(cost3)
        else:
            ok = np.where(np.isfinite(cost3) & (cost3 <= budget_eff))
            if len(ok[0]) == 0:
                raise ValueError(
                    f"No transfer on the grid fits the {budget_eff:.1f} m/s "
                    f"effective budget (cheapest impulsive candidate: "
                    f"{np.nanmin(cost3):.1f} m/s; requested budget "
                    f"{dv_budget:.1f} m/s with finite-burn losses "
                    "reserved). Increase dv_budget or widen the "
                    "windows.")
            jmin = np.argmin(ok[1])          # smallest tof index
            flat = np.ravel_multi_index(tuple(idx[jmin] for idx in ok),
                                        cost3.shape)
        idx = np.unravel_index(flat, cost3.shape)
        i0, j0 = idx[0], idx[1]
        k0 = idx[2] if not rendezvous else None

        # --- continuous polish (Nelder-Mead) ---------------------------------
        def eval_point(t_dep, tof, phase=None):
            (r1,), (v1,) = _ephemeris(o1, [t_dep])
            if rendezvous:
                (r2,), (v2,) = _ephemeris(o2, [t_dep + tof])
            else:
                (r2,), (v2,) = _ephemeris(o2, [float(o2.t) + phase])
            (c, _, _), sense = candidate_cost(r1, v1, r2, v2, tof)
            return c, sense, (r1, v1, r2, v2)

        x_best = [t_deps[i0], tofs[j0]] + ([] if rendezvous else [phases[k0]])
        if polish:
            from scipy.optimize import minimize

            lo = [t_window[0], tof_range[0]] + ([] if rendezvous else [-np.inf])
            hi = [t_window[1], tof_range[1]] + ([] if rendezvous else [np.inf])

            def penalty(x):
                x = np.clip(x, lo, hi)
                c, _, _ = eval_point(*x)
                if not np.isfinite(c):
                    return 1e12
                if objective == "min_dv":
                    return c
                return x[1] + (0.0 if c <= budget_eff else 1e9 + c)

            res = minimize(penalty, x_best, method="Nelder-Mead",
                           options=dict(maxfev=200, xatol=1.0, fatol=1e-3))
            if np.isfinite(res.fun) and res.fun < 1e9:
                x_best = list(np.clip(res.x, lo, hi))

        t_dep, tof = float(x_best[0]), float(x_best[1])
        phase = float(x_best[2]) % p2 if not rendezvous else None
        c, sense, (r1, v1, r2, v2) = eval_point(*x_best)

        # --- final propagated, refined plan under the full force model ------
        transfer = transfer_ssapy(
            (r1, v1, t_dep), (r2, v2, t_dep + tof),
            accel=accel, propagator=propagator, burn_duration=burn_duration,
            burn_accel=burn_accel, thrust=thrust, mass=mass, isp=isp,
            prograde=sense, arrival_burn=arrival_burn,
            dv_budget=dv_budget if objective == "min_dv" else None,
            **transfer_kwargs)


        if objective == "min_dv" or transfer.dv_total <= dv_budget:
            break
        budget_eff = budget_eff * dv_budget / transfer.dv_total
    if (objective == "min_time" and dv_budget is not None
            and transfer.dv_total > dv_budget):
        warnings.warn(
            f"min_time plan requires {transfer.dv_total:.1f} m/s, "
            f"exceeding the {dv_budget:.1f} m/s budget even after "
            "reserving finite-burn losses; treat this budget as "
            "infeasible for this geometry/engine.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c_flat = np.moveaxis(cost3, 1, 0).reshape(n_tof, -1)
        d1_flat = np.moveaxis(dv1g3, 1, 0).reshape(n_tof, -1)
        d2_flat = np.moveaxis(dv2g3, 1, 0).reshape(n_tof, -1)
        pareto_dv = np.full(n_tof, np.nan)
        pareto_dv1 = np.full(n_tof, np.nan)
        pareto_dv2 = np.full(n_tof, np.nan)
        for _j in range(n_tof):
            if np.any(np.isfinite(c_flat[_j])):
                _k = np.nanargmin(c_flat[_j])
                pareto_dv[_j] = c_flat[_j, _k]
                pareto_dv1[_j] = d1_flat[_j, _k]
                pareto_dv2[_j] = d2_flat[_j, _k]
    result = OptimalTransferResult(
        transfer=transfer,
        t_depart=t_dep, t_arrive=t_dep + tof, tof=tof,
        dv_total=transfer.dv_total,
        prograde=bool(sense),
        arrival_phase=phase,
        objective=objective, rendezvous=rendezvous,
        arrival_burn=arrival_burn,
        perigee_altitude=_conic_perigee(
            np.ravel(transfer.transfer_orbit.r),
            np.ravel(transfer.transfer_orbit.v), mu) - EARTH_RADIUS,
        dv_budget=dv_budget,
        grid=dict(t_dep=t_deps, tof=tofs, cost=cost,
                  feasible_fraction=feasible_fraction),
        pareto=dict(tof=tofs, dv=pareto_dv,
                    dv1=pareto_dv1, dv2=pareto_dv2),
    )

    if visualize:
        from ssapy_toolkit.plots.transfer_designer_curves_plot import (
            transfer_designer_curves_plot)
        transfer_designer_curves_plot(
            result, save_path=f"{fig_prefix}_designer_curves.jpg")
    return result
