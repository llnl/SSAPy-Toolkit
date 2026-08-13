"""Structured transfer-problem API demo for SSAPy-Toolkit.

This demo exercises the high-level ``transfer_optimal(problem={...})`` schema
and its section-keyword equivalent. It covers SSAPy Orbit inputs, raw state
vectors, rendezvous/intercept/insertion arrival modes, total/first/final/time
objectives, direct/immediate/timed/best route selection, burn-count and arrival
window constraints, engine constraints, and solver controls. Each run saves one
overview image under ``~/ssatk_figures/demo_gallery/figures`` with exactly three
subplots: all transfer trajectories, all burn events, and the objective/designer
trade space.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from textwrap import fill


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from ssapy import Orbit, rv
from ssapy.propagator import KeplerianPropagator

from ssapy_toolkit.constants import EARTH_MU, EARTH_RADIUS
from ssapy_toolkit.orbital_mechanics.transfer_optimal_function import transfer_optimal
from ssapy_toolkit.plots.figpath import figpath
from ssapy_toolkit.plots.plotutils import figsave


TITLE = "Structured Transfer API"
DESCRIPTION = (
    "Demonstrates transfer_optimal(problem={...}) boundary, objective, "
    "constraint, route, and solver sections with transfer plots for each case."
)
UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "demo_gallery/figures"
COMPARISON_TOF = 3600.0
STAGED_LEG_TOF = COMPARISON_TOF / 2.0
DIRECT_TOF_RANGE = (COMPARISON_TOF, COMPARISON_TOF)
STAGED_LEG_TOF_RANGE = (STAGED_LEG_TOF, STAGED_LEG_TOF)
TITLE_FONTSIZE = 26
SUBTITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 17
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 13
CASE_LABEL_FONTSIZE = 12.5
ANNOTATION_FONTSIZE = 12.5


def _circular_state(radius=7000e3, theta=0.0, inclination=0.0, t=0.0):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_inc = np.cos(inclination)
    sin_inc = np.sin(inclination)
    r = radius * np.array([cos_theta, sin_theta * cos_inc, sin_theta * sin_inc])
    v = np.sqrt(EARTH_MU / radius) * np.array([-sin_theta, cos_theta * cos_inc, cos_theta * sin_inc])
    return r, v, t


def _orbit(state):
    r, v, t = state
    return Orbit(r, v, t=t)


def _solver(*, fast=False, n_grid=None, n_phase=None):
    solver = {
        "n_grid": n_grid or ((1, 2) if fast else (1, 3)),
        "polish": False,
        "refine": False,
        "propagate": False,
        "samples": 40 if fast else 90,
        "burn_duration": 1.0,
    }
    if n_phase is not None:
        solver["n_phase"] = n_phase
    return solver


def _build_cases(fast=False):
    """Build examples that show the supported structured transfer API forms.

    The plotting helpers below are intentionally separated from this function
    so users can read this block from top to bottom as a set of copyable
    ``transfer_optimal(...)`` examples. Each ``call`` dictionary is passed to
    ``transfer_optimal(**case["call"])`` in ``main``.
    """
    # All examples use the same start state and the same final target state so
    # the overview figure compares API options, not different transfers.
    target_radius = 9000e3
    target_inclination = np.deg2rad(8.0)
    target_initial_theta = 0.35
    target_arrival_theta = target_initial_theta + np.sqrt(EARTH_MU / target_radius ** 3) * COMPARISON_TOF
    initial_state = _circular_state(7000e3)
    target_state = _circular_state(
        target_radius,
        theta=target_arrival_theta,
        inclination=target_inclination,
        t=COMPARISON_TOF,
    )

    initial = _orbit(initial_state)
    target = _orbit(target_state)
    r0, v0, _ = initial_state
    rf, vf, _ = target_state

    # Staged routes search over candidate parking/staging orbits. For a quick
    # demo, keep this small; for real design work, expand radii, plane fractions,
    # and phase_count.
    route_candidates = {
        "radii": [8500e3],
        "plane_fractions": [0.0, 0.5, 1.0],
        "phase_count": 32,
    }

    cases = []

    # ------------------------------------------------------------------
    # 01. Full ``problem={...}`` schema with SSAPy Orbit objects.
    #
    # This is the recommended pattern for most user code: organize the call
    # into boundary, objective, constraints, route, and solver sections.
    direct_rendezvous_call = {
        "problem": {
            "boundary": {
                "initial": initial,
                "target": target,
                "departure_mode": "leave now",
                "arrival_mode": "rendezvous",
            },
            "objective": {
                "minimize": "delta_v",
                "delta_v_mode": "total",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "perigee_altitude_min": 100e3,
                "max_burns": 2,
            },
            "route": "direct",
            "solver": _solver(fast=fast),
        },
    }
    cases.append({
        "slug": "01_direct_rendezvous_orbits",
        "title": "Orbit objects: direct rendezvous, total delta-v",
        "call": direct_rendezvous_call,
        "designer": True,
    })

    # ------------------------------------------------------------------
    # 02. Intercept mode: match the target position but do not pay an arrival
    # burn to match target velocity. This demonstrates a one-burn objective.
    intercept_first_burn_call = {
        "problem": {
            "boundary": {
                "initial": initial,
                "target": target,
                "departure_mode": "leave now",
                "arrival_mode": "intercept",
            },
            "objective": {
                "minimize": "first_burn",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "max_burns": 1,
            },
            "route": "direct",
            "solver": _solver(fast=fast),
        },
    }
    cases.append({
        "slug": "02_intercept_first_burn",
        "title": "Intercept: first burn only, no arrival match",
        "call": intercept_first_burn_call,
        "designer": True,
    })

    # ------------------------------------------------------------------
    # 03. Insertion mode: minimize the arrival/insertion burn. With n_phase=1
    # and the fixed target epoch below, this still reaches the same target state
    # as the other examples while demonstrating the API knob.
    insertion_arrival_burn_call = {
        "problem": {
            "boundary": {
                "initial": initial,
                "target": target,
                "departure_mode": "leave now",
                "arrival_mode": "insertion",
            },
            "objective": {
                "minimize": "arrival_burn",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "max_burns": 2,
            },
            "route": "direct",
            "solver": _solver(fast=fast, n_phase=1),
        },
    }
    cases.append({
        "slug": "03_insertion_arrival_burn",
        "title": "Insertion: fixed target-state arrival burn",
        "call": insertion_arrival_burn_call,
        "designer": True,
    })

    # ------------------------------------------------------------------
    # 04. Time objective with a delta-v budget. This demo keeps tof_range fixed
    # for apples-to-apples plotting; in real use, widen tof_range to let the
    # solver choose the fastest feasible arrival.
    min_time_budget_call = {
        "problem": {
            "boundary": {
                "initial": initial,
                "target": target,
                "departure_mode": "leave now",
                "arrival_mode": "rendezvous",
            },
            "objective": {
                "minimize": "time",
                "delta_v_mode": "total",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "dv_budget": 10000.0,
                "max_burns": 2,
            },
            "route": "direct",
            "solver": _solver(fast=fast),
        },
    }
    cases.append({
        "slug": "04_min_time_budget",
        "title": "Time objective at the fixed comparison arrival",
        "call": min_time_budget_call,
        "designer": True,
    })

    # ------------------------------------------------------------------
    # 05. Raw state-vector input. This is useful when the caller does not have
    # SSAPy Orbit objects; pass r1/v1/r2/v2 directly and supply the final epoch.
    raw_vector_call = {
        "problem": {
            "r1": r0,
            "v1": v0,
            "r2": rf,
            "v2": vf,
            "t2": COMPARISON_TOF,
            "departure_mode": "now",
            "arrival_mode": "rendezvous",
            "objective": {
                "minimize": "delta_v",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "arrival_window": (COMPARISON_TOF, COMPARISON_TOF),
                "max_burns": 2,
            },
            "route": "direct",
            "solver": _solver(fast=fast, n_grid=(1, 1)),
        },
    }
    cases.append({
        "slug": "05_raw_vectors_arrival_window",
        "title": "Raw vectors: fixed arrival window constraint",
        "call": raw_vector_call,
        "designer": True,
    })

    # ------------------------------------------------------------------
    # 06. Section-keyword form. Instead of nesting everything under
    # ``problem={...}``, pass boundary/objective/constraints/route/solver as
    # top-level keyword sections. This is equivalent to the problem schema.
    immediate_staged_call = {
        "boundary": {
            "initial": initial,
            "target": target,
            "departure_mode": "now",
            "arrival_mode": "rendezvous",
        },
        "objective": {
            "minimize": "delta_v",
            "delta_v_mode": "total",
        },
        "constraints": {
            "tof_range": STAGED_LEG_TOF_RANGE,
            "max_burns": 4,
        },
        "route": {
            "mode": "immediate",
            "n_stage_stops": 1,
            "stage_candidates": route_candidates,
        },
        "solver": _solver(fast=fast, n_grid=(2, 2)),
    }
    cases.append({
        "slug": "06_immediate_staged_sections",
        "title": "Section kwargs: immediate staged transfer",
        "call": immediate_staged_call,
        "designer": False,
    })

    # ------------------------------------------------------------------
    # 07. Timed multi-stage route inside the nested problem schema. The route
    # section tells ssatk to insert one staging orbit and optimize stage timing.
    timed_multistage_call = {
        "problem": {
            "boundary": {
                "initial": initial,
                "target": target,
                "departure_mode": "now",
                "arrival_mode": "rendezvous",
            },
            "objective": {
                "minimize": "delta_v",
                "delta_v_mode": "total",
            },
            "constraints": {
                "tof_range": STAGED_LEG_TOF_RANGE,
                "max_burns": 4,
            },
            "route": {
                "mode": "multi_stage",
                "timing": "optimized",
                "n_stage_stops": 1,
                "stage_candidates": route_candidates,
            },
            "solver": _solver(fast=fast, n_grid=(2, 2)),
        },
    }
    cases.append({
        "slug": "07_timed_multistage",
        "title": "Problem schema: timed multi-stage route",
        "call": timed_multistage_call,
        "designer": False,
    })

    # ------------------------------------------------------------------
    # 08. Best-route mode. Here max_burns=2 prevents a four-burn staged route,
    # so the solver records that it chose the direct fallback.
    best_route_call = {
        "problem": {
            "boundary": {
                "initial": initial,
                "target": target,
                "departure_mode": "leave now",
                "arrival_mode": "rendezvous",
            },
            "objective": {
                "minimize": "delta_v",
                "delta_v_mode": "total",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "max_burns": 2,
            },
            "route": {
                "mode": "best",
                "n_stage_stops": 1,
            },
            "solver": _solver(fast=fast, n_grid=(2, 2)),
        },
    }
    cases.append({
        "slug": "08_best_route_burn_limit",
        "title": "Best route: direct fallback under max_burns",
        "call": best_route_call,
        "designer": True,
    })

    # ------------------------------------------------------------------
    # 09. Engine constraints. Supplying thrust/mass/isp lets ssatk size finite
    # burn timing and propellant mass while preserving the same boundary format.
    engine_constraints_call = {
        "problem": {
            "boundary": {
                "initial": initial,
                "target": target,
                "departure_mode": "leave now",
                "arrival_mode": "rendezvous",
            },
            "objective": {
                "minimize": "delta_v",
                "delta_v_mode": "total",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "max_burns": 2,
                "thrust": 10000.0,
                "mass": 1000.0,
                "isp": 300.0,
            },
            "route": "direct",
            "solver": _solver(fast=fast, n_grid=(1, 2)),
        },
    }
    cases.append({
        "slug": "09_engine_constraints",
        "title": "Engine constraints: thrust, mass, and specific impulse",
        "call": engine_constraints_call,
        "designer": True,
    })

    return cases


def _burn_value(burn, key, default=None):
    return burn.get(key, default) if isinstance(burn, dict) else getattr(burn, key, default)


def _transfer_time_bounds(result):
    trajectory = result.get("trajectory")
    if trajectory is not None and trajectory.get("t") is not None:
        times = np.asarray(trajectory["t"], dtype=float)
        return float(times[0]), float(times[-1])
    burns = result.get("burns", [])
    if not burns:
        return 0.0, float(result.get("tof", 0.0))
    starts = [_burn_value(burn, "t_start", _burn_value(burn, "t", 0.0)) for burn in burns]
    ends = [_burn_value(burn, "t_end", start) for burn, start in zip(burns, starts)]
    return float(min(starts)), float(max(ends))


def _burn_duration(burn):
    duration = _burn_value(burn, "duration", None)
    if duration is not None:
        return float(duration)
    start = _burn_value(burn, "t_start", _burn_value(burn, "t", 0.0))
    end = _burn_value(burn, "t_end", start)
    return max(0.0, float(end) - float(start))


def _burn_acceleration(burn):
    acceleration = _burn_value(burn, "acceleration_mag", None)
    if acceleration is not None:
        return float(acceleration)
    duration = _burn_duration(burn)
    delta_v = float(_burn_value(burn, "delta_v_mag", _burn_value(burn, "dv_mag", 0.0)))
    return delta_v / duration if duration > 0.0 else 0.0


def _cumulative_delta_v_curve(result, t0):
    burns = result.get("burns", [])
    if not burns:
        return np.array([0.0]), np.array([0.0])
    t_end = _transfer_time_bounds(result)[1]
    times = [0.0]
    cumulative = [0.0]
    total = 0.0
    for burn in burns:
        start = float(_burn_value(burn, "t_start", _burn_value(burn, "t", t0)))
        end = float(_burn_value(burn, "t_end", start))
        delta_v = float(_burn_value(burn, "delta_v_mag", _burn_value(burn, "dv_mag", 0.0)))
        times.extend([(start - t0) / 3600.0, (end - t0) / 3600.0])
        cumulative.extend([total, total + delta_v])
        total += delta_v
    times.append((t_end - t0) / 3600.0)
    cumulative.append(total)
    return np.asarray(times, dtype=float), np.asarray(cumulative, dtype=float)


def _case_number(case):
    return case["slug"].split("_", 1)[0]


def _wrapped_case_label(case, width=24):
    wrapped = fill(case["title"], width=width)
    return f"{_case_number(case)} {wrapped.replace(chr(10), chr(10) + '   ')}"


def _cleanup_legacy_overview_outputs():
    output_dir = Path(figpath(f"{FIGDIR}/structured_transfer_api_overview.jpg")).parent
    legacy_dir = output_dir / "structured_transfers"
    for pattern in (
        "*_trajectory.jpg",
        "*_designer.jpg",
        "*_burn_profile.jpg",
        "structured_transfer_burns_overview.jpg",
        "structured_transfer_api_triptychs.jpg",
    ):
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)
        for path in legacy_dir.glob(pattern):
            path.unlink(missing_ok=True)
    legacy_overview = legacy_dir / "structured_transfer_api_overview.jpg"
    legacy_overview.unlink(missing_ok=True)
    try:
        legacy_dir.rmdir()
    except OSError:
        pass


def _sample_transfer_leg(leg, samples=90):
    transfer_orbits = leg.get("transfer_orbits") or []
    if not transfer_orbits:
        trajectory = leg.get("trajectory")
        if trajectory is None or trajectory.get("r") is None or trajectory.get("t") is None:
            return None, None
        return np.asarray(trajectory["t"], dtype=float), np.asarray(trajectory["r"], dtype=float)

    t0 = float(leg["initial"]["t"])
    t1 = float(leg["final"]["t"])
    times = np.linspace(t0, t1, max(2, int(samples)))
    positions, _velocities = rv(transfer_orbits[0], times, propagator=KeplerianPropagator())
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    positions[0] = np.asarray(leg["initial"]["r"], dtype=float).reshape(3)
    positions[-1] = np.asarray(leg["final"]["r"], dtype=float).reshape(3)
    return times, positions


def _planned_trajectory_samples(result, samples_per_leg=90):
    stage_legs = result.get("stage_legs") or []
    if stage_legs:
        times = []
        positions = []
        leg_samples = max(16, int(samples_per_leg) // max(1, len(stage_legs)))
        for leg_index, leg in enumerate(stage_legs):
            leg_times, leg_positions = _sample_transfer_leg(leg, leg_samples)
            if leg_times is None:
                continue
            if leg_index and len(leg_times) > 1:
                leg_times = leg_times[1:]
                leg_positions = leg_positions[1:]
            times.append(leg_times)
            positions.append(leg_positions)
        if times:
            return np.concatenate(times), np.vstack(positions)

    return _sample_transfer_leg(result, samples_per_leg)


def _draw_trajectory_overview(ax, results, cases, colors):
    theta = np.linspace(0.0, 2.0 * np.pi, 181)
    earth_radius_km = EARTH_RADIUS / 1e3
    ax.fill(
        earth_radius_km * np.cos(theta),
        earth_radius_km * np.sin(theta),
        color="0.86",
        label="Earth",
        zorder=0,
    )
    max_extent = earth_radius_km
    common_initial_km = None
    common_final_km = None

    for index, case in enumerate(cases):
        result = results[case["slug"]]
        t, r = _planned_trajectory_samples(result, 90)
        if t is None or r is None:
            continue
        color = colors(index % 10)
        r_km = np.asarray(r, dtype=float) / 1e3
        t = np.asarray(t, dtype=float)
        if common_initial_km is None:
            common_initial_km = np.asarray(result["initial"]["r"], dtype=float).reshape(3) / 1e3
            common_final_km = np.asarray(result["final"]["r"], dtype=float).reshape(3) / 1e3
        ax.plot(r_km[:, 0], r_km[:, 1], color=color, lw=2.6, label=_case_number(case))
        max_extent = max(max_extent, float(np.nanmax(np.abs(r_km[:, :2]))))
        for burn in result.get("burns", []):
            burn_time = float(_burn_value(burn, "t_start", _burn_value(burn, "t", t[0])))
            burn_xy = np.array([np.interp(burn_time, t, r_km[:, axis]) for axis in range(2)])
            ax.plot(burn_xy[0], burn_xy[1], marker="*", color="k", ms=10, zorder=4)

    if common_initial_km is not None and common_final_km is not None:
        ax.plot(
            common_initial_km[0], common_initial_km[1], marker="o", ms=13,
            color="#1f77b4", mec="k", mew=1.0, label="common start", zorder=5,
        )
        ax.plot(
            common_final_km[0], common_final_km[1], marker="X", ms=14,
            color="#2ca02c", mec="k", mew=1.0, label="common target", zorder=5,
        )
        max_extent = max(max_extent, float(np.nanmax(np.abs([common_initial_km[:2], common_final_km[:2]]))))

    padding = 1.12 * max_extent
    ax.set_xlim(-padding, padding)
    ax.set_ylim(-padding, padding)
    ax.set_aspect("equal")
    ax.set_xlabel("x [km]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("y [km]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        "Planned Transfer Trajectories\n(stars mark burns; markers show common endpoints)",
        fontsize=SUBTITLE_FONTSIZE,
    )
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(alpha=0.25)
    ax.legend(title="case", fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE, ncol=3, loc="lower left")


def _draw_burn_event_overview(ax, results, cases, colors):
    all_burns = [burn for case in cases for burn in results[case["slug"]].get("burns", [])]
    max_delta_v = max((float(_burn_value(burn, "delta_v_mag", 0.0)) for burn in all_burns), default=1.0)
    max_accel = max((_burn_acceleration(burn) for burn in all_burns), default=1.0)
    y_positions = np.arange(len(cases))[::-1]
    y_labels = []

    for index, (case, y_position) in enumerate(zip(cases, y_positions)):
        result = results[case["slug"]]
        color = colors(index % 10)
        t0, t1 = _transfer_time_bounds(result)
        duration_hours = max((t1 - t0) / 3600.0, 1e-6)
        y_labels.append(_wrapped_case_label(case))
        ax.hlines(y_position, 0.0, duration_hours, color="0.82", lw=4.0, zorder=0)
        for burn_index, burn in enumerate(result.get("burns", []), 1):
            start = float(_burn_value(burn, "t_start", _burn_value(burn, "t", t0)))
            end = float(_burn_value(burn, "t_end", start))
            middle_hours = (0.5 * (start + end) - t0) / 3600.0
            delta_v = float(_burn_value(burn, "delta_v_mag", _burn_value(burn, "dv_mag", 0.0)))
            acceleration = _burn_acceleration(burn)
            marker_size = 80.0 + 360.0 * np.sqrt(delta_v / max_delta_v) if max_delta_v > 0.0 else 120.0
            alpha = 0.35 + 0.55 * (acceleration / max_accel) if max_accel > 0.0 else 0.75
            ax.scatter(
                middle_hours,
                y_position,
                s=marker_size,
                color=color,
                alpha=min(alpha, 0.95),
                edgecolor="k",
                linewidth=0.45,
                zorder=3,
            )
            ax.annotate(
                f"{burn_index}: {delta_v:.0f}",
                (middle_hours, y_position),
                textcoords="offset points",
                xytext=(0, 8 if burn_index % 2 else -13),
                ha="center",
                va="bottom" if burn_index % 2 else "top",
                fontsize=ANNOTATION_FONTSIZE,
                color="0.15",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=CASE_LABEL_FONTSIZE, linespacing=1.15)
    ax.set_xlabel("time since departure [h]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Burn Events\n(size = delta-v; opacity = acceleration)", fontsize=SUBTITLE_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.75, len(cases) - 0.25)


def _draw_objective_overview(ax, results, cases, colors):
    for index, case in enumerate(cases):
        result = results[case["slug"]]
        diagnostics = result.get("diagnostics", {})
        color = colors(index % 10)
        pareto = diagnostics.get("pareto") if case.get("designer") else None
        if pareto is not None:
            tof = np.asarray(pareto.get("tof", []), dtype=float) / 3600.0
            dv = np.asarray(pareto.get("dv", []), dtype=float)
            if tof.size and dv.size:
                ax.plot(tof, dv, color=color, lw=2.0, alpha=0.55)
        selected_tof = float(result.get("tof", 0.0)) / 3600.0
        selected_dv = float(diagnostics.get("objective_delta_v", result.get("delta_v_total", 0.0)))
        marker = "s" if result.get("method") == "transfer_optimal_staged" else "o"
        ax.scatter(selected_tof, selected_dv, color=color, marker=marker, s=135, edgecolor="k", linewidth=0.8, zorder=4)
        ax.annotate(
            _case_number(case),
            (selected_tof, selected_dv),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=ANNOTATION_FONTSIZE + 1,
            color=color,
            weight="bold",
        )
        dv_budget = diagnostics.get("dv_budget")
        if dv_budget is not None:
            ax.axhline(float(dv_budget), color=color, ls=":", lw=1.4, alpha=0.45)

    ax.set_xlabel("time of flight [h]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("objective delta-v [m/s]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        "Objective / Design Trade\n"
        "Pareto lines = direct cases\n"
        "markers = selected solutions",
        fontsize=SUBTITLE_FONTSIZE,
    )
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(alpha=0.3)


def _save_three_panel_overview(results, cases, *, make_figures=True):
    if not make_figures:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _cleanup_legacy_overview_outputs()

    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(34, 14.5),
        gridspec_kw={"width_ratios": [1.1, 1.9, 1.1]},
    )
    _draw_trajectory_overview(axes[0], results, cases, colors)
    _draw_burn_event_overview(axes[1], results, cases, colors)
    _draw_objective_overview(axes[2], results, cases, colors)
    fig.suptitle("Structured Transfer API Overview", fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=3.0)

    save_rel = f"{FIGDIR}/structured_transfer_api_overview.jpg"
    figsave(fig, save_rel)
    return figpath(save_rel)

def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    cases = _build_cases(fast=fast)
    results = {}
    rows = []
    output_paths = []

    for case in cases:
        result = transfer_optimal(**case["call"])
        diagnostics = result.get("diagnostics", {})
        results[case["slug"]] = result
        rows.append(
            {
                "case": case["slug"],
                "title": case["title"],
                "method": result.get("method"),
                "arrival_mode": diagnostics.get("arrival_mode"),
                "route_mode": diagnostics.get("route_mode"),
                "delta_v_total_m_s": float(result.get("delta_v_total", np.nan)),
                "burn_count": len(result.get("burns", [])),
                "tof_s": float(result.get("tof", np.nan)),
                "problem_schema": diagnostics.get("problem_schema"),
            }
        )

    overview_path = _save_three_panel_overview(results, cases, make_figures=make_figures)
    if overview_path is not None:
        output_paths.append(overview_path)

    print("Structured transfer API cases:")
    for row in rows:
        print(
            f"  {row['case']}: {row['method']} | {row['arrival_mode']} | "
            f"route={row['route_mode']} | burns={row['burn_count']} | "
            f"dv={row['delta_v_total_m_s']:.1f} m/s | tof={row['tof_s']:.1f} s"
        )

    return {
        "title": TITLE,
        "description": DESCRIPTION,
        "results": results,
        "summary": rows,
        "output_paths": output_paths,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
