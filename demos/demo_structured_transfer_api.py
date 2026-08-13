"""Structured transfer-problem API demo for SSAPy-Toolkit.

This demo exercises the high-level ``transfer_optimal(problem={...})`` schema
and its section-keyword equivalent. It covers SSAPy Orbit inputs, raw state
vectors, rendezvous/intercept/insertion arrival modes, total/first/final/time
objectives, direct/immediate/timed/best route selection, burn-count and arrival
window constraints, engine constraints, and solver controls. Each case saves a
transfer trajectory plot and a burn-timeline plot under
``~/ssatk_figures/demo_gallery/figures/structured_transfers``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from ssapy import Orbit

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.orbital_mechanics.transfer_optimal_function import transfer_optimal
from ssapy_toolkit.plots.figpath import figpath
from ssapy_toolkit.plots.orbit_plot import orbit_plot


TITLE = "Structured Transfer API"
DESCRIPTION = (
    "Demonstrates transfer_optimal(problem={...}) boundary, objective, "
    "constraint, route, and solver sections with transfer plots for each case."
)
UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "demo_gallery/figures/structured_transfers"


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
        "propagate": True,
        "samples": 40 if fast else 90,
        "burn_duration": 1.0,
    }
    if n_phase is not None:
        solver["n_phase"] = n_phase
    return solver


def _direct_problem(initial, target, *, arrival_mode="rendezvous", objective=None,
                    constraints=None, route="direct", solver=None):
    return {
        "boundary": {
            "initial": initial,
            "target": target,
            "departure_mode": "leave now",
            "arrival_mode": arrival_mode,
        },
        "objective": objective or {"minimize": "delta_v", "delta_v_mode": "total"},
        "constraints": constraints or {"tof_range": (900.0, 1800.0), "max_burns": 2},
        "route": route,
        "solver": solver or _solver(),
    }


def _build_cases(fast=False):
    direct_initial_state = _circular_state(7000e3)
    direct_target_state = _circular_state(9000e3, theta=0.35, inclination=np.deg2rad(8.0))
    staged_initial_state = _circular_state(7000e3)
    staged_target_state = _circular_state(9000e3, theta=0.4)

    direct_initial = _orbit(direct_initial_state)
    direct_target = _orbit(direct_target_state)
    staged_initial = _orbit(staged_initial_state)
    staged_target = _orbit(staged_target_state)
    r0, v0, _ = direct_initial_state
    rf, vf, _ = direct_target_state

    route_candidates = {
        "radii": [8000e3],
        "plane_fractions": [0.0],
        "phase_count": 2,
    }

    return [
        {
            "slug": "01_direct_rendezvous_orbits",
            "title": "Orbit objects: direct rendezvous, total delta-v",
            "call": {
                "problem": _direct_problem(
                    direct_initial,
                    direct_target,
                    constraints={
                        "tof_range": (900.0, 1800.0),
                        "perigee_altitude_min": 100e3,
                        "max_burns": 2,
                    },
                    solver=_solver(fast=fast),
                ),
            },
            "designer": True,
        },
        {
            "slug": "02_intercept_first_burn",
            "title": "Intercept: first burn only, no arrival match",
            "call": {
                "problem": _direct_problem(
                    direct_initial,
                    direct_target,
                    arrival_mode="intercept",
                    objective={"minimize": "first_burn"},
                    constraints={"tof_range": (900.0, 1800.0), "max_burns": 1},
                    solver=_solver(fast=fast),
                ),
            },
            "designer": True,
        },
        {
            "slug": "03_insertion_arrival_burn",
            "title": "Insertion: free target phase, minimize final burn",
            "call": {
                "problem": _direct_problem(
                    direct_initial,
                    direct_target,
                    arrival_mode="insertion",
                    objective={"minimize": "arrival_burn"},
                    constraints={"tof_range": (900.0, 1800.0), "max_burns": 2},
                    solver=_solver(fast=fast, n_phase=4),
                ),
            },
            "designer": True,
        },
        {
            "slug": "04_min_time_budget",
            "title": "Minimum time subject to a delta-v budget",
            "call": {
                "problem": _direct_problem(
                    direct_initial,
                    direct_target,
                    objective={"minimize": "time", "delta_v_mode": "total"},
                    constraints={"tof_range": (900.0, 2400.0), "dv_budget": 10000.0, "max_burns": 2},
                    solver=_solver(fast=fast),
                ),
            },
            "designer": True,
        },
        {
            "slug": "05_raw_vectors_arrival_window",
            "title": "Raw vectors: fixed arrival window constraint",
            "call": {
                "problem": {
                    "r1": r0,
                    "v1": v0,
                    "r2": rf,
                    "v2": vf,
                    "departure_mode": "now",
                    "arrival_mode": "rendezvous",
                    "objective": {"minimize": "delta_v"},
                    "constraints": {
                        "tof_range": (1200.0, 1200.0),
                        "arrival_window": (1190.0, 1210.0),
                        "max_burns": 2,
                    },
                    "route": "direct",
                    "solver": _solver(fast=fast, n_grid=(1, 1)),
                },
            },
            "designer": True,
        },
        {
            "slug": "06_immediate_staged_sections",
            "title": "Section kwargs: immediate staged transfer",
            "call": {
                "boundary": {
                    "initial": staged_initial,
                    "target": staged_target,
                    "departure_mode": "now",
                    "arrival_mode": "rendezvous",
                },
                "objective": {"minimize": "delta_v", "delta_v_mode": "total"},
                "constraints": {"tof_range": (1000.0, 6000.0), "max_burns": 4},
                "route": {
                    "mode": "immediate",
                    "n_stage_stops": 1,
                    "stage_candidates": route_candidates,
                },
                "solver": _solver(fast=fast, n_grid=(2, 2)),
            },
            "designer": False,
        },
        {
            "slug": "07_timed_multistage",
            "title": "Problem schema: timed multi-stage route",
            "call": {
                "problem": {
                    "boundary": {
                        "initial": staged_initial,
                        "target": staged_target,
                        "departure_mode": "now",
                        "arrival_mode": "rendezvous",
                    },
                    "objective": {"minimize": "delta_v", "delta_v_mode": "total"},
                    "constraints": {"tof_range": (1000.0, 6000.0), "max_burns": 4},
                    "route": {
                        "mode": "multi_stage",
                        "timing": "optimized",
                        "n_stage_stops": 1,
                        "stage_candidates": route_candidates,
                    },
                    "solver": _solver(fast=fast, n_grid=(2, 2)),
                },
            },
            "designer": False,
        },
        {
            "slug": "08_best_route_burn_limit",
            "title": "Best route: direct fallback under max_burns",
            "call": {
                "problem": _direct_problem(
                    staged_initial,
                    staged_target,
                    constraints={"tof_range": (1000.0, 6000.0), "max_burns": 2},
                    route={"mode": "best", "n_stage_stops": 1},
                    solver=_solver(fast=fast, n_grid=(2, 2)),
                ),
            },
            "designer": True,
        },
        {
            "slug": "09_engine_constraints",
            "title": "Engine constraints: thrust, mass, and specific impulse",
            "call": {
                "problem": _direct_problem(
                    direct_initial,
                    direct_target,
                    constraints={
                        "tof_range": (2400.0, 3600.0),
                        "max_burns": 2,
                        "thrust": 5000.0,
                        "mass": 1000.0,
                        "isp": 300.0,
                    },
                    solver=_solver(fast=fast, n_grid=(1, 2)),
                ),
            },
            "designer": True,
        },
    ]


def _save_transfer_plots(result, case, *, make_figures=True):
    if not make_figures:
        return []

    output_paths = []
    base = f"{FIGDIR}/{case['slug']}"
    title = case["title"]

    trajectory_rel = f"{base}_trajectory.jpg"
    orbit_plot(
        result,
        view="transfer_trajectory",
        title=title,
        save=trajectory_rel,
        annotate_burns=len(result.get("burns", [])) <= 2,
    )
    output_paths.append(figpath(trajectory_rel))

    burns_rel = f"{base}_burn_profile.jpg"
    orbit_plot(
        result,
        view="transfer_burn_profile",
        title=f"{title}: burn timeline",
        save=burns_rel,
    )
    output_paths.append(figpath(burns_rel))

    if case.get("designer"):
        designer_rel = f"{base}_designer.jpg"
        orbit_plot(
            result,
            view="transfer_designer",
            title=f"{title}: search grid",
            save=designer_rel,
        )
        output_paths.append(figpath(designer_rel))

    return output_paths


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
        output_paths.extend(_save_transfer_plots(result, case, make_figures=make_figures))
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
