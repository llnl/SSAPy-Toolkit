"""All-in-one orbital maneuver demo for SSAPy-Toolkit.

This gallery demo consolidates the previous small transfer, burn-conversion,
and rendezvous maneuver demos into one user-facing workflow.  It compares
analytic impulsive transfers, fixed-time Lambert wrappers, optimized searches,
explicit staged optimal transfers, continuous low-thrust burns, and
finite-burn/impulse conversions, then renders summary figures under
``~/ssatk_figures/demo_gallery/figures``.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from ssapy import Orbit
from ssapy.compute import rv

from ssapy_toolkit.constants import EARTH_MU
from ssapy_toolkit.orbital_mechanics.burn_to_deltav import burn_to_deltav
from ssapy_toolkit.orbital_mechanics.deltav_to_burn import deltav_to_burn
from ssapy_toolkit.orbital_mechanics.transfer_bielliptic import transfer_bielliptic
from ssapy_toolkit.orbital_mechanics.transfer_coplanar import transfer_coplanar
from ssapy_toolkit.orbital_mechanics.transfer_hohmann import transfer_hohmann
from ssapy_toolkit.orbital_mechanics.transfer_inclination_continuous import transfer_inclination_continuous
from ssapy_toolkit.orbital_mechanics.transfer_lambertian import transfer_lambertian
from ssapy_toolkit.orbital_mechanics.transfer_optimal_function import transfer_optimal
from ssapy_toolkit.orbital_mechanics.transfer_optimal_function import transfer_rendezvous
from ssapy_toolkit.orbital_mechanics.transfer_shooter import transfer_shooter
from ssapy_toolkit.orbital_mechanics.transfer_ssapy_function import transfer_ssapy
from ssapy_toolkit.orbital_mechanics.transfer_velocity_and_inclination_continuous import (
    transfer_velocity_and_inclination_continuous,
)
from ssapy_toolkit.orbital_mechanics.transfer_velocity_continuous import transfer_velocity_continuous
from ssapy_toolkit.plots.plotutils import figsave


UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "demo_gallery/figures"
INITIAL_ORBIT_COLOR = "#1f77b4"
FINAL_ORBIT_COLOR = "#2ca02c"
MANEUVER_COLOR = "#d62728"
IMPULSE_COLOR = "#ff7f0e"


def _circular_state(radius=7000e3, theta=0.0, inclination=0.0, t=0.0):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_inc = np.cos(inclination)
    sin_inc = np.sin(inclination)
    r = radius * np.array([cos_theta, sin_theta * cos_inc, sin_theta * sin_inc])
    v = np.sqrt(EARTH_MU / radius) * np.array([-sin_theta, cos_theta * cos_inc, cos_theta * sin_inc])
    return r, v, t


def _trajectory(result, samples=300):
    trajectory = result.get("trajectory") if isinstance(result, dict) else None
    if trajectory and trajectory.get("r") is not None:
        return np.asarray(trajectory["r"], dtype=float)
    path = _coast_path_from_first_burn(result, samples=samples)
    if path is not None:
        return path
    if isinstance(result, dict) and "initial" in result and "target" in result:
        return np.vstack((result["initial"]["r"], result["target"]["r"]))
    return None


def _coast_path_from_first_burn(result, samples=300):
    if not isinstance(result, dict) or not result.get("burns") or result.get("tof") is None:
        return None
    initial = result.get("initial", {})
    r0 = np.asarray(initial.get("r"), dtype=float)
    v0 = np.asarray(initial.get("v"), dtype=float) + np.asarray(result["burns"][0]["delta_v"], dtype=float)
    t0 = float(initial.get("t", 0.0))
    times = np.linspace(t0, t0 + float(result["tof"]), samples)
    try:
        r_path, _ = rv(Orbit(r0, v0, t=t0), times)
    except Exception:
        return None
    return np.asarray(r_path, dtype=float)


def _orbit_period(r, v):
    radius = np.linalg.norm(r)
    speed_squared = float(np.dot(v, v))
    specific_energy = 0.5 * speed_squared - EARTH_MU / radius
    if specific_energy >= 0.0:
        return 2.0 * np.pi * np.sqrt(radius**3 / EARTH_MU)
    semimajor_axis = -EARTH_MU / (2.0 * specific_energy)
    return 2.0 * np.pi * np.sqrt(semimajor_axis**3 / EARTH_MU)


def _orbit_path_from_state(state, samples=360):
    r0 = np.asarray(state["r"], dtype=float)
    v0 = np.asarray(state["v"], dtype=float)
    t0 = float(state.get("t", 0.0))
    period = _orbit_period(r0, v0)
    times = np.linspace(t0, t0 + period, samples)
    r_path, _ = rv(Orbit(r0, v0, t=t0), times)
    return np.asarray(r_path, dtype=float)


def _project(points, view="xy"):
    points = np.asarray(points, dtype=float)
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[view]
    labels = {"xy": ("x [km]", "y [km]"), "xz": ("x [km]", "z [km]"), "yz": ("y [km]", "z [km]")}[view]
    return points[:, axes[0]] / 1e3, points[:, axes[1]] / 1e3, labels


def _plot_projected(ax, points, *, view, color, label, linestyle="-", linewidth=2.0, alpha=1.0, zorder=2):
    x, y, _ = _project(points, view=view)
    ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _scatter_projected(ax, points, *, view, color, marker, label, size=42, zorder=5):
    x, y, _ = _project(np.asarray(points, dtype=float).reshape(-1, 3), view=view)
    ax.scatter(x, y, color=color, marker=marker, s=size, label=label, edgecolor="black", linewidth=0.4, zorder=zorder)


def _style_axis(ax, title, view="xy"):
    _, _, labels = _project(np.zeros((1, 3)), view=view)
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.axis("equal")
    ax.grid(alpha=0.25)


def _format_dv(result):
    return f"Δv={result['delta_v_total']:.0f} m/s"


def _format_tof(result):
    tof = result.get("tof")
    if tof is None:
        return ""
    if tof < 3600.0:
        return f"TOF={tof / 60.0:.1f} min"
    return f"TOF={tof / 3600.0:.1f} h"


def _maneuver_title(name, result):
    parts = [_format_dv(result)]
    tof = _format_tof(result)
    if tof:
        parts.append(tof)
    return f"{name}\n" + ", ".join(parts)


def _burn_label(index, count):
    if count == 1:
        return r"$\Delta v_0$"
    if count == 2:
        return (r"$\Delta v_0$", r"$\Delta v_f$")[index]
    if count == 3:
        return (r"$\Delta v_0$", r"$\Delta v_m$", r"$\Delta v_f$")[index]
    if index == 0:
        return r"$\Delta v_0$"
    if index == count - 1:
        return r"$\Delta v_f$"
    return rf"$\Delta v_{{s{index}}}$"


def _annotate_burns(ax, result, view="xy"):
    burns = result.get("burns") or []
    if not burns:
        return
    offsets = [(16, 14), (18, -22), (-58, 18), (-58, -26), (10, 34), (-66, 36), (22, -42), (-72, -44)]
    for index, burn in enumerate(burns):
        state = burn.get("state") or {}
        if state.get("r") is None:
            continue
        label = _burn_label(index, len(burns))
        text = f"{label}\n{burn['delta_v_mag']:.0f} m/s"
        x, y, _ = _project(np.asarray(state["r"], dtype=float).reshape(1, 3), view=view)
        ax.scatter(x[0], y[0], color=IMPULSE_COLOR, marker="*", s=130, edgecolor="black", linewidth=0.6, zorder=6)
        ax.annotate(
            text,
            xy=(x[0], y[0]),
            xytext=offsets[index % len(offsets)],
            textcoords="offset points",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": IMPULSE_COLOR, "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": IMPULSE_COLOR, "lw": 0.8},
            zorder=7,
        )


def _plot_transfer_panel(ax, result, title, *, view="xy", trajectory=None, extra_paths=None):
    initial = result["initial"]
    final = result.get("final") or result.get("target")
    start_orbit = _orbit_path_from_state(initial)
    final_orbit = _orbit_path_from_state(final)
    transfer_path = _trajectory(result) if trajectory is None else np.asarray(trajectory, dtype=float)

    _plot_projected(ax, start_orbit, view=view, color=INITIAL_ORBIT_COLOR, label="initial orbit", linewidth=1.7, alpha=0.85, zorder=1)
    _plot_projected(ax, final_orbit, view=view, color=FINAL_ORBIT_COLOR, label="final orbit", linewidth=1.7, alpha=0.85, zorder=1)
    if transfer_path is not None:
        _plot_projected(ax, transfer_path, view=view, color=MANEUVER_COLOR, label="maneuver trajectory", linewidth=2.8, zorder=3)
    for path_label, path_points, path_color, path_style in extra_paths or []:
        _plot_projected(ax, path_points, view=view, color=path_color, label=path_label, linestyle=path_style, linewidth=2.0, alpha=0.95, zorder=3)
    _scatter_projected(ax, initial["r"], view=view, color=INITIAL_ORBIT_COLOR, marker="o", label="start", size=44)
    _scatter_projected(ax, final["r"], view=view, color=FINAL_ORBIT_COLOR, marker="s", label="target/final", size=44)
    _annotate_burns(ax, result, view=view)
    _style_axis(ax, title, view=view)
    ax.legend(fontsize=7, loc="best")


def _stage_waits(result):
    legs = result.get("stage_legs") or []
    waits = []
    for first_leg, next_leg in zip(legs[:-1], legs[1:]):
        first_arrive = first_leg.get("diagnostics", {}).get("t_arrive", first_leg.get("final", {}).get("t", 0.0))
        next_depart = next_leg.get("diagnostics", {}).get("t_depart", next_leg.get("initial", {}).get("t", first_arrive))
        waits.append(max(0.0, float(next_depart) - float(first_arrive)))
    return waits


def _format_waits(result):
    waits = _stage_waits(result)
    if not waits:
        return "waits=none"
    return "waits=" + "/".join(f"{wait / 60.0:.0f} min" for wait in waits)


def _staged_title(name, result):
    parts = [_format_dv(result)]
    tof = _format_tof(result)
    if tof:
        parts.append(tof)
    parts.append(_format_waits(result))
    return f"{name}\n" + ", ".join(parts)


def _plot_staged_panel(ax, result, title, *, view="xy"):
    legs = result.get("stage_legs") or []
    if not legs:
        _plot_transfer_panel(ax, result, title, view=view)
        return

    initial = result["initial"]
    final = result.get("final") or result.get("target")
    _plot_projected(ax, _orbit_path_from_state(initial), view=view, color=INITIAL_ORBIT_COLOR, label="initial orbit", linewidth=1.7, alpha=0.85, zorder=1)
    _plot_projected(ax, _orbit_path_from_state(final), view=view, color=FINAL_ORBIT_COLOR, label="target orbit", linewidth=1.7, alpha=0.85, zorder=1)

    leg_colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    for index, leg in enumerate(legs):
        path = _trajectory(leg)
        if path is not None:
            _plot_projected(
                ax,
                path,
                view=view,
                color=leg_colors[index % len(leg_colors)],
                label=f"transfer leg {index + 1}",
                linewidth=2.6,
                alpha=0.95,
                zorder=3,
            )
        if index < len(legs) - 1:
            stage_state = leg.get("final") or leg.get("target")
            if stage_state and stage_state.get("r") is not None and stage_state.get("v") is not None:
                _plot_projected(
                    ax,
                    _orbit_path_from_state(stage_state),
                    view=view,
                    color=leg_colors[index % len(leg_colors)],
                    label=f"staging orbit {index + 1}",
                    linestyle="--",
                    linewidth=1.3,
                    alpha=0.55,
                    zorder=1,
                )
                _scatter_projected(ax, stage_state["r"], view=view, color=leg_colors[index % len(leg_colors)], marker="D", label=f"stage {index + 1}", size=38)

    _scatter_projected(ax, initial["r"], view=view, color=INITIAL_ORBIT_COLOR, marker="o", label="start", size=44)
    _scatter_projected(ax, final["r"], view=view, color=FINAL_ORBIT_COLOR, marker="s", label="target/final", size=44)
    _annotate_burns(ax, result, view=view)
    _style_axis(ax, title, view=view)
    ax.legend(fontsize=6.5, loc="best")


def _burn_event_time(result, burn, index):
    state = burn.get("state") or {}
    if state.get("t") is not None:
        return float(state["t"])
    if index == 0:
        return float(result.get("diagnostics", {}).get("t_depart", result.get("initial", {}).get("t", 0.0)))
    if index == len(result.get("burns", [])) - 1:
        return float(result.get("diagnostics", {}).get("t_arrive", result.get("final", {}).get("t", 0.0)))
    return float(result.get("initial", {}).get("t", 0.0))


def _burn_conversion_panel_result(result):
    r_cont = np.asarray(result["r_continuous"], dtype=float)
    r_inst = np.asarray(result["r_instantaneous"], dtype=float)
    v_cont = np.asarray(result["v_continuous"], dtype=float)
    final_state = {
        "r": r_cont[-1],
        "v": v_cont[-1],
        "t": float(r_cont.shape[0] - 1),
    }
    return r_cont, r_inst, final_state


def _summarize_delta_v(results_by_family):
    rows = []
    for family, entries in results_by_family.items():
        for name, result in entries.items():
            rows.append(
                {
                    "family": family,
                    "case": name,
                    "delta_v_total": float(result["delta_v_total"]),
                    "burn_count": len(result.get("burns", [])),
                    "tof": result.get("tof"),
                    "method": result.get("method"),
                }
            )
    return rows


def _build_maneuver_results(fast):
    samples = 32 if fast else 180
    r0, v0, _ = _circular_state(7000e3)
    transfer_angle = np.deg2rad(150.0)
    r_target, v_target, _ = _circular_state(9000e3, transfer_angle)
    tof = (150.0 / 180.0) * np.pi * np.sqrt(((7000e3 + 9000e3) / 2.0) ** 3 / EARTH_MU)

    impulsive = {
        "Hohmann raise": transfer_hohmann(7000e3, 9000e3, samples=samples),
        "Hohmann lower": transfer_hohmann(9000e3, 7000e3, samples=samples),
        "Bi-elliptic raise": transfer_bielliptic(7000e3, 9000e3, intermediate_radius=20_000e3, samples_per_arc=samples // 2),
        "Bi-elliptic lower": transfer_bielliptic(9000e3, 7000e3, intermediate_radius=20_000e3, samples_per_arc=samples // 2),
    }

    fixed_time_kwargs = {"propagate": False, "refine": False, "burn_duration": 1.0}
    departure = (r0, v0, 0.0)
    arrival = (r_target, v_target, tof)
    fixed_time = {
        "transfer_ssapy": transfer_ssapy(departure, arrival, **fixed_time_kwargs),
        "Lambert wrapper": transfer_lambertian(departure, arrival, **fixed_time_kwargs),
        "Shooter wrapper": transfer_shooter(departure, arrival, **fixed_time_kwargs),
        "Coplanar wrapper": transfer_coplanar(departure, arrival, coplanar_tol=1e-9, **fixed_time_kwargs),
    }

    velocity_delta_v = 10.0 if fast else 300.0
    velocity_lower_delta_v = -5.0 if fast else -180.0
    inclination_delta_v = 5.0 if fast else 300.0
    inclination_lower_delta_v = -2.0 if fast else -180.0
    continuous_accel = 1.0 if fast else 2.0
    continuous_time = 30.0 if fast else 900.0
    continuous = {
        f"Velocity +{velocity_delta_v:.0f} m/s": transfer_velocity_continuous(r0, v0, v_target=velocity_delta_v, a_thrust=continuous_accel, max_time=continuous_time),
        f"Velocity {velocity_lower_delta_v:.0f} m/s": transfer_velocity_continuous(r0, v0, v_target=velocity_lower_delta_v, a_thrust=continuous_accel, max_time=continuous_time),
        f"Inclination +{inclination_delta_v:.0f} m/s": transfer_inclination_continuous(r0, v0, delta_v=inclination_delta_v, a_thrust=continuous_accel, max_time=continuous_time),
        f"Inclination {inclination_lower_delta_v:.0f} m/s": transfer_inclination_continuous(r0, v0, delta_v=inclination_lower_delta_v, a_thrust=continuous_accel, max_time=continuous_time),
        "Velocity then inclination": transfer_velocity_and_inclination_continuous(
            r0,
            v0,
            i_target=np.deg2rad(0.01),
            a_thrust=1.0,
            max_time1=2.0,
            max_time2=200.0,
        ),
    }

    optimized_radius = 9000e3
    optimized_angle = 0.4
    r_short, v_short, _ = _circular_state(optimized_radius, optimized_angle)
    optimal_kwargs = {
        "t_window": (0.0, 1000.0),
        "tof_range": (1000.0, 6000.0),
        "n_grid": (2, 2) if fast else (8, 8),
        "polish": False,
        "propagate": False,
        "refine": False,
        "burn_duration": 1.0,
    }
    optimal = {
        "Optimal total Δv": transfer_optimal((r0, v0, 0.0), (r_short, v_short, 0.0), delta_v_mode="total", **optimal_kwargs),
        "Optimal first burn": transfer_optimal(
            (r0, v0, 0.0),
            (r_short, v_short, 0.0),
            delta_v_mode="first",
            arrival_burn=False,
            **optimal_kwargs,
        ),
        "Optimal last burn": transfer_optimal((r0, v0, 0.0), (r_short, v_short, 0.0), delta_v_mode="last", **optimal_kwargs),
        "Min time under cap": transfer_optimal(
            (r0, v0, 0.0),
            (r_short, v_short, 0.0),
            objective="time",
            dv_budget=5000.0,
            **optimal_kwargs,
        ),
        "Rendezvous wrapper": transfer_rendezvous((r0, v0, 0.0), (r_short, v_short, 0.0), **optimal_kwargs),
    }

    staged_angle = 0.2
    staged_inclination = np.deg2rad(25.0)
    r_staged_target, v_staged_target, _ = _circular_state(12_500e3, staged_angle, staged_inclination)
    staged_kwargs = {
        "departure_mode": "now",
        "tof_range": (1800.0, 9000.0),
        "n_grid": (3, 3) if fast else (5, 5),
        "polish": False,
        "propagate": False,
        "refine": False,
        "burn_duration": 1.0,
    }
    staged_search_kwargs = {
        "stage_radii": [9000e3, 11_500e3] if fast else [8500e3, 10_500e3, 12_500e3],
        "stage_plane_fractions": [0.0, 1.0] if fast else [0.0, 0.5, 1.0],
        "n_stage_phase": 1 if fast else 2,
        "stage_beam_width": 2 if fast else 4,
        "stage_wait_window": 6000.0 if fast else 12_000.0,
    }
    staged_boundary = ((r0, v0, 0.0), (r_staged_target, v_staged_target, 0.0))
    staged_optimal = {
        "Direct leave-now": transfer_optimal(*staged_boundary, **staged_kwargs),
        "Immediate one-stop": transfer_optimal(
            *staged_boundary,
            stage_mode="immediate",
            n_stage_stops=1,
            **staged_kwargs,
            **staged_search_kwargs,
        ),
        "Timed one-stop": transfer_optimal(
            *staged_boundary,
            stage_mode="timed",
            stage_timing="appropriately timed",
            n_stage_stops=1,
            **staged_kwargs,
            **staged_search_kwargs,
        ),
        "Timed two-stop min-time": transfer_optimal(
            *staged_boundary,
            stage_mode="timed",
            stage_timing="appropriately timed",
            n_stage_stops=2,
            objective="time",
            dv_budget=50_000.0,
            **staged_kwargs,
            **staged_search_kwargs,
        ),
    }

    orbit = Orbit(r0, v0, t=0.0)
    burn_times = np.arange(0.0, 30.0 if fast else 600.0, 1.0)
    burn_accel = np.array([0.01, 0.02, 0.002]) if fast else np.array([0.10, 0.50, 0.05])
    impulse_delta_v = np.array([0.2, 0.4, 0.04]) if fast else np.array([60.0, 300.0, 30.0])
    burn_conversion = {
        "burn_to_deltav": burn_to_deltav(orbit, burn_times, burn_accel),
        "deltav_to_burn": deltav_to_burn(orbit, burn_times, impulse_delta_v),
    }

    return {
        "impulsive": impulsive,
        "fixed_time": fixed_time,
        "continuous": continuous,
        "optimal": optimal,
        "staged_optimal": staged_optimal,
        "burn_conversion": burn_conversion,
    }


def _first_key(entries, prefix):
    return next(key for key in entries if key.startswith(prefix))


def _make_summary_figure(results):
    fig, axes = plt.subplots(3, 3, figsize=(19, 17), constrained_layout=True)
    axes = axes.ravel()

    panels = [
        ("Hohmann raise", results["impulsive"]["Hohmann raise"], "xy", None),
        ("Hohmann lower", results["impulsive"]["Hohmann lower"], "xy", None),
        ("Bi-elliptic raise", results["impulsive"]["Bi-elliptic raise"], "xy", None),
        ("Bi-elliptic lower", results["impulsive"]["Bi-elliptic lower"], "xy", None),
        ("Fixed-time Lambert", results["fixed_time"]["transfer_ssapy"], "xy", None),
        ("Optimized transfer", results["optimal"]["Optimal total Δv"], "xy", None),
        (
            "Continuous tangential burn",
            results["continuous"][_first_key(results["continuous"], "Velocity +")],
            "xy",
            None,
        ),
        (
            "Continuous plane-change burn",
            results["continuous"][_first_key(results["continuous"], "Inclination +")],
            "xz",
            None,
        ),
    ]

    for ax, (name, result, view, trajectory) in zip(axes[:8], panels):
        _plot_transfer_panel(ax, result, _maneuver_title(name, result), view=view, trajectory=trajectory)

    burn_result = results["burn_conversion"]["burn_to_deltav"]
    r_cont, r_inst, final_state = _burn_conversion_panel_result(burn_result)
    burn_as_transfer = {
        "initial": {"r": r_cont[0], "v": burn_result["v_continuous"][0], "t": 0.0},
        "final": final_state,
        "target": final_state,
        "trajectory": {"r": r_cont},
        "burns": [
            {
                "state": {"r": r_cont[0], "v": burn_result["v_continuous"][0], "t": 0.0},
                "delta_v_mag": float(np.linalg.norm(burn_result["delta_v_ntw"])),
            }
        ],
        "tof": float(r_cont.shape[0] - 1),
        "delta_v_total": float(np.linalg.norm(burn_result["delta_v_ntw"])),
    }
    _plot_transfer_panel(
        axes[8],
        burn_as_transfer,
        _maneuver_title("Finite burn vs impulse", burn_as_transfer),
        view="xy",
        extra_paths=[("impulse approximation", r_inst, IMPULSE_COLOR, "--")],
    )

    fig.suptitle(
        "SSAPy-Toolkit orbital maneuver overview\n"
        "Each panel uses blue for the starting orbit, green for the target/final orbit, and red for the maneuver path.",
        fontsize=16,
    )
    return fig


def _make_staged_optimal_figure(results):
    fig, axes = plt.subplots(2, 2, figsize=(17, 14), constrained_layout=True)
    axes = axes.ravel()
    staged = results["staged_optimal"]
    panels = [
        ("Direct leave-now baseline", staged["Direct leave-now"], "xy"),
        ("Immediate one-stop staging", staged["Immediate one-stop"], "xy"),
        ("Timed one-stop staging", staged["Timed one-stop"], "xz"),
        ("Timed two-stop min-time", staged["Timed two-stop min-time"], "xz"),
    ]
    for ax, (name, result, view) in zip(axes, panels):
        _plot_staged_panel(ax, result, _staged_title(name, result), view=view)
    fig.suptitle(
        "Explicit staged optimal transfer search\n"
        "Blue is the starting orbit, green is the final orbit, colored dashed curves are staging orbits, and colored solid curves are sequential transfer legs.",
        fontsize=15,
    )
    return fig


def _make_staged_timeline_figure(results):
    staged = results["staged_optimal"]
    names = list(staged)
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    leg_colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    wait_label_added = False

    for row, name in enumerate(names):
        result = staged[name]
        legs = result.get("stage_legs") or [result]
        y = len(names) - row - 1
        for leg_index, leg in enumerate(legs):
            depart = float(leg.get("diagnostics", {}).get("t_depart", leg.get("initial", {}).get("t", 0.0))) / 3600.0
            arrive = float(leg.get("diagnostics", {}).get("t_arrive", leg.get("final", {}).get("t", 0.0))) / 3600.0
            ax.plot(
                [depart, arrive],
                [y, y],
                color=leg_colors[leg_index % len(leg_colors)],
                linewidth=4.0,
                solid_capstyle="round",
                label="transfer leg" if row == 0 and leg_index == 0 else None,
            )
            if leg_index < len(legs) - 1:
                next_depart = float(legs[leg_index + 1].get("diagnostics", {}).get("t_depart", arrive * 3600.0)) / 3600.0
                if next_depart > arrive:
                    ax.broken_barh(
                        [(arrive, next_depart - arrive)],
                        (y - 0.24, 0.48),
                        facecolors="#9e9e9e",
                        alpha=0.28,
                        label="optimized wait" if not wait_label_added else None,
                    )
                    wait_label_added = True
        burns = result.get("burns") or []
        for burn_index, burn in enumerate(burns):
            event_time = _burn_event_time(result, burn, burn_index) / 3600.0
            delta_v = float(burn.get("delta_v_mag", 0.0))
            marker_size = 45.0 + 0.03 * delta_v
            ax.scatter(
                event_time,
                y,
                s=marker_size,
                color=IMPULSE_COLOR,
                marker="*",
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
                label="impulse" if row == 0 and burn_index == 0 else None,
            )
            ax.annotate(
                f"{_burn_label(burn_index, len(burns))}\n{delta_v:.0f} m/s",
                xy=(event_time, y),
                xytext=(0, 18 if burn_index % 2 == 0 else -24),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": IMPULSE_COLOR, "alpha": 0.88},
                arrowprops={"arrowstyle": "-", "color": IMPULSE_COLOR, "lw": 0.7},
            )

        ax.text(
            float(result.get("diagnostics", {}).get("t_arrive", result.get("tof", 0.0))) / 3600.0 + 0.25,
            y,
            f"{result['delta_v_total']:.0f} m/s, {_format_waits(result)}",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)))
    ax.set_xlabel("mission elapsed time [hours]")
    ax.set_title("Staged optimal transfer burn timing and optimized waits")
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.legend(loc="upper right", fontsize=9)
    return fig


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    results = _build_maneuver_results(fast=fast)
    summary_delta_v = _summarize_delta_v({key: value for key, value in results.items() if key != "burn_conversion"})
    output_path = None
    staged_output_path = None
    staged_timeline_output_path = None
    fig = None
    staged_fig = None
    staged_timeline_fig = None
    if make_figures:
        fig = _make_summary_figure(results)
        output_path = figsave(fig, f"{FIGDIR}/orbital_maneuvers_overview.jpg")
        staged_fig = _make_staged_optimal_figure(results)
        staged_output_path = figsave(staged_fig, f"{FIGDIR}/orbital_maneuvers_staged_optimal.jpg")
        staged_timeline_fig = _make_staged_timeline_figure(results)
        staged_timeline_output_path = figsave(staged_timeline_fig, f"{FIGDIR}/orbital_maneuvers_staged_timeline.jpg")

    return {
        "title": "Orbital Maneuvers Overview",
        "description": "Consolidated SSATK demo for impulsive, continuous, fixed-time, optimized, staged, and finite-burn maneuvers.",
        "results": results,
        "summary_delta_v": summary_delta_v,
        "figure": fig,
        "staged_figure": staged_fig,
        "staged_timeline_figure": staged_timeline_fig,
        "output_path": output_path,
        "staged_output_path": staged_output_path,
        "staged_timeline_output_path": staged_timeline_output_path,
        "output_paths": [path for path in (output_path, staged_output_path, staged_timeline_output_path) if path is not None],
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
