"""Create a captioned PDF report from the benchmark gallery figures."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from ssapy_toolkit.io.ssatk_data import ssatk_data
from ssapy_toolkit._paths import output_root
from ssapy_toolkit.plots.figpath import document_path

PLOT_ORDER = (
    "artemis_benchmark_cislunar_context.png",
    "artemis_benchmark_position_error.png",
    "artemis_benchmark_velocity_error.png",
    "artemis_benchmark_burn_events.png",
    "gmat_two_body_position_error.png",
    "gmat_two_body_velocity_error.png",
    "orekit_two_body_position_error.png",
    "orekit_two_body_velocity_error.png",
    "long_term_summary.png",
    "long_term_leo_residuals.png",
    "long_term_geo_residuals.png",
    "long_term_cislunar_radius_residuals.png",
    "nbody_summary.png",
    "nbody_earth_moon_sun_leo_residuals.png",
    "nbody_earth_moon_sun_geo_residuals.png",
    "nbody_earth_moon_sun_cislunar_radius_residuals.png",
    "nbody_solar_system_leo_residuals.png",
    "nbody_solar_system_geo_residuals.png",
    "nbody_solar_system_cislunar_radius_residuals.png",
    "benchmark_regime_matrix.png",
)

CAPTIONS = {
    "artemis_benchmark_cislunar_context.png": "Cislunar context for the Artemis II/Orion benchmark; the trajectory is compared with JPL Horizons state-vector data.",
    "artemis_benchmark_position_error.png": "Artemis II/Orion position residuals for the SSAPy Kepler propagation against the JPL Horizons reference states.",
    "artemis_benchmark_velocity_error.png": "Artemis II/Orion velocity residuals for the SSAPy Kepler propagation against the JPL Horizons reference states.",
    "artemis_benchmark_burn_events.png": "Executed Artemis II/Orion burn matches from the NASA mission timeline overlaid on position and velocity residuals. Vertical lines mark the nearest Horizons sample used for each burn synchronization.",
    "gmat_two_body_position_error.png": "Short two-body position residuals: SSATK DOP853 versus GMAT R2026a RungeKutta89 using an Earth degree/order-0 point mass.",
    "gmat_two_body_velocity_error.png": "Short two-body velocity residuals: SSATK DOP853 versus GMAT R2026a RungeKutta89 using an Earth degree/order-0 point mass.",
    "orekit_two_body_position_error.png": "Short two-body position residuals: SSATK DOP853 versus Orekit 10.3.1 KeplerianPropagator.",
    "orekit_two_body_velocity_error.png": "Short two-body velocity residuals: SSATK DOP853 versus Orekit 10.3.1 KeplerianPropagator.",
    "long_term_summary.png": "RMS position and velocity residuals for the matched 7-day LEO, 30-day GEO, and 60-day cislunar-radius two-body cases. The annotation reports the GMAT JGM2 versus SSATK Earth gravitational-parameter difference.",
    "long_term_leo_residuals.png": "Seven-day LEO two-body residual histories for GMAT and Orekit relative to SSATK DOP853; the GMAT JGM2 μ mismatch is shown in the title.",
    "long_term_geo_residuals.png": "Thirty-day GEO two-body residual histories for GMAT and Orekit relative to SSATK DOP853; the GMAT JGM2 μ mismatch is shown in the title.",
    "long_term_cislunar_radius_residuals.png": "Sixty-day Earth-centered two-body residual histories at lunar orbital radius, spanning more than two orbital periods. This is not an Earth–Moon–Sun dynamical model.",
    "nbody_summary.png": "RMS residuals for the Earth–Moon–Sun and full planetary point-mass ladders across LEO, GEO, and cislunar-radius cases.",
    "nbody_earth_moon_sun_leo_residuals.png": "Seven-day LEO residuals with Earth, Moon, and Sun point-mass perturbations.",
    "nbody_earth_moon_sun_geo_residuals.png": "Thirty-day GEO residuals with Earth, Moon, and Sun point-mass perturbations.",
    "nbody_earth_moon_sun_cislunar_radius_residuals.png": "Sixty-day cislunar-radius residuals with Earth, Moon, and Sun point-mass perturbations.",
    "nbody_solar_system_leo_residuals.png": "Seven-day LEO residuals with Earth, Moon, Sun, and the seven additional planetary point masses.",
    "nbody_solar_system_geo_residuals.png": "Thirty-day GEO residuals with Earth, Moon, Sun, and the seven additional planetary point masses.",
    "nbody_solar_system_cislunar_radius_residuals.png": "Sixty-day cislunar-radius residuals with Earth, Moon, Sun, and the seven additional planetary point masses.",
    "benchmark_regime_matrix.png": "Combined residual histories for two-body, Earth-Moon-Sun, and full planetary point-mass models across LEO, GEO, and cislunar-radius cases. Position and velocity axes use logarithmic scaling with a 1 mm floor.",
}


def _figure_paths(output_dir: Path) -> list[Path]:
    rank = {name: index for index, name in enumerate(PLOT_ORDER)}
    return sorted(
        (path for path in output_dir.glob("*.png") if path.is_file()),
        key=lambda path: (rank.get(path.name, len(rank)), path.name),
    )


def _load_summaries(summaries: dict[str, dict] | None) -> dict[str, dict]:
    if summaries is not None:
        return summaries
    loaded = {}
    for name in (
        "artemis_benchmark_results.json",
        "gmat_two_body_results.json",
        "orekit_two_body_results.json",
        "long_term_propagation_results.json",
        "nbody_propagation_results.json",
    ):
        path = Path(ssatk_data(f"data/benchmarks/{name}"))
        if path.is_file():
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _summary_lines(summaries: dict[str, dict]) -> list[str]:
    lines = []
    long_term = summaries.get("long_term_propagation_results.json")
    if long_term:
        mu = long_term.get("mu_m3_s2", {})
        lines.append(
            "Two-body μ difference: "
            f"GMAT JGM2 − SSATK = {mu.get('gmat_minus_ssatk', float('nan')):.3e} m³/s² "
            f"({mu.get('relative_difference', float('nan')):.3e} relative)."
        )
    nbody = summaries.get("nbody_propagation_results.json")
    if nbody:
        ephemerides = nbody.get("ephemeris_sources", {})
        lines.append(
            "N-body ephemerides: "
            f"SSAPy {ephemerides.get('ssatk', 'unknown')}, "
            f"GMAT {ephemerides.get('gmat', 'unknown')}, "
            f"Orekit {ephemerides.get('orekit', 'unknown')}."
        )
    return lines


def _write_regime_matrix(summaries: dict[str, dict], output_dir: Path) -> Path | None:
    """Write directly comparable residual histories for all regimes."""
    long_term = summaries.get("long_term_propagation_results.json", {})
    nbody = summaries.get("nbody_propagation_results.json", {})
    cases = {case["name"]: case for case in long_term.get("cases", [])}
    nbody_cases = {}
    for case in nbody.get("cases", []):
        nbody_cases[(case["mode"], case["name"])] = case
    labels = ("2-body", "Earth-Moon-Sun", "Full n-body")
    modes = (None, "earth_moon_sun", "solar_system")
    regimes = ("leo", "geo", "cislunar_radius")
    orbit_labels = ("LEO", "GEO", "Cislunar")
    tools = ("GMAT", "Orekit")
    from demos.benchmarks import demo_long_term_propagation_benchmark as two_body
    from demos.benchmarks import demo_nbody_propagation_benchmark as n_body

    residuals = {}
    ssatk_rows_cache = {}
    for regime in regimes:
        for mode, label in zip(modes, labels):
            case = cases.get(regime) if mode is None else nbody_cases.get((mode, regime))
            if case is None:
                continue
            for tool in tools:
                result = case.get("tools", {}).get(tool, {})
                state_path = result.get("state_path")
                if not state_path or not Path(state_path).is_file():
                    continue
                rows = np.loadtxt(state_path, delimiter=",")
                if mode is None:
                    values, _ = two_body._compare(rows, scale=1_000.0 if tool == "GMAT" else 1.0)
                else:
                    cache_key = (mode, regime)
                    if cache_key not in ssatk_rows_cache:
                        ssatk_rows_cache[cache_key] = n_body._ssatk_rows(
                            mode=mode,
                            radius=case["radius_m"],
                            duration=case["duration_s"],
                            step=case["step_s"],
                        )
                    values, _ = n_body._compare(
                        rows,
                        ssatk_rows_cache[cache_key],
                        scale=1_000.0 if tool == "GMAT" else 1.0,
                    )
                residuals[(regime, label, tool)] = values
    if not residuals:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    colors = {"GMAT": "#1976d2", "Orekit": "#ef6c00"}
    styles = {"2-body": "-", "Earth-Moon-Sun": "--", "Full n-body": ":"}
    floor = {"position": 1e-3, "velocity": 1e-3}
    for i, (regime, orbit_label) in enumerate(zip(regimes, orbit_labels)):
        for (current_regime, model_label, tool), values in residuals.items():
            if current_regime != regime:
                continue
            hours = values[:, 0] / 3_600.0
            position = np.maximum(values[:, 1], floor["position"])
            velocity = np.maximum(values[:, 2], floor["velocity"])
            line_kwargs = dict(
                color=colors[tool],
                linestyle=styles[model_label],
                linewidth=1.5,
                marker="|",
                markevery=max(1, len(values) // 16),
                label=f"{model_label} / {tool}",
            )
            axes[0, i].plot(
                hours,
                position,
                **line_kwargs,
            )
            axes[1, i].plot(
                hours,
                velocity,
                **line_kwargs,
            )
        axes[0, i].set_title(orbit_label)
        axes[1, i].set_xlabel("Elapsed time [hr]")
        for axis, ylabel in (
            (axes[0, i], "Position error [m]"),
            (axes[1, i], "Velocity error [m/s]"),
        ):
            axis.set_yscale("log")
            axis.set_ylim(bottom=1e-3)
            axis.set_ylabel(ylabel)
            axis.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Curves at the 1e-3 floor are below the plotted range; tick markers retain their time history.",
        ha="center",
        fontsize=8,
    )
    fig.suptitle("SSATK external residual histories by orbit regime and force model")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = output_dir / "benchmark_regime_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_cover(pdf: PdfPages, figure_count: int, summaries: dict[str, dict]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("#f5f7fa")
    fig.text(0.08, 0.82, "SSATK Propagation\nBenchmark Report", fontsize=28, weight="bold", color="#16324f")
    fig.text(0.08, 0.68, "GMAT, Orekit, and SSATK comparisons", fontsize=15, color="#37627f")
    scope = (
        "This report collects the benchmark figures generated by SSATK Toolkit. "
        "Residuals are norm differences between SSATK and each external tool at "
        "the reported sample epochs. The long-term cislunar-radius two-body case "
        "is not a full Earth–Moon–Sun model. N-body residuals include the listed "
        "ephemeris-version differences."
    )
    fig.text(0.08, 0.52, textwrap.fill(scope, 85), fontsize=12, color="#253746", va="top", linespacing=1.5)
    lines = [f"Included benchmark plots: {figure_count}", *_summary_lines(summaries)]
    fig.text(0.08, 0.27, "\n".join(textwrap.fill(line, 85) for line in lines), fontsize=11, color="#253746", va="top", linespacing=1.6)
    fig.text(0.08, 0.08, "SSATK Toolkit • benchmark gallery", fontsize=10, color="#6b7c8c")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _write_summary_page(pdf: PdfPages, summaries: dict[str, dict]) -> None:
    lines = ["Measured result files", ""]
    for name, summary in summaries.items():
        lines.append(name)
        if summary.get("skipped"):
            lines.append(f"  skipped: {summary.get('reason', 'unspecified')}")
            continue
        cases = summary.get("cases", [])
        if cases:
            lines.append(f"  cases: {len(cases)}")
            for case in cases[:6]:
                lines.append(f"  • {case.get('label', case.get('name', 'case'))}: {case.get('duration_s', 'n/a')} s")
        else:
            lines.append(f"  tool: {summary.get('tool', summary.get('benchmark', 'n/a'))}")
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.92, "Benchmark inputs and measured outputs", fontsize=20, weight="bold", color="#16324f")
    fig.text(0.08, 0.84, "\n".join(lines), fontsize=11, family="monospace", va="top", linespacing=1.55)
    fig.text(0.08, 0.08, "The individual pages that follow provide the plot-specific captions and model context.", fontsize=10, color="#6b7c8c")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _write_plot_page(pdf: PdfPages, path: Path) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes((0.06, 0.19, 0.88, 0.68))
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    fig.text(0.06, 0.91, path.stem.replace("_", " ").title(), fontsize=17, weight="bold", color="#16324f")
    caption = CAPTIONS.get(path.name, f"Benchmark figure generated by SSATK Toolkit: {path.name}.")
    fig.text(0.06, 0.11, textwrap.fill(caption, 105), fontsize=10.5, color="#253746", va="top", linespacing=1.4)
    fig.text(0.06, 0.055, f"Source figure: {path.name}", fontsize=8.5, color="#6b7c8c")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_benchmark_report(*, output_dir: Path | None = None, summaries: dict[str, dict] | None = None) -> str:
    """Write a captioned PDF from figures in the shared benchmark figure directory."""
    output_dir = Path(output_dir or document_path("benchmarks/report.pdf")).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ssatk_propagation_benchmark_report.pdf"
    loaded = _load_summaries(summaries)
    _write_regime_matrix(loaded, output_root() / "figures" / "benchmarks")
    paths = _figure_paths(output_root() / "figures" / "benchmarks")
    with PdfPages(report_path) as pdf:
        _write_cover(pdf, len(paths), loaded)
        _write_summary_page(pdf, loaded)
        for path in paths:
            _write_plot_page(pdf, path)
        pdf.infodict().update(
            {
                "Title": "SSATK Propagation Benchmark Report",
                "Author": "Travis Yeager",
                "Subject": "SSATK comparisons with GMAT and Orekit",
            }
        )
    return str(report_path)


__all__ = ["write_benchmark_report"]
