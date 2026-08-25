"""Create a captioned PDF report from the benchmark gallery figures."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ssapy_toolkit.io.ssatk_data import ssatk_data
from ssapy_toolkit.plots.figpath import figpath

PLOT_ORDER = (
    "artemis_benchmark_cislunar_context.png",
    "artemis_benchmark_position_error.png",
    "artemis_benchmark_velocity_error.png",
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
)

CAPTIONS = {
    "artemis_benchmark_cislunar_context.png": "Cislunar context for the Artemis II/Orion benchmark; the trajectory is compared with JPL Horizons state-vector data.",
    "artemis_benchmark_position_error.png": "Artemis II/Orion position residuals for the SSAPy Kepler propagation against the JPL Horizons reference states.",
    "artemis_benchmark_velocity_error.png": "Artemis II/Orion velocity residuals for the SSAPy Kepler propagation against the JPL Horizons reference states.",
    "gmat_two_body_position_error.png": "Short two-body position residuals: SSATK DOP853 versus GMAT R2026a RungeKutta89 using an Earth degree/order-0 point mass.",
    "gmat_two_body_velocity_error.png": "Short two-body velocity residuals: SSATK DOP853 versus GMAT R2026a RungeKutta89 using an Earth degree/order-0 point mass.",
    "orekit_two_body_position_error.png": "Short two-body position residuals: SSATK DOP853 versus Orekit 10.3.1 KeplerianPropagator.",
    "orekit_two_body_velocity_error.png": "Short two-body velocity residuals: SSATK DOP853 versus Orekit 10.3.1 KeplerianPropagator.",
    "long_term_summary.png": "RMS position and velocity residuals for the matched 7-day LEO and 30-day GEO/cislunar-radius two-body cases. The annotation reports the GMAT JGM2 versus SSATK Earth gravitational-parameter difference.",
    "long_term_leo_residuals.png": "Seven-day LEO two-body residual histories for GMAT and Orekit relative to SSATK DOP853; the GMAT JGM2 μ mismatch is shown in the title.",
    "long_term_geo_residuals.png": "Thirty-day GEO two-body residual histories for GMAT and Orekit relative to SSATK DOP853; the GMAT JGM2 μ mismatch is shown in the title.",
    "long_term_cislunar_radius_residuals.png": "Thirty-day Earth-centered two-body residual histories at lunar orbital radius. This is not an Earth–Moon–Sun dynamical model.",
    "nbody_summary.png": "RMS residuals for the Earth–Moon–Sun and full planetary point-mass ladders across LEO, GEO, and cislunar-radius cases.",
    "nbody_earth_moon_sun_leo_residuals.png": "Seven-day LEO residuals with Earth, Moon, and Sun point-mass perturbations.",
    "nbody_earth_moon_sun_geo_residuals.png": "Thirty-day GEO residuals with Earth, Moon, and Sun point-mass perturbations.",
    "nbody_earth_moon_sun_cislunar_radius_residuals.png": "Thirty-day cislunar-radius residuals with Earth, Moon, and Sun point-mass perturbations.",
    "nbody_solar_system_leo_residuals.png": "Seven-day LEO residuals with Earth, Moon, Sun, and the seven additional planetary point masses.",
    "nbody_solar_system_geo_residuals.png": "Thirty-day GEO residuals with Earth, Moon, Sun, and the seven additional planetary point masses.",
    "nbody_solar_system_cislunar_radius_residuals.png": "Thirty-day cislunar-radius residuals with Earth, Moon, Sun, and the seven additional planetary point masses.",
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
        path = Path(ssatk_data(f"benchmarks/{name}"))
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
    """Write a captioned PDF containing every benchmark PNG in ``output_dir``."""
    output_dir = Path(output_dir or figpath("figures/benchmarks"))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _figure_paths(output_dir)
    report_path = output_dir / "ssatk_propagation_benchmark_report.pdf"
    loaded = _load_summaries(summaries)
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
