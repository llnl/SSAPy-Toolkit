"""
demo_orbit_stats_dashboard_si.py

Demo script showing how to call:
    from ssapy_toolkit import orbit_stats_dashboard, synthetic_orbit_population

All units are SI:
  r in meters (m)
  v in meters/second (m/s)
  t in seconds (s)
  mu in m^3/s^2

This script:
  1) builds a synthetic population of SSAPy orbits by sampling classical elements,
  2) samples each orbit to r(t), v(t) on a common grid,
  3) runs orbit_stats_dashboard in BOTH modes:
        A) mode="population" (distribution/envelope view)
        B) mode="benchmark"  (per-model time series vs nominal)
  4) prints compact summaries and saves/displays the dashboard figures.
"""

import numpy as np

from ssapy_toolkit import synthetic_orbit_population, orbit_stats_dashboard, yufig


def print_population_summary(out, *, top_k=8):
    pop = out["population"]
    meta = out["meta"]
    per = pop["per_orbit"]

    M = int(meta["M"])
    mode = str(meta.get("mode", "population"))
    baseline = str(meta.get("baseline", "nominal"))
    ref = int(meta.get("reference", 0))

    pop_mask = np.asarray(pop.get("population_mask", np.ones((M,), dtype=bool)), dtype=bool)

    sep_max = np.asarray(per["sep_max"], dtype=float)
    sep_final = np.asarray(per["sep_final"], dtype=float)

    print(f"\n=== orbit_stats_dashboard summary (mode={mode}) ===")
    print("meta:", meta)

    x = sep_max[pop_mask]
    x = x[np.isfinite(x)]
    if x.size:
        p = np.percentile(x, [50, 90, 95, 99])
        label = f"reference orbit index {ref}" if baseline == "nominal" else baseline
        print(f"\nPosition spread vs {label} (per-orbit max ||Δr||):")
        print(f"  median: {p[0]:.3e} m")
        print(f"     p90: {p[1]:.3e} m")
        print(f"     p95: {p[2]:.3e} m")
        print(f"     p99: {p[3]:.3e} m")
        print(f"     max: {np.max(x):.3e} m")
    else:
        print("\nPosition spread: no finite data")

    order = np.argsort(sep_max.copy())
    order = order[np.isfinite(sep_max[order])]
    order = order[::-1]

    label = f"nominal (ref={ref})" if baseline == "nominal" else baseline
    print(f"\nTop {min(top_k, order.size)} orbits by max position spread vs {label}:")
    shown = 0
    for k in order:
        if not bool(pop_mask[int(k)]):
            continue
        line = (
            f"orbit={int(k):d}  "
            f"max||Δr||={sep_max[int(k)]:.3e} m  "
            f"final||Δr||={sep_final[int(k)]:.3e} m"
        )
        print(line)
        shown += 1
        if shown >= top_k:
            break

    if "vsep_max" in per:
        v_max = np.asarray(per["vsep_max"], dtype=float)[pop_mask]
        v_max = v_max[np.isfinite(v_max)]
        if v_max.size:
            pv = np.percentile(v_max, [50, 90, 95, 99])
            print(f"\nVelocity spread vs {label} (per-orbit max ||Δv||):")
            print(f"  median: {pv[0]:.3e} m/s")
            print(f"     p90: {pv[1]:.3e} m/s")
            print(f"     p95: {pv[2]:.3e} m/s")
            print(f"     p99: {pv[3]:.3e} m/s")
            print(f"     max: {np.max(v_max):.3e} m/s")


def main():
    # Generate SSAPy-based orbit population (orbit 0 is deterministic nominal by default)
    orbits, r_list, v_list, t_list, mu_si = synthetic_orbit_population(
        M=40,
        N=7200,
        dt=1.0,
        seed=1,
        # Optional ranges (SI/radians):
        # a_range_m=(6_800_000.0, 8_000_000.0),
        # e_range=(0.0, 0.02),
        # i_range_rad=(0.0, np.deg2rad(98.0)),
    )

    # Add labels for benchmark mode (used in legends); population mode ignores per-orbit labels
    labels = ["nominal"] + [f"model_{k:02d}" for k in range(1, len(r_list))]

    # -------------------------
    # A) Population mode
    # -------------------------
    out_pop = orbit_stats_dashboard(
        r_list,
        v_list=v_list,
        t_list=t_list,
        mu=mu_si,
        reference=0,
        baseline="nominal",         # or "mean"/"median"
        mode="population",
        resample="intersection",
        n_resample=3000,
        percentiles=(5, 25, 50, 75, 95),
        make_plots=True,
        plot_title="Orbit dashboard (population mode, SI, SSAPy population)",
        time_unit="s",
        r_unit="m",
        v_unit="m/s",
        labels=labels,
        show_legend=True,
    )

    print_population_summary(out_pop, top_k=8)

    fig_pop = out_pop["figure"]
    yufig(fig_pop, "tests/orbital_stats_dashboard_population.jpg")
    if fig_pop is not None:
        fig_pop.show()

    # -------------------------
    # B) Benchmark mode
    #    (use a smaller subset to keep plots readable)
    # -------------------------
    idx = np.array([0, 1, 2, 3, 4, 5], dtype=int)  # nominal + 5 "acceleration packages"
    r_list_b = [r_list[i] for i in idx]
    v_list_b = [v_list[i] for i in idx] if v_list is not None else None
    t_list_b = [t_list[i] for i in idx]
    labels_b = [labels[i] for i in idx]

    out_bench = orbit_stats_dashboard(
        r_list_b,
        v_list=v_list_b,
        t_list=t_list_b,
        mu=mu_si,
        reference=0,
        baseline="nominal",         # implied by benchmark, but explicit is fine
        mode="benchmark",
        resample="intersection",
        n_resample=3000,
        make_plots=True,
        plot_title="Orbit dashboard (benchmark mode, SI, vs nominal)",
        time_unit="s",
        r_unit="m",
        v_unit="m/s",
        labels=labels_b,
        show_legend=True,
    )

    print_population_summary(out_bench, top_k=8)

    fig_bench = out_bench["figure"]
    yufig(fig_bench, "tests/orbital_stats_dashboard_benchmark.jpg")
    if fig_bench is not None:
        fig_bench.show()


if __name__ == "__main__":
    main()
