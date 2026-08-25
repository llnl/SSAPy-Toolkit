# Propagation benchmarks

These demos compare SSATK with reproducible reference sources and optional
external propagators. Generated figures belong under
`~/ssatk_figures/demo_gallery/figures/benchmarks/`.
Benchmark summaries belong under `~/ssatk_data/benchmarks/`.

- `demo_artemis_benchmark.py` uses JPL Horizons Artemis II/Orion state vectors
  and reports SSAPy Kepler propagation residuals.
- `demo_orekit_benchmark.py` runs Orekit's Java `KeplerianPropagator` and
  compares its Cartesian states with SSATK. It downloads Orekit 10.3.1 through
  Maven only when the demo is run outside pytest and Maven is available.

GMAT, STK, and FreeFlyer require separate licensed/native installations and
are not silently substituted by this benchmark suite.
