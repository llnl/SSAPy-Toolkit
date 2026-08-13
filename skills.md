# SSAPy Toolkit Skill Guide for AI Agents

This guide helps AI coding agents use or modify `ssapy-toolkit` when the
package is installed from PyPI or cloned from GitHub.

## Package Identity

- PyPI package: `ssapy-toolkit`
- Import package: `ssapy_toolkit`
- GitHub repository: `https://github.com/llnl/SSAPy-Toolkit`
- Core dependency: `llnl-ssapy` imported as `ssapy`
- External data dependency: `llnl-ssapy-data` imported as `ssapy_data`
- Minimum Python version: Python 3.10

## Install and Import

Install from PyPI for normal use:

```bash
python -m pip install ssapy-toolkit
```

Install from a clone for development:

```bash
git clone https://github.com/llnl/SSAPy-Toolkit.git
cd SSAPy-Toolkit
python -m pip install -e .[dev]
```

Basic import check:

```python
import ssapy
import ssapy_toolkit as ssatk
import ssapy_toolkit.data as ssatk_data

print(ssatk.__version__)
print(ssatk.EARTH_MU)
print(ssatk_data.data_package_available())
```

Use Toolkit as the main user-facing entry point where possible. Shared
astrodynamics constants are re-exported from base SSAPy through
`ssapy_toolkit.constants` and lazy top-level attributes such as
`ssapy_toolkit.EARTH_MU`, so user code does not need to import
`ssapy.constants` directly.

Base SSAPy core objects are also lazily exposed through the Toolkit entry point.
Prefer examples like `import ssapy_toolkit as ssatk`, then `ssatk.Orbit`,
`ssatk.rv`, `ssatk.groundTrack`, and `ssatk.AccelKepler` when writing
user-facing Toolkit code. If a helper exists in both packages and Toolkit
already has a public implementation, the top-level `ssatk` alias should resolve
to the Toolkit implementation, for example `ssatk.norm`, `ssatk.deg0to360`, and
`ssatk.period`. Toolkit submodules win on name collisions (`ssatk.io` is Toolkit
I/O, not `ssapy.io`); use `ssatk.ssapy` for direct access to the base SSAPy
package when a base submodule is required.

## Data Access Rules

Do not add reusable datasets, generated images, videos, notebooks with embedded
outputs, archives, or binary payloads to this repository. Toolkit code should
read packaged data through `ssapy_toolkit.data`, which defaults to the
`ssapy_data/data` directory supplied by `llnl-ssapy-data`.

Use these helpers instead of hard-coded repository-relative paths:

```python
from ssapy_toolkit.data import data_path, read_data_binary, read_data_text

text = read_data_text("README.md")
blob = read_data_binary("catalogs/example.bin")

with data_path("catalogs/example.csv") as path:
    # Use path only inside this context; wheels may extract resources temporarily.
    print(path)
```

If a new Toolkit function needs data, add the data to SSAPy-Data and publish a
new `llnl-ssapy-data` wheel instead of committing the data here.

For demo-only public sample data, use
`ssapy_toolkit.io.demo_data.ensure_demo_data_file`. It checks `~/ssatk_data` and
nearby `ssatk_data` caches first, optionally downloads from a configured public
source, and returns `None` with `DemoDataUnavailableWarning` when offline or
unavailable so demos/tests can skip gracefully.

## Module Map

- `ssapy_toolkit.accelerations`: acceleration models and maneuver accelerations.
- `ssapy_toolkit.compute`: numerical helpers, sampling, metrics, and brightness calculations.
- `ssapy_toolkit.coordinates`: GCRF, ITRF, NTW, LLH, lunar, and sky-coordinate transforms.
- `ssapy_toolkit.integrators`: RK4, leapfrog, gravity-turn, and integration utilities.
- `ssapy_toolkit.io`: CSV, HDF5, JSON, XML, pickle, file listing, and TLE/3LE helpers.
- `ssapy_toolkit.orbital_mechanics`: Keplerian elements, transfers, burns, orbit fitting, and orbit statistics.
- `ssapy_toolkit.plots`: orbit, cislunar, ground-track, dashboard, GIF, and video plotting helpers.
- `ssapy_toolkit.ssapy_wrappers`: convenience wrappers around SSAPy orbits and propagation.
- `demos/`: runnable examples that should stay small and avoid checked-in generated outputs.

For transfer work, prefer `transfer_ssapy` for fixed boundary states and
`transfer_optimal` for departure/time-of-flight/phase searches. Use
`transfer_bielliptic` only for the analytic three-impulse, coplanar circular
orbit-to-orbit case through an intermediate apoapsis; it intentionally does not
solve target phasing.
Transfer solvers should accept either `orbit1`/`orbit2` SSAPy `Orbit` objects
or raw `r1, v1, r2, v2` state vectors. For `transfer_optimal`, use
`departure_mode="now"` / `leave_now=True` to depart from the supplied state, or
`departure_mode="optimize"` / `"leave_whenever"` to search departure phase.
Use `stage_mode="immediate"` for explicit staged transfers with no wait at the
staging orbit, `stage_mode="timed"` for staged transfers with an optimized
wait/phase before each post-stage leg, and `stage_mode="best"` to compare direct
versus staged routes. The default `n_stage_stops=1` searches one intermediate
staging orbit; increase `n_stage_stops` with a bounded `stage_beam_width` for
multi-stop searches.
For broad transfer-design requests, prefer the structured `transfer_optimal`
schema over long flat argument lists. Use `problem={"boundary": ..., "objective":
..., "constraints": ..., "route": ..., "solver": ...}` when the user gives
multiple boundary conditions, burn objectives, route preferences, or solver
limits. Key aliases are intentionally user-friendly: `arrival_mode` accepts
`"inject"`, `"intercept"`, `"rendezvous"`, or `"insertion"`. Use `inject` for a
free-phase departure burn onto a transfer, `intercept` for target position at a
selected time without velocity match, `rendezvous` for target position and
velocity at a selected time, and `insertion` for free-phase target-orbit velocity
match. `route` accepts `"direct"`, `"immediate"`, `"multi_stage"`, or `"best"`;
`timing="optimized"` maps to the timed staged search. Structured results include
`diagnostics["problem_schema"] == "ssatk.transfer_problem.v1"`.
The user-facing maneuver gallery is consolidated in
`demos/demo_orbital_maneuvers.py`, including staged optimal geometry, burn
timeline figures, and elliptical GEO-or-below direct-vs-staged comparisons.
Avoid adding one-off transfer demos unless a new workflow cannot fit that
all-in-one summary. Put solver-specific regression coverage in
`tests/test_orbital_maneuver_modes.py` or adjacent transfer tests.

Prefer correctly spelled module paths in new code:
`coordinates.equatorial_and_ecliptic`, `coordinates.local_and_equatorial`,
`accelerations.accel_equatorial`, and
`orbital_mechanics.all_orbit_quantities`. Do not add misspelled compatibility
modules or aliases for new APIs.
`ssapy_toolkit.launch_pads` is the canonical launch/test-site metadata module;
`ssapy_toolkit.orbital_mechanics.launch_pads` re-exports the same dictionaries.

## Development Workflow

Before changing behavior, search for an existing implementation:

```bash
rg "def function_name|class ClassName|keyword" ssapy_toolkit demos tests
```

Prefer small shared helpers when two modules implement the same private utility.
Keep shared helpers lightweight so imports do not force heavy packages such as
`pandas`, `astropy`, plotting backends, or SSAPy unless they are required.
Use `ssapy_toolkit._paths.ensure_file_parent` before writing nested output
files from lightweight IO helpers instead of duplicating parent-directory
creation logic.
Use `ssapy_toolkit._namespace.import_public_modules` for legacy subpackage
namespace imports, and `ssapy_toolkit.time_functions._gps._to_gps_seconds` for
private GPS-second conversion from floats or `astropy.time.Time` objects.

Use `ssapy_toolkit.plots.ssatk_path` / `ssapy_toolkit.plots.ssatk_fig` for
figure outputs; `figpath`, `figsave`, `fpath`, and `fsave` remain short aliases.
Plot functions accept `save`, `savefig`, `save_fig`, `save_figure`, `savepath`,
and `save_path`; relative filenames go under `~/ssatk_figures`, while absolute
paths are honored. Use `ssapy_toolkit.io.datapath.datapath` (`dpath`) or
`ssapy_toolkit.io.ssatk_data.ssatk_data` for local user data/cache outputs; do
not add those outputs to the repository. Prefer `h5cache` / `h5load` or the
`ssatk_save_cache` / `ssatk_load_cache` convenience wrappers for HDF5 caches.
Use `ssapy_toolkit.io.ssatk_save.ssatk_save` and `ssatk_load` as the generic
extension-dispatched save/load entry points for data products; `ssatk_load` is
reserved for this universal loader, not the HDF5 cache wrapper.

For plotting cleanup, keep public plot names as stable wrappers and place shared
implementation in private core modules. Current examples are
`ssapy_toolkit.plots._orbit_plot_core._orbit_plot_core` for `orbit_plot`,
`orbit_plot_xy`, and `orbit_plot_xyxz`, and
`ssapy_toolkit.plots._cislunar_plot_core._cislunar_plot_core` for `cislunar_plot`,
`cislunar_plot_3d`, and `cislunar_plot_xy`.
Prefer `ssapy_toolkit.plots.orbit_plot` as the main public entry point for
in-space plots. It accepts selectors such as `view="xy"`, `view="3d"`,
`view=("xy", "xz", "3d")`, `view="lunar_yz"`, `view="ground track"`,
`view="globe"`, `view="dashboard"`, `view="cislunar_3d"`,
`view="cislunar_xy"`, `view="cislunar_dashboard"`,
`view="transfer_trajectory"`, `view="transfer_burn_profile"`,
`view="transfer_designer"`, and `view="divergence"`; older wrappers remain for
compatibility. `frame`, `coordinate`, and `coordinates` are aliases; `lunar_*`
views default to `coordinate="lunar_fixed"` unless explicitly overwritten.
Ground-track views reserve a two-column-wide subplot; when `view="groundtrack"`
is the only view, the figure uses a one-row, two-column aspect. Mixed layouts
backfill one-column views into current-row gaps before wrapping a wide
ground-track panel. For `orbit_plot`, `.mp4` and `.gif` save paths create
animated quicklooks with short fading tails. `.png`, `.jpg`, and other static
image extensions save the full time-series figure.

For new or changed public behavior:

- Add or update a focused test under `tests/` when an adjacent test pattern exists.
- Add or update a demo under `demos/` for user-facing workflows.
- Keep demos deterministic, short-running, and free of generated data artifacts.
- Keep version metadata synchronized in `pyproject.toml` and `ssapy_toolkit/__init__.py` when preparing a release.
- Preserve the existing file structure unless the cleanup removes duplication or fixes a structural issue.

## Validation Commands

Run focused checks before committing:

```bash
python scripts/check_repository_policy.py
python -m pytest -q tests/test_data_access.py
python -m pytest -q tests/test_demos_easy.py
python -m ruff check <changed-python-files>
git diff --check
```

Run the full demo gallery only after demo/plot changes are otherwise stable, or
as a final release/PR validation step. Prefer the local module invocation from a
clone to avoid stale console scripts from another install:

```bash
python -m ssapy_toolkit.run_all_demos --output /tmp/ssatk-demo-gallery
```

Build-check release metadata when package dependencies or versioning changes:

```bash
python -m build
python -m twine check dist/*
```

## Common Pitfalls

- Do not use Git LFS as the default solution for Toolkit data.
- Do not commit generated figures, movies, cache files, large datasets, or local environment files.
- Do not use absolute paths or `..` traversal for packaged data resources.
- Do not assume packaged resources are real filesystem paths outside a `data_path()` context.
- Do not import broad `ssapy_toolkit` subpackages only to access one small helper; prefer narrow imports.
