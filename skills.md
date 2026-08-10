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
import ssapy_toolkit
import ssapy_toolkit.data as ssatk_data

print(ssapy_toolkit.__version__)
print(ssatk_data.data_package_available())
```

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

## Development Workflow

Before changing behavior, search for an existing implementation:

```bash
rg "def function_name|class ClassName|keyword" ssapy_toolkit demos tests
```

Prefer small shared helpers when two modules implement the same private utility.
Keep shared helpers lightweight so imports do not force heavy packages such as
`pandas`, `astropy`, plotting backends, or SSAPy unless they are required.

Use `ssapy_toolkit.plots.figpath` / `ssapy_toolkit.plots.figsave` (`fpath` /
`fsave` for short imports) for figure outputs. If `figsave` receives no
`save_path`, it writes `figure.jpg` under `~/ssatk_figures`, with a local
`./ssatk_figures` fallback if the home directory is not writable. Use
`ssapy_toolkit.io.datapath.datapath` (`dpath`) for local user data/cache outputs;
do not add those outputs to the repository. Prefer `h5cache` / `h5load` for HDF5
caches; legacy `yu*` names are compatibility-only.

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
python -m flake8 <changed-python-files>
git diff --check
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
