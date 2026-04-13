# SSAPy Toolkit

**SSAPy Toolkit** (Python package: `ssapy_toolkit`) is a collection of extensions for the SSAPy ecosystem, providing tools for orbital mechanics, plotting, and data IO to support research and engineering workflows.

SSAPy itself is a fast, flexible, high-fidelity orbital modeling and analysis tool for orbits spanning from low-Earth orbit into the cislunar regime. It supports rich satellite definitions, multiple element types and coordinate frames, configurable force models (Earth and lunar gravity, radiation pressure, drag, planetary perturbations, maneuvers), a variety of integrators, Monte Carlo and UQ workflows, and extensive ground/space observer and plotting capabilities. See the SSAPy repository for full details:

https://github.com/llnl/SSAPy/tree/main

---

## Features
- Utility functions for orbital mechanics and astrodynamics
- Coordinate transforms and time conversions
- Data IO helpers for HDF5, CSV, JSON, and more
- Plotting helpers for orbits, ground tracks, and dashboards

---

## Installation

SSAPy Toolkit is a standard Python package.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev]
```

This installs the package in editable mode along with development dependencies (testing, linting, docs tools, etc.).

---

## Usage

Once installed, you can import the package in Python:

```python
import yeager_utils as yu

from yeager_utils.Orbital_Mechanics import keplerian
from yeager_utils.Plots import orbit_plot
```

More detailed examples can be found in the `demos/` directory.

---

## Development

To run the test suite:

```bash
pytest demos
```

Code formatting and linting are handled via `ruff` (see `pyproject.toml` for configuration).

---

## Documentation

Project documentation is built with Sphinx and hosted on Read the Docs.
Once configured, you will be able to find the latest documentation at:

https://ssapy-toolkit.readthedocs.io

To build the docs locally (after installing dev dependencies):

```bash
cd docs
make html
```

The built HTML files will be in `docs/_build/html/`.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
