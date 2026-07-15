# SSAPy Toolkit

**SSAPy Toolkit** (Python package: `ssapy_toolkit`, sometimes abbreviated
*SSATK*) is a collection of higher-level, analysis-ready extensions for the
[SSAPy](https://github.com/llnl/SSAPy) orbital-modeling ecosystem. Where SSAPy
provides the core high-fidelity propagation and modeling engine, the Toolkit
adds astrodynamics utilities, orbital-transfer design, coordinate/time
conversions, brightness and observables modeling, launch and rocket helpers,
integrators, rich plotting, and data I/O to support day-to-day research and
engineering workflows.

SSAPy itself is a fast, flexible, high-fidelity orbital modeling and analysis
tool for orbits spanning from low-Earth orbit into the cislunar regime, with
configurable force models (Earth and lunar gravity, radiation pressure, drag,
planetary perturbations, maneuvers), multiple integrators, orbit determination,
Monte Carlo / uncertainty-quantification workflows, and ground/space observer
models. See the SSAPy repository for full details:
<https://github.com/llnl/SSAPy>.

---

## Features

- **Orbital mechanics & astrodynamics** — Keplerian utilities, ellipse fitting,
  r/v conversions, Lagrange points, and synthetic orbit populations.
- **Orbital transfers** — a full transfer-design suite: Hohmann, coplanar,
  Lambert, inclination-change, and rendezvous transfers; continuous-thrust
  transfers; shooter and optimal-transfer methods; and burn-to-delta-v
  conversions with finite-burn modeling.
- **Coordinate transforms & time conversions** — GCRF-to-ITRF, GCRF-to-NTW,
  GCRF-to-LLH/lon-lat, GCRF-to-lunar, J2000-to-GCRF, Cartesian/spherical/
  cylindrical, equatorial/ecliptic, and sky-angle helpers, using a right-handed
  NTW (N = T x W) convention consistent with SSAPy.
- **Observables & brightness modeling** — Lambertian magnitude / brightness,
  including object thermal emission, Earth-shadow effects, and ground
  reflectance.
- **Plotting & visualization** — orbit, ground-track, cislunar (2-D and 3-D),
  and transfer plots; interactive dashboards; and animated GIF/video output.
- **Integrators** — Runge-Kutta (RK4), leapfrog, and gravity-turn integrators.
- **Launch & rockets** — launch-pad definitions, gravity-turn ascent, and
  fuel/burn utilities.
- **Data I/O** — HDF5 helpers (including dictionary/HDF5 conversion with array
  handling and selective key loading), plus CSV, JSON, XML, and pickle I/O, and
  TLE/3LE parsing.
- **SSAPy wrappers & HPC helpers** — convenience wrappers around SSAPy orbits,
  propagators, and satellite keyword arguments, plus utilities for HPC
  workflows.
- **Demo gallery** — a runnable gallery of worked examples with inline output.

---

## Installation

SSAPy Toolkit is a standard Python package.

```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev]
```

This installs the package in editable mode along with development dependencies
(testing, linting, docs tools, etc.). SSAPy Toolkit builds on SSAPy; see the
[SSAPy](https://github.com/llnl/SSAPy) repository for its installation details.

---

## Usage

The top-level package does not re-export submodules, so import the specific
module you need:

```
from ssapy_toolkit.orbital_mechanics import keplerian
from ssapy_toolkit.orbital_mechanics import transfer_hohmann
from ssapy_toolkit.coordinates import gcrf_to_itrf
from ssapy_toolkit.plots import orbit_plot
```

More detailed examples can be found in the `demos/` directory. To render the
full demo gallery as a visualization document:

```
ssapy-demo-gallery
```

---

## Development

To run the test suite:

```
pytest tests
```

Code formatting and linting are handled via `flake8` (see `.flake8` for
configuration).

---

## Documentation

Project documentation is built with Sphinx and hosted on Read the Docs.
Once configured, the latest documentation will be available at:

<https://ssapy-toolkit.readthedocs.io>

To build the docs locally (after installing dev dependencies):

```
cd docs
make html
```

The built HTML files will be in `docs/_build/html/`.

---

## Contributing

Contributions are welcome via pull request against the `main` branch. Work that
primarily concerns the core propagation/modeling engine should target the
[SSAPy](https://github.com/llnl/SSAPy) repository instead.

---

## License

SSAPy Toolkit is distributed under the terms of the BSD 3-Clause license. All
new contributions must be made under the same license. See the
[LICENSE](LICENSE) file for details.

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-2015996
