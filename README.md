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

The Toolkit is intended to be the main user-facing entry point. Shared SSAPy
constants are available through the Toolkit, so users do not need to know
whether a constant originates in base SSAPy or Toolkit-specific helpers:

```
import ssapy_toolkit as ssatk

print(ssatk.EARTH_MU)
print(ssatk.constants.RGEO)

orbit = ssatk.Orbit.fromKeplerianElements(
    a=ssatk.constants.RGEO,
    e=0.0,
    i=0.0,
    pa=0.0,
    raan=0.0,
    trueAnomaly=0.0,
    t=0.0,
)
r, v = ssatk.rv(orbit, time=[0.0, 60.0])
```

Base SSAPy core objects such as `Orbit`, `rv`, `groundTrack`, and `AccelKepler`
are lazily available through `ssapy_toolkit`. Toolkit duplicate helpers take
precedence at the top level, so names such as `ssapy_toolkit.norm`,
`ssapy_toolkit.deg0to360`, and `ssapy_toolkit.period` resolve to Toolkit
implementations. Toolkit submodules also win on collisions such as
`ssapy_toolkit.io` and `ssapy_toolkit.utils`; the base package remains available
as `ssapy_toolkit.ssapy` when direct SSAPy module access is needed.

For workflow functions, import the specific Toolkit module you need:

```
from ssapy_toolkit.orbital_mechanics import keplerian
from ssapy_toolkit.orbital_mechanics import transfer_hohmann
from ssapy_toolkit.orbital_mechanics import transfer_bielliptic
from ssapy_toolkit.coordinates import gcrf_to_itrf
from ssapy_toolkit.plots import orbit_plot
```

`transfer_bielliptic` computes the analytic three-impulse, two-half-ellipse
transfer between coplanar circular orbits through an intermediate apoapsis
radius. It is useful for quick radius-to-radius trade studies; use
`transfer_ssapy` or `transfer_optimal` when fixed epochs, target phasing,
perturbed propagation, or non-circular boundary states matter.
Transfer entry points accept either SSAPy `Orbit` objects (`orbit1`/`orbit2`,
`initial`/`target`) or raw inertial state vectors (`r1, v1, r2, v2`). For
`transfer_optimal`, set `departure_mode="now"` or `leave_now=True` to depart
from the supplied state; leave the default `departure_mode="optimize"` to search
for the best departure phase/time.
Set `stage_mode="immediate"` or `stage_mode="timed"` to explicitly search
staged transfers through candidate staging orbits; `stage_mode="best"` compares
the direct and staged routes. Timed staging allows each post-stage leg to wait
for an appropriate phase instead of leaving the staging orbit immediately. The
default `n_stage_stops=1` searches one intermediate staging orbit; increase
`n_stage_stops` and set `stage_beam_width` for bounded multi-stop searches.

`orbit_plot` is the main entry point for in-space trajectory plots. It keeps the
legacy four-panel orbit view by default, and also accepts compact selectors for
common slices and cislunar views:

```
orbit_plot(r, t, frame="gcrf")                         # xy, xz, yz, and 3-D
orbit_plot(r, t, view="xy", frame="itrf")             # one 2-D slice
orbit_plot(r, t, view=("xy", "xz", "3d"))            # custom panels
orbit_plot(r, t, view="lunar_yz")                     # lunar-fixed YZ slice
orbit_plot(r, t, view="lunar_xy", coordinate="gcrf")  # override coordinates
orbit_plot(r, t, view="ground track")                 # wide ground-track map
orbit_plot(r, t, view=("groundtrack", "globe"))       # map + 3-D globe
orbit_plot(r, t, view="dashboard")                    # map, globe, and slices
orbit_plot(r, t, view="cislunar_3d")                  # lunar-fixed 3-D view
orbit_plot(r, t, view="cislunar_xy")                  # GCRF + lunar XY views
orbit_plot(r, t, view="cislunar_dashboard")           # cislunar dashboard
orbit_plot(r, t, view="xy", save="quicklooks/orbit.mp4")  # animated MP4
orbit_plot(r, t, view="xy", save="quicklooks/orbit.gif")  # animated GIF
```

All plotting helpers accept `save`, `savefig`, `save_fig`, `save_figure`,
`savepath`, and `save_path` as equivalent save-path keywords. Relative names
are saved under `~/ssatk_figures`; absolute paths are used exactly as provided.
Set `SSATK_FIGURES_DIR` to choose a different figure-output root explicitly.
Use `ssatk_path` and `ssatk_fig` for direct path and figure-save helpers.

For general data products, `ssatk_save` and `ssatk_load` choose the storage
format from the file extension. Bare and relative data filenames are rooted
under `~/ssatk_data`; bare and relative figure filenames are rooted under
`~/ssatk_figures`; absolute paths are honored.

```
ssatk.ssatk_save({"r": r, "v": v, "t": t}, "runs/orbit.h5")
state = ssatk.ssatk_load("runs/orbit.h5")

ssatk.ssatk_save(r, "arrays/state.npy")
ssatk.ssatk_save({"r": r, "v": v}, "arrays/state.npz")
ssatk.ssatk_save(table, "tables/summary.csv")
ssatk.ssatk_save(fig, "quicklooks/orbit.png")
```

For keyed HDF5 or NPZ outputs, pass `key=`. Non-mapping objects default to
`"data"`; dictionaries use their own keys; nested dictionaries become nested
HDF5 groups or slash-delimited NPZ members.

More detailed examples can be found in the `demos/` directory. To render the
full demo gallery as a visualization document:

```
ssapy-demo-gallery
```

The command can be run from any directory after installation. It writes the
HTML report to `~/ssatk_figures/demo_gallery/index.html` by default and prints
the exact output path when it finishes. Use `--open` to open the report in a
browser, `--output PATH` to choose a different output directory, or
`--demos-dir PATH` to run demos from a source checkout explicitly. The default
does not fall back to the clone directory; set `SSATK_FIGURES_DIR` if you want
a non-home output root.

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
