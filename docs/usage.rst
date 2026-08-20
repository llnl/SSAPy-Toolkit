Usage Guide
===========

Installation
------------

Install in editable mode with development extras:

.. code-block:: bash

   python -m pip install -e .[dev]

Plotting installs the Python packages needed for HTML, image, GIF, and MP4
outputs, including Plotly, Matplotlib, Pillow, Kaleido, imageio, and SSAPy-Data.
Node.js 20+ is only needed for validating the JavaScript satellite-viewer source;
GitHub Actions installs it with ``actions/setup-node``.

Basic Example
-------------

.. code-block:: python

   import ssapy_toolkit as ssatk

   from ssapy_toolkit.orbital_mechanics import keplerian

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

   # Use Toolkit keplerian routines and plotting helpers around SSAPy objects.
   from ssapy_toolkit.plots import orbit_plot

   orbit_plot(r, view="xy", frame="gcrf")
   orbit_plot(r, view=("xy", "xz", "3d"), frame="itrf")
   orbit_plot(r, view="lunar_yz")
   orbit_plot(r, view="lunar_xy", coordinate="gcrf")
   orbit_plot(r, view="ground track")
   orbit_plot(r, view=("groundtrack", "globe"))
   orbit_plot(r, view="dashboard")
   orbit_plot(r, view="cislunar_3d")
   orbit_plot(r, view="cislunar_dashboard")

Plot saving
------------

All plotting helpers accept ``save``, ``savefig``, ``save_fig``,
``save_figure``, ``savepath``, and ``save_path`` as equivalent save-path
keywords. Relative filenames are saved under ``~/ssatk_figures`` and absolute
paths are honored exactly as provided:

.. code-block:: python

   from ssapy_toolkit.plots import orbit_plot, ssatk_fig, ssatk_path

   orbit_plot(r, view="xy", save="quicklook/orbit_xy")
   orbit_plot(r, view="globe", save_fig="/tmp/orbit_globe.png")
   orbit_plot(r, t, view="xy", save="quicklook/orbit_xy.mp4")
   orbit_plot(r, t, view="xy", save="quicklook/orbit_xy.gif")

   figure_path = ssatk_path("reports/summary")
   saved_path = ssatk_fig(fig, save_path=figure_path)

For :func:`ssapy_toolkit.plots.orbit_plot`, ``.mp4`` and ``.gif`` save paths
create animated quicklooks with short fading tails. Static extensions such as
``.png`` and ``.jpg`` save the full time-series figure.


Relationship to SSAPy
----------------------

SSAPy Toolkit is designed as an extension library for `SSAPy <https://github.com/llnl/SSAPy/tree/main>`_, which provides high-fidelity orbital modeling and analysis across LEO through the cislunar regime. SSAPy handles orbit propagation, force models, integrators, and rich coordinate/frame support; SSAPy Toolkit builds on top of that to provide convenience utilities for data IO, plotting (including ground tracks and cislunar visualizations), and higher-level orbital mechanics helpers.

Use ``import ssapy_toolkit as ssatk`` as the main user-facing entry point.
Shared astrodynamics constants are available through ``ssatk.constants`` and as
lazy top-level attributes such as ``ssatk.EARTH_MU``. Core SSAPy classes and
functions such as ``ssatk.Orbit``, ``ssatk.rv``, ``ssatk.groundTrack``, and
``ssatk.AccelKepler`` are also lazily available through the Toolkit. If a name
exists in both packages, Toolkit helpers and submodules take precedence; direct
base-package access remains available through ``ssatk.ssapy``.

Satellite operation frames
--------------------------

Use :mod:`ssapy_toolkit.coordinates.satellite_frames` for common satellite
operation frames. Matrices are returned as frame-to-GCRF rotations: columns are
the requested frame axes expressed in GCRF, so ``matrix @ vector_in_frame``
returns a GCRF vector.

.. code-block:: python

   from ssapy_toolkit.coordinates.satellite_frames import (
       frame_to_gcrf_matrix,
       transform_from_gcrf,
       transform_to_gcrf,
   )

   # SSAPy maneuver convention: [N, T, W].
   ntw_to_gcrf = frame_to_gcrf_matrix("ntw", r=r_gcrf, v=v_gcrf)
   thrust_gcrf = transform_to_gcrf([0.0, 1e-7, 0.0], "ntw", r=r_gcrf, v=v_gcrf)

   # Other supported selectors include "rtn", "rsw", "ric", "lvlh", "vnb",
   # "body", "nadir_velocity", "enu", "ned", "sez", "los", and "sun".
   thrust_ntw = transform_from_gcrf(thrust_gcrf, "ntw", r=r_gcrf, v=v_gcrf)

High-accuracy propagation
-------------------------

Use :func:`ssapy_toolkit.propagators_6dof.propagate_orbit_state` for adaptive
high-accuracy translational propagation. It defaults to SciPy's eighth-order
``DOP853`` method and accepts inertial perturbing acceleration callbacks.

.. code-block:: python

   import numpy as np
   from ssapy_toolkit.constants import EARTH_MU
   from ssapy_toolkit.propagators_6dof import propagate_orbit_state

   radius = 7_000_000.0
   speed = np.sqrt(EARTH_MU / radius)

   traj = propagate_orbit_state(
       r0=[radius, 0.0, 0.0],
       v0=[0.0, speed, 0.0],
       times=np.linspace(0.0, 3600.0, 121),
   )

6-DoF dynamics
--------------

Use :class:`ssapy_toolkit.dynamics.Spacecraft` for an ``Orbit``-like object
with attitude, angular rate, inertia, and mass attached. Use
:func:`ssapy_toolkit.dynamics.propagate_6dof` directly for lower-level coupled
translational and rigid-body attitude propagation. The state uses inertial
``r``/``v`` vectors, a quaternion ``q=[w, x, y, z]`` that rotates body-frame
vectors into the inertial frame, and body-frame angular rates ``omega`` in
rad/s.

.. code-block:: python

   import numpy as np
   import ssapy_toolkit as ssatk
   from ssapy_toolkit.accelerations_6dof import SpacecraftAccelJ2, constant_body_thrust
   from ssapy_toolkit.plots import orbit_plot

   sat = ssatk.Spacecraft(
       r=[7_000_000.0, 0.0, 0.0],
       v=[0.0, 7_500.0, 0.0],
       q=[1.0, 0.0, 0.0, 0.0],
       omega=[0.0, 0.0, 0.001],
       inertia=np.diag([10.0, 12.0, 8.0]),
       mass=100.0,
   )

   traj = sat.propagate(
       times=np.linspace(0.0, 600.0, 61),
       acceleration=SpacecraftAccelJ2(),
       body_acceleration=constant_body_thrust([0.0, 0.01, 0.0], sat.mass),
       gravity_gradient=True,
   )

   orbit_plot(traj.r, traj.t, view="3d")

Without an attitude-dependent ``acceleration`` callback, attitude does not feed
back into the orbital trajectory. Provide ``acceleration(t, r, v, q, omega)``
for inertial/GCRF m/s², ``ntw_acceleration(t, r, v, q, omega)`` for SSAPy
``[N, T, W]`` m/s², or ``body_acceleration(t, r, v, q, omega)`` for body-frame
m/s² when thrust, drag, solar-radiation pressure, or another orientation-
dependent force should change ``r`` and ``v``. The propagated quaternion remains
body-to-GCRF; use NTW acceleration for orbit-frame maneuvers rather than
satellite attitude.

Reusable models live in :mod:`ssapy_toolkit.accelerations_6dof` and include
``SpacecraftAccelKepler``, ``SpacecraftAccelJ2``,
``SpacecraftAccelThirdBody``, ``SpacecraftAccelDrag``,
``SpacecraftAccelSolRad``, ``SpacecraftAccelConstInertial``,
``SpacecraftAccelConstNTW``, ``SpacecraftAccelConstBody``,
``SpacecraftAccelSum``, and constant thrust/torque callback helpers.

Packaged data
-------------

SSAPy Toolkit should not commit reusable datasets, generated figures, or other
binary artifacts. Toolkit functions that require reusable data should read it
from the installed ``llnl-ssapy-data`` dependency instead. The dependency exposes
the ``ssapy_data`` import package with resources below ``ssapy_data/data``.

Use :mod:`ssapy_toolkit.data` when a toolkit function needs a packaged data
file:

.. code-block:: python

   from ssapy_toolkit.data import data_path, read_data_text

   catalog_text = read_data_text("catalogs/example.csv")

   with data_path("catalogs/example.csv") as catalog_path:
       # Pass catalog_path to libraries that require a filesystem path.
       print(catalog_path)

This keeps ``SSAPy-Toolkit`` source-only while allowing users to get required
data through normal ``pip`` installation.

Optional demo data
------------------

Demo-only files should also stay out of the repository. Demos that need public
sample data can call :func:`ssapy_toolkit.io.demo_data.ensure_demo_data_file`,
which first checks the local ``ssatk_data`` cache, then downloads from a known
public source when internet access is available. When the file cannot be found
or fetched, the helper emits ``DemoDataUnavailableWarning`` and returns
``None`` so demos and tests can skip gracefully instead of failing hard.
