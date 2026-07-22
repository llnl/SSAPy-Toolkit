Usage Guide
===========

Installation
------------

Install in editable mode with development extras:

.. code-block:: bash

   pip install -e .[dev]

Basic Example
-------------

.. code-block:: python

   import ssapy_toolkit as st

   from ssapy_toolkit.orbital_mechanics import keplerian

   # Use keplerian routines and plotting helpers here


Relationship to SSAPy
----------------------

SSAPy Toolkit is designed as an extension library for `SSAPy <https://github.com/llnl/SSAPy/tree/main>`_, which provides high-fidelity orbital modeling and analysis across LEO through the cislunar regime. SSAPy handles orbit propagation, force models, integrators, and rich coordinate/frame support; SSAPy Toolkit builds on top of that to provide convenience utilities for data IO, plotting (including ground tracks and cislunar visualizations), and higher-level orbital mechanics helpers.

Packaged data
-------------

SSAPy Toolkit should not commit reusable datasets, generated figures, or other
binary artifacts. Toolkit functions that require reusable data should read it
from an installed data package instead. The expected package import name is
``ssapy_data`` with resources below ``ssapy_data/data``.

Use :mod:`ssapy_toolkit.data` when a toolkit function needs a packaged data
file:

.. code-block:: python

   from ssapy_toolkit.data import data_path, read_data_text

   catalog_text = read_data_text("catalogs/example.csv")

   with data_path("catalogs/example.csv") as catalog_path:
       # Pass catalog_path to libraries that require a filesystem path.
       print(catalog_path)

This keeps ``SSAPy-Toolkit`` source-only while allowing users to get required
data through normal ``pip`` installation once ``SSAPy-Data`` is published as a
wheel dependency.
