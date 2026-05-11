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

   from ssapy_toolkit.Orbital_Mechanics import keplerian

   # Use keplerian routines and plotting helpers here


Relationship to SSAPy
----------------------

SSAPy Toolkit is designed as an extension library for `SSAPy <https://github.com/llnl/SSAPy/tree/main>`_, which provides high-fidelity orbital modeling and analysis across LEO through the cislunar regime. SSAPy handles orbit propagation, force models, integrators, and rich coordinate/frame support; SSAPy Toolkit builds on top of that to provide convenience utilities for data IO, plotting (including ground tracks and cislunar visualizations), and higher-level orbital mechanics helpers.
