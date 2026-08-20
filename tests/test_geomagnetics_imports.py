"""
tests/test_geomagnetics_imports.py
==================================
Import-structure tests for ssapy_toolkit.geomagnetics.

Why this file exists
--------------------
geomagnetics.py shipped on main with a circular import that made
`import ssapy_toolkit.geomagnetics` fail outright:

    geomagnetics
      -> ssapy_toolkit.plots            (for magnetosphere_core)
      -> plots/__init__ auto-imports every module in the package
      -> magfield_plot_3d
      -> geomagnetics                   (still half-initialised)
      -> ImportError: cannot import name '_geo_to_gsm_matrix'

Nothing caught it. The whole existing suite imports ssapy_toolkit.plots first
-- directly, or via `from ssapy_toolkit.plots import magfield_plot_3d` -- and
entering the cycle from that side happens to work. The failure only appears
when geomagnetics is the *first* thing imported in a fresh interpreter, which
no test did.

So these tests are deliberately about import mechanics rather than physics.
Each one runs in a subprocess with a clean interpreter, because once a module
is in sys.modules the cycle cannot reproduce -- testing this in-process would
pass whether or not the bug were present, which is exactly the trap the
original suite fell into.

The physics itself is covered by tests/test_magfield_physics.py.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _import_in_fresh_interpreter(statement: str) -> subprocess.CompletedProcess:
    """Run one import statement in a new interpreter and return the result.

    A subprocess is the point: sys.modules is empty, so module resolution
    order is genuinely exercised instead of being masked by whatever a
    previous test already imported.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(statement)],
        capture_output=True, text=True, timeout=300,
    )


# ---------------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------------

def test_geomagnetics_imports_first_in_a_clean_interpreter():
    """geomagnetics must import with nothing else imported first.

    This is the exact failure that shipped: it works if you import
    ssapy_toolkit.plots beforehand, and raises ImportError if you do not.
    """
    proc = _import_in_fresh_interpreter("""
        import ssapy_toolkit.geomagnetics as g
        assert hasattr(g, "_bfield_batch"), "physics functions missing"
        print("OK")
    """)
    assert proc.returncode == 0, (
        "importing geomagnetics on its own failed:\n" + proc.stderr[-2000:]
    )


def test_magfield_imports_first_in_a_clean_interpreter():
    """The other side of the cycle: magfield first, geomagnetics not yet loaded."""
    proc = _import_in_fresh_interpreter("""
        import ssapy_toolkit.plots.magfield_plot_3d as mf
        assert hasattr(mf, "_bfield_batch"), "re-exports missing"
        print("OK")
    """)
    assert proc.returncode == 0, (
        "importing magfield_plot_3d on its own failed:\n" + proc.stderr[-2000:]
    )


@pytest.mark.parametrize("first", [
    "ssapy_toolkit.geomagnetics",
    "ssapy_toolkit.plots.magfield_plot_3d",
    "ssapy_toolkit.plots.magnetosphere_core",
    "ssapy_toolkit.plots.van_allen_plot_3d",
    "ssapy_toolkit.plots",
    "ssapy_toolkit",
])
def test_any_module_can_be_imported_first(first):
    """No module in the magnetosphere group may depend on import order.

    Parametrised rather than written as one test because the bug was
    order-dependent: four of these six succeeded while the other two failed,
    and a single entry point would have had an even chance of missing it.
    """
    proc = _import_in_fresh_interpreter(f"""
        import {first}
        import ssapy_toolkit.geomagnetics
        import ssapy_toolkit.plots.magfield_plot_3d
        print("OK")
    """)
    assert proc.returncode == 0, (
        f"importing {first} first broke the group:\n" + proc.stderr[-1500:]
    )


# ---------------------------------------------------------------------------
# The structure that makes the above true
# ---------------------------------------------------------------------------

def test_geomagnetics_does_not_import_the_plots_package():
    """geomagnetics must reach magnetosphere_core as a sibling, not via the package.

    `from .plots.magnetosphere_core import ...` was what created the cycle: it
    pulls in plots/__init__, which auto-imports every module in the package,
    including the one that imports back into this module. A plain
    `from .magnetosphere_core import ...` resolves the single module and
    nothing else.
    """
    import ast
    import pathlib
    from ssapy_toolkit import geomagnetics

    source = pathlib.Path(geomagnetics.__file__).read_text(encoding="utf-8")
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            # level>0 is a relative import; "plots.x" as a relative target means
            # it is going up and back down through the package __init__.
            if node.level > 0 and node.module.startswith("plots."):
                offenders.append(f"line {node.lineno}: from {'.' * node.level}{node.module}")
    assert not offenders, (
        "geomagnetics imports through the plots package, which re-creates the "
        "circular import: " + "; ".join(offenders)
    )


def test_external_model_state_is_shared_not_copied():
    """Both modules must see the same _EXTERNAL_MODEL.

    magfield_plot_3d re-exports geomagnetics' names for backwards
    compatibility. If the re-export ever became a copy rather than a
    reference, set_external_model() would update one module's idea of the
    state and not the other's -- which is the silent-no-op bug the guarded
    module attribute was added to prevent.
    """
    from ssapy_toolkit import geomagnetics
    from ssapy_toolkit.plots import magfield_plot_3d

    saved = geomagnetics.get_external_model()
    try:
        sentinel = object()
        geomagnetics.set_external_model(sentinel)
        assert magfield_plot_3d.get_external_model() is sentinel, (
            "magfield_plot_3d does not see geomagnetics' external-model state"
        )
    finally:
        geomagnetics.set_external_model(saved)