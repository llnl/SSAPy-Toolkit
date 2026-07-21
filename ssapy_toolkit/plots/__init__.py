"""Plotting helpers for orbits, ground tracks, dashboards, and more.

Each .py file in this folder is auto-imported below, one at a time, inside
its own try/except. Previously a single bad import anywhere in this folder
(a typo, a missing optional dependency, a stale cross-reference to a file
that got renamed/moved) raised straight out of the loop and took the
*entire* ssapy_toolkit.plots package down with it — including unrelated
files like groundtrack_plot that had nothing wrong with them. That's why
`python -m ssapy_toolkit.plots.*` or `import ssapy_toolkit.plots` would
appear to hang/fail/spam warnings whenever any one new file had an issue.

Now a failure in one file is caught, reported via warnings.warn, and only
that file's names are left out of the package namespace — everything else
in the folder still imports and works normally.
"""
import importlib
import os
import warnings

sub_dir = os.path.dirname(__file__)

# Sorted for a deterministic import order (helps reproduce/debug failures).
for filename in sorted(os.listdir(sub_dir)):
    if not filename.endswith(".py") or filename == "__init__.py":
        continue
    module_name = f"{__name__}.{filename[:-3]}"
    try:
        module = importlib.import_module(module_name)
    except Exception as ex:
        warnings.warn(
            f"ssapy_toolkit/plots/__init__.py: failed to import "
            f"{filename} — skipping it, everything else still loads. "
            f"Error: {ex}"
        )
        continue
    # Add all public attributes of the module to the package namespace
    for attr in dir(module):
        if not attr.startswith("_"):  # Avoid internal attributes
            globals()[attr] = getattr(module, attr)

del importlib, os, warnings, sub_dir
try:
    del filename, module_name, module, attr
except NameError:
    pass