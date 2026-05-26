"""Acceleration models and utilities for orbit propagation."""

import importlib
import os

# Get the directory of the current file
sub_dir = os.path.dirname(__file__)

# Loop through all Python files in this directory and import their functions
for filename in os.listdir(sub_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = f"{__name__}.{filename[:-3]}"
        module = importlib.import_module(module_name)

        # Add all attributes of the module to the package namespace
        for attr in dir(module):
            if not attr.startswith("_"):  # Avoid internal attributes
                globals()[attr] = getattr(module, attr)
