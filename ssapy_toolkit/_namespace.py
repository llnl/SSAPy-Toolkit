"""Helpers for package namespace compatibility imports."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import MutableMapping


def import_public_modules(package_name: str, package_file: str, namespace: MutableMapping[str, object]) -> None:
    """Import sibling modules and copy their public attributes into a package namespace."""
    package_dir = Path(package_file).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue

        module = importlib.import_module(f"{package_name}.{path.stem}")
        for attr in dir(module):
            if not attr.startswith("_"):
                namespace[attr] = getattr(module, attr)
