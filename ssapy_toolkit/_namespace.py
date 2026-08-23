"""Helpers for package namespace compatibility imports."""

from __future__ import annotations

import importlib
from collections.abc import MutableMapping
from pathlib import Path


def import_public_modules(
    package_name: str,
    package_file: str,
    namespace: MutableMapping[str, object],
    *,
    skip: set[str] | None = None,
) -> None:
    """Import sibling modules and copy their public attributes into a package namespace."""
    skip = set() if skip is None else skip
    package_dir = Path(package_file).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        if path.name == "__init__.py" or path.stem in skip:
            continue

        module = importlib.import_module(f"{package_name}.{path.stem}")
        for attr in dir(module):
            if not attr.startswith("_"):
                namespace[attr] = getattr(module, attr)
