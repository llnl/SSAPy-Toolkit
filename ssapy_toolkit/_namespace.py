"""Helpers for package namespace compatibility imports."""

from __future__ import annotations

import importlib
import inspect
import os
import warnings
from collections.abc import MutableMapping
from pathlib import Path

# Every failure seen this interpreter session, keyed "package.module".
# Aggregated across packages so one call surfaces the whole picture rather
# than whichever package happened to be imported first.
_FAILURES: dict[str, BaseException] = {}

_STRICT_ENV = "SSAPY_TOOLKIT_STRICT_IMPORTS"

_IMPORTED_PUBLIC_NAME_DENYLIST = {
    "Any",
    "Callable",
    "Iterable",
    "Path",
    "Sequence",
    "Time",
    "annotations",
    "dataclass",
    "datetime",
    "field",
    "np",
    "os",
    "plt",
    "sys",
    "warnings",
}


def import_failures() -> dict[str, BaseException]:
    """Modules that failed to import, keyed 'package.module'."""
    return dict(_FAILURES)


def _format(failures: dict[str, BaseException]) -> str:
    lines = []
    for name, exc in sorted(failures.items()):
        detail = str(exc).strip().splitlines()
        first = detail[0] if detail else ""
        lines.append(f"  {name}: {type(exc).__name__}: {first}")
    return "\n".join(lines)


def _default_public_names(module) -> list[str]:
    """Return public names defined by ``module`` when it does not declare ``__all__``."""
    names = []
    for attr in dir(module):
        if attr.startswith("_") or attr in _IMPORTED_PUBLIC_NAME_DENYLIST:
            continue

        value = getattr(module, attr)
        if inspect.ismodule(value):
            continue
        if (inspect.isfunction(value) or inspect.isclass(value)) and value.__module__ != module.__name__:
            continue
        names.append(attr)
    return names


def import_public_modules(
    package_name: str,
    package_file: str,
    namespace: MutableMapping[str, object],
    *,
    skip: set[str] | None = None,
    exclude_prefixes: tuple[str, ...] = (),
    strict: bool | None = None,
) -> dict[str, BaseException]:
    """Import sibling modules and copy their public attributes into a package namespace.

    Only names a module actually defines are re-exported: ``__all__`` when it
    declares one, otherwise everything public that is not a re-imported
    module, function, or class. Without that filter every ``import numpy as
    np`` leaks ``np`` into the package namespace, and modules silently
    overwrite one another's names.

    A module that fails to import is recorded and skipped rather than
    aborting the package. Without that, a single broken module makes every
    other module in the package unimportable, and because the loop stops at
    the first failure it also hides however many more are behind it.

    Failures are never silent: they are collected, warned about once with the
    full list, exposed on the package as ``__import_failures__`` and through
    :func:`import_failures`. Set ``strict=True`` or the
    ``SSAPY_TOOLKIT_STRICT_IMPORTS=1`` environment variable to re-raise
    instead, which is what CI should do so a clean-import guarantee still
    means something.

    Returns the failures for this package, empty when all imported.
    """
    if strict is None:
        strict = os.environ.get(_STRICT_ENV, "") not in ("", "0")

    skip = set() if skip is None else skip
    package_dir = Path(package_file).resolve().parent
    exported_names = {
        name for name in namespace if not name.startswith("_") and name != "import_public_modules"
    }
    failures: dict[str, BaseException] = {}

    for path in sorted(package_dir.glob("*.py")):
        if (path.name == "__init__.py" or path.stem.startswith("_")
                or path.stem in skip
                or (exclude_prefixes and path.stem.startswith(exclude_prefixes))):
            continue

        qualified = f"{package_name}.{path.stem}"
        try:
            module = importlib.import_module(qualified)
        except Exception as exc:
            # Deliberately Exception, not ImportError: a module can also die
            # on a SyntaxError, a missing data file, or anything else run at
            # import time, and those should not take the package down either.
            # BaseException is excluded so KeyboardInterrupt still propagates.
            failures[qualified] = exc
            _FAILURES[qualified] = exc
            if strict:
                raise
            continue

        public_names = getattr(module, "__all__", None)
        if public_names is None:
            public_names = _default_public_names(module)
        for attr in public_names:
            namespace[attr] = getattr(module, attr)
            exported_names.add(attr)

    namespace.pop("import_public_modules", None)
    namespace["__all__"] = sorted(name for name in exported_names if name in namespace)
    namespace["__import_failures__"] = failures
    if failures:
        warnings.warn(
            f"{package_name}: {len(failures)} module(s) failed to import and "
            f"were skipped:\n{_format(failures)}\n"
            f"Set {_STRICT_ENV}=1 to raise instead.",
            ImportWarning,
            stacklevel=2,
        )
    return failures