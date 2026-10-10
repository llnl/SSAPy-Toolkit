"""Deprecated import name for the Space Situational Awareness Toolkit (SSATK).

SSATK was imported as ``ssapy_toolkit`` through 1.0.7; since 1.1.0 the package
is ``ssatk``. This shim keeps old code working: ``ssapy_toolkit`` and every
``ssapy_toolkit.<submodule>`` forward attribute access, assignment and
``dir()`` to the matching ``ssatk`` module, so the functions, classes and
constants reached through either name are the same objects (``isinstance``
checks and pickles that name the old path keep working).

Importing it emits one ``FutureWarning``. Replace ``ssapy_toolkit`` with
``ssatk`` in imports; the shim will be removed in a future major release.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import types
import warnings

_OLD = __name__
_NEW = "ssatk"

warnings.warn(
    "'ssapy_toolkit' was renamed to 'ssatk' in 1.1.0; import ssatk instead. "
    "The ssapy_toolkit alias will be removed in a future major release.",
    FutureWarning,
    stacklevel=2,
)


class _AliasModule(types.ModuleType):
    """``ssapy_toolkit.<x>``: forwards to ``ssatk.<x>``.

    A proxy, not the real module, because the import system assigns each
    freshly imported child module onto its parent. Through the real
    ``ssatk.plots`` that would replace re-exported functions such as
    ``ssatk.plots.globe_plot`` with the submodule of the same name.
    """

    def __init__(self, name, real):
        super().__init__(name, real.__doc__)
        object.__setattr__(self, "_ssatk_real", real)
        if hasattr(real, "__path__"):
            object.__setattr__(self, "__path__", list(real.__path__))

    def __getattr__(self, attr):
        return getattr(object.__getattribute__(self, "_ssatk_real"), attr)

    def __setattr__(self, attr, value):
        if attr.startswith("__") and attr.endswith("__"):
            object.__setattr__(self, attr, value)   # import-system bookkeeping
        elif isinstance(value, _AliasModule):
            pass                                     # child alias; the real parent already has it
        else:
            setattr(object.__getattribute__(self, "_ssatk_real"), attr, value)

    def __dir__(self):
        return dir(object.__getattribute__(self, "_ssatk_real"))


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, real_name):
        self._real_name = real_name

    def create_module(self, spec):
        return _AliasModule(spec.name, importlib.import_module(self._real_name))

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        real_name = _NEW + fullname[len(_OLD):]
        try:
            found = importlib.util.find_spec(real_name)
        except ModuleNotFoundError:
            found = None
        if found is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(real_name), is_package=found.submodule_search_locations is not None
        )


if not any(type(finder).__name__ == "_AliasFinder" for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

_real = importlib.import_module(_NEW)


def __getattr__(attr):
    return getattr(_real, attr)


def __dir__():
    return dir(_real)
