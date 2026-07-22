"""Access data packaged outside SSAPy Toolkit.

SSAPy Toolkit keeps source code separate from bulky datasets and generated
media. Reusable datasets should live in a dedicated data package, currently
expected to expose files below ``ssapy_data/data``. This module provides a small
``importlib.resources`` wrapper so toolkit functions can read those files from a
normal wheel install without Git LFS, git submodules, or runtime GitHub pulls.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.util import find_spec
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Iterator

DEFAULT_DATA_PACKAGE = "ssapy_data"
DEFAULT_DATA_ROOT = "data"


class DataPackageNotFoundError(ModuleNotFoundError):
    """Raised when the configured SSAPy data package is not installed."""


class DataResourceNotFoundError(FileNotFoundError):
    """Raised when a requested resource is absent from the data package."""


def data_package_available(package: str = DEFAULT_DATA_PACKAGE) -> bool:
    """Return ``True`` when the named data package can be imported."""

    return find_spec(package) is not None


def data_resource(
    relative_path: str | PathLike[str] = "",
    *,
    package: str = DEFAULT_DATA_PACKAGE,
    data_root: str | PathLike[str] = DEFAULT_DATA_ROOT,
    must_exist: bool = True,
) -> Traversable:
    """Return an ``importlib.resources`` object for packaged data.

    Parameters
    ----------
    relative_path
        POSIX-style path below ``data_root`` inside the data package. Absolute
        paths and ``..`` traversal are rejected.
    package
        Import package that owns the data resources. Toolkit code should use the
        default ``ssapy_data`` package once SSAPy-Data is published as a wheel.
    data_root
        Directory inside ``package`` that contains data resources.
    must_exist
        If ``True``, raise :class:`DataResourceNotFoundError` when the resource
        is missing.
    """

    resource = _package_root(package)

    for part in _safe_parts(data_root):
        resource = resource.joinpath(part)
    for part in _safe_parts(relative_path):
        resource = resource.joinpath(part)

    if must_exist and not resource.exists():
        requested = _display_path(data_root, relative_path)
        raise DataResourceNotFoundError(
            f"Data resource '{requested}' was not found in package '{package}'."
        )

    return resource


@contextmanager
def data_path(
    relative_path: str | PathLike[str],
    *,
    package: str = DEFAULT_DATA_PACKAGE,
    data_root: str | PathLike[str] = DEFAULT_DATA_ROOT,
) -> Iterator[Path]:
    """Yield a filesystem path for a packaged data file.

    Use this when a downstream library requires a real path instead of a file
    object. The yielded path may be a temporary extraction path for zipped wheels,
    so callers should use it only inside the context manager.
    """

    resource = data_resource(relative_path, package=package, data_root=data_root)
    if not resource.is_file():
        requested = _display_path(data_root, relative_path)
        raise DataResourceNotFoundError(
            f"Data resource '{requested}' in package '{package}' is not a file."
        )

    with as_file(resource) as path:
        yield path


@contextmanager
def open_data(
    relative_path: str | PathLike[str],
    mode: str = "rb",
    *,
    package: str = DEFAULT_DATA_PACKAGE,
    data_root: str | PathLike[str] = DEFAULT_DATA_ROOT,
    encoding: str | None = None,
):
    """Open a packaged data file.

    Parameters mirror :meth:`importlib.resources.abc.Traversable.open`.
    """

    resource = data_resource(relative_path, package=package, data_root=data_root)
    if not resource.is_file():
        requested = _display_path(data_root, relative_path)
        raise DataResourceNotFoundError(
            f"Data resource '{requested}' in package '{package}' is not a file."
        )

    with resource.open(mode, encoding=encoding) as file_handle:
        yield file_handle


def read_data_text(
    relative_path: str | PathLike[str],
    *,
    package: str = DEFAULT_DATA_PACKAGE,
    data_root: str | PathLike[str] = DEFAULT_DATA_ROOT,
    encoding: str = "utf-8",
) -> str:
    """Read a packaged text data file."""

    with open_data(
        relative_path,
        "r",
        package=package,
        data_root=data_root,
        encoding=encoding,
    ) as file_handle:
        return file_handle.read()


def read_data_binary(
    relative_path: str | PathLike[str],
    *,
    package: str = DEFAULT_DATA_PACKAGE,
    data_root: str | PathLike[str] = DEFAULT_DATA_ROOT,
) -> bytes:
    """Read a packaged binary data file."""

    with open_data(relative_path, "rb", package=package, data_root=data_root) as file_handle:
        return file_handle.read()


def _package_root(package: str) -> Traversable:
    try:
        return files(package)
    except ModuleNotFoundError as exc:
        if exc.name != package:
            raise
        raise DataPackageNotFoundError(
            f"Data package '{package}' is not installed. Install the SSAPy data "
            "package that provides the required resource, then retry."
        ) from exc


def _safe_parts(path: str | PathLike[str]) -> tuple[str, ...]:
    path_string = str(path)
    if path_string in {"", "."}:
        return ()

    pure_path = PurePosixPath(path_string)
    if pure_path.is_absolute():
        raise ValueError(f"Packaged data paths must be relative, got '{path_string}'.")

    parts = tuple(part for part in pure_path.parts if part not in {"", "."})
    if ".." in parts:
        raise ValueError(f"Packaged data paths cannot contain '..', got '{path_string}'.")
    return parts


def _display_path(data_root: str | PathLike[str], relative_path: str | PathLike[str]) -> str:
    return "/".join((*_safe_parts(data_root), *_safe_parts(relative_path)))
