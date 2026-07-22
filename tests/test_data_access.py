from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from ssapy_toolkit.data import (
    DataPackageNotFoundError,
    DataResourceNotFoundError,
    data_package_available,
    data_path,
    data_resource,
    read_data_binary,
    read_data_text,
)


def _make_data_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str = "fake_ssapy_data") -> str:
    package_root = tmp_path / package
    data_root = package_root / "data" / "catalogs"
    data_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text('"""Fake data package for tests."""\n', encoding="utf-8")
    (data_root / "sample.txt").write_text("object_id,value\nA,1\n", encoding="utf-8")
    (data_root / "sample.bin").write_bytes(b"\x00\x01ssapy")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop(package, None)
    return package


def test_data_resource_reads_from_installed_package(tmp_path, monkeypatch):
    package = _make_data_package(tmp_path, monkeypatch)

    resource = data_resource("catalogs/sample.txt", package=package)

    assert data_package_available(package)
    assert resource.is_file()
    assert resource.read_text(encoding="utf-8") == "object_id,value\nA,1\n"


def test_data_path_yields_filesystem_path(tmp_path, monkeypatch):
    package = _make_data_package(tmp_path, monkeypatch)

    with data_path("catalogs/sample.txt", package=package) as path:
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "object_id,value\nA,1\n"


def test_read_helpers_return_text_and_binary(tmp_path, monkeypatch):
    package = _make_data_package(tmp_path, monkeypatch)

    assert read_data_text("catalogs/sample.txt", package=package) == "object_id,value\nA,1\n"
    assert read_data_binary("catalogs/sample.bin", package=package) == b"\x00\x01ssapy"


def test_missing_package_error_is_actionable():
    package = "definitely_missing_ssapy_data_package"

    assert not data_package_available(package)
    with pytest.raises(DataPackageNotFoundError, match="Data package .* is not installed"):
        data_resource("catalogs/sample.txt", package=package)


def test_missing_resource_error_names_requested_path(tmp_path, monkeypatch):
    package = _make_data_package(tmp_path, monkeypatch)

    with pytest.raises(DataResourceNotFoundError, match="data/catalogs/missing.txt"):
        data_resource("catalogs/missing.txt", package=package)


def test_absolute_and_parent_paths_are_rejected(tmp_path, monkeypatch):
    package = _make_data_package(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="must be relative"):
        data_resource("/catalogs/sample.txt", package=package)
    with pytest.raises(ValueError, match="cannot contain"):
        data_resource("catalogs/../sample.txt", package=package)
