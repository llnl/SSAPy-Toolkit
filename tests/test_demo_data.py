from pathlib import Path
from urllib.error import URLError

import pytest

import ssapy_toolkit.io.demo_data as demo_data


def _patch_demo_datapath(monkeypatch, cache_dir):
    def fake_datapath(filename):
        target = Path(cache_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(target)

    monkeypatch.setattr(demo_data, "datapath", fake_datapath)


def test_ensure_demo_data_file_returns_cached_datapath_file(tmp_path, monkeypatch):
    _patch_demo_datapath(monkeypatch, tmp_path / "cache")
    cached = tmp_path / "cache" / "full_catalog_3le.txt"
    cached.parent.mkdir(parents=True)
    cached.write_text("ISS\n1 00000\n2 00000\n", encoding="utf-8")

    path = demo_data.ensure_demo_data_file("full_catalog_3le.txt", allow_download=False)

    assert path == cached


def test_ensure_demo_data_file_finds_explicit_local_data_dir(tmp_path, monkeypatch):
    _patch_demo_datapath(monkeypatch, tmp_path / "empty_cache")
    local_dir = tmp_path / "local_ssatk_data"
    local_file = local_dir / "artemis2_orion_state_vectors.csv"
    local_file.parent.mkdir(parents=True)
    local_file.write_text("JDTDB,Calendar_Date_TDB\n", encoding="utf-8")

    path = demo_data.ensure_demo_data_file(
        "artemis2_orion_state_vectors.csv",
        allow_download=False,
        local_dirs=[local_dir],
    )

    assert path == local_file


def test_ensure_demo_data_file_warns_and_returns_none_when_offline(tmp_path, monkeypatch):
    _patch_demo_datapath(monkeypatch, tmp_path / "cache")

    with pytest.warns(demo_data.DemoDataUnavailableWarning, match="auto-download disabled"):
        path = demo_data.ensure_demo_data_file("missing_demo_file.txt", allow_download=False)

    assert path is None


def test_ensure_demo_data_file_uses_configured_fetcher(tmp_path, monkeypatch):
    _patch_demo_datapath(monkeypatch, tmp_path / "cache")

    def fake_fetcher(target, *, timeout):
        assert timeout == 12
        Path(target).write_text("downloaded\n", encoding="utf-8")

    monkeypatch.setitem(demo_data._DEMO_DATA_FETCHERS, "fetched.txt", fake_fetcher)

    path = demo_data.ensure_demo_data_file("fetched.txt", timeout=12)

    assert path == tmp_path / "cache" / "fetched.txt"
    assert path.read_text(encoding="utf-8") == "downloaded\n"


def test_ensure_demo_data_file_warns_and_cleans_up_failed_fetch(tmp_path, monkeypatch):
    _patch_demo_datapath(monkeypatch, tmp_path / "cache")

    def failing_fetcher(target, *, timeout):
        Path(target).write_text("partial\n", encoding="utf-8")
        raise URLError("network unavailable")

    monkeypatch.setitem(demo_data._DEMO_DATA_FETCHERS, "broken.txt", failing_fetcher)

    with pytest.warns(demo_data.DemoDataUnavailableWarning, match="download failed"):
        path = demo_data.ensure_demo_data_file("broken.txt")

    assert path is None
    assert not (tmp_path / "cache" / "broken.txt").exists()


def test_horizons_vector_rows_extracts_csv_rows():
    text = "header\n$$SOE\n2461132.5,A.D. 2026-Apr-02 00:00:00.0000,1,2,3,4,5,6,\n$$EOE\nfooter"

    rows = demo_data._horizons_vector_rows(text)

    assert rows == ["2461132.5,A.D. 2026-Apr-02 00:00:00.0000,1,2,3,4,5,6"]


def test_demo_parsing_3le_skips_when_optional_data_unavailable(monkeypatch):
    from demos import demo_parsing_3le

    monkeypatch.setattr(demo_parsing_3le, "ensure_demo_data_file", lambda *args, **kwargs: None)

    out = demo_parsing_3le.main(verbose=False, fast=True, allow_download=False)

    assert out["skipped"] is True
    assert out["reason"] == "missing_data_file"
    assert out["tle_path"] is None


def test_demo_artemis_benchmark_skips_when_optional_data_unavailable(monkeypatch):
    from demos import demo_artemis_benchmark

    monkeypatch.setattr(demo_artemis_benchmark, "_find_csv", lambda allow_download=True: None)

    out = demo_artemis_benchmark.main(make_figures=False, fast=True, verbose=False, allow_download=False)

    assert out["skipped"] is True
    assert out["reason"] == "missing_data_file"
    assert out["csv_path"] is None
