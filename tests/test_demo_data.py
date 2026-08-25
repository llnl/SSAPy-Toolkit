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
    local_dir = tmp_path / "local_ssatk_output"
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


def test_ensure_demo_data_file_no_fetcher_empty_fetch_and_unlink_failure(tmp_path, monkeypatch):
    _patch_demo_datapath(monkeypatch, tmp_path / "cache")

    with pytest.warns(demo_data.DemoDataUnavailableWarning, match="no fetcher"):
        assert demo_data.ensure_demo_data_file("unknown.bin") is None

    def empty_fetcher(target, *, timeout):
        Path(target).touch()

    monkeypatch.setitem(demo_data._DEMO_DATA_FETCHERS, "empty.txt", empty_fetcher)
    with pytest.warns(demo_data.DemoDataUnavailableWarning, match="produced no file"):
        assert demo_data.ensure_demo_data_file("empty.txt") is None

    def failing_fetcher(target, *, timeout):
        Path(target).write_text("partial\n", encoding="utf-8")
        raise OSError("disk issue")

    def blocked_unlink(self, missing_ok=False):
        raise OSError("permission denied")

    monkeypatch.setitem(demo_data._DEMO_DATA_FETCHERS, "unlink-blocked.txt", failing_fetcher)
    monkeypatch.setattr(demo_data.Path, "unlink", blocked_unlink)
    with pytest.warns(demo_data.DemoDataUnavailableWarning, match="download failed"):
        assert demo_data.ensure_demo_data_file("unlink-blocked.txt") is None


def test_horizons_vector_rows_extracts_csv_rows():
    text = "header\n$$SOE\n2461132.5,A.D. 2026-Apr-02 00:00:00.0000,1,2,3,4,5,6,\n$$EOE\nfooter"

    rows = demo_data._horizons_vector_rows(text)

    assert rows == ["2461132.5,A.D. 2026-Apr-02 00:00:00.0000,1,2,3,4,5,6"]


def test_demo_data_fetchers_validate_and_write_without_network(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, *, timeout):
        calls.append((url, timeout))
        if "stations" in url:
            return "ISS\n1 00000\n2 00000\n"
        if "FORMAT=xml" in url:
            return "<ndm><omm /></ndm>"
        if "horizons" in url:
            return "header\n$$SOE\n2461132.5,A.D. 2026-Apr-02 00:00:00.0000,1,2,3,4,5,6,extra\n$$EOE\n"
        return "not tle"

    monkeypatch.setattr(demo_data, "_download_text", fake_download)

    tle_path = tmp_path / "tle.txt"
    demo_data._fetch_full_catalog_3le(tle_path, timeout=4)
    assert tle_path.read_text(encoding="utf-8").startswith("ISS")
    assert len(calls) == 2

    xml_path = tmp_path / "catalog.xml"
    demo_data._fetch_full_catalog_xml(xml_path, timeout=5)
    assert "<ndm" in xml_path.read_text(encoding="utf-8")

    horizons_path = tmp_path / "horizons.csv"
    demo_data._fetch_artemis2_orion_state_vectors(horizons_path, timeout=6)
    contents = horizons_path.read_text(encoding="utf-8")
    assert "JDTDB,Calendar_Date_TDB" in contents
    assert "2461132.5" in contents

    monkeypatch.setattr(demo_data, "_download_text", lambda *args, **kwargs: "invalid")
    with pytest.raises(URLError):
        demo_data._fetch_full_catalog_3le(tmp_path / "bad_tle.txt", timeout=1)
    with pytest.raises(ValueError, match="OMM XML"):
        demo_data._fetch_full_catalog_xml(tmp_path / "bad.xml", timeout=1)
    with pytest.raises(ValueError, match="vector rows"):
        demo_data._fetch_artemis2_orion_state_vectors(tmp_path / "bad.csv", timeout=1)


def test_download_text_and_local_candidate_deduplication(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return "payload".encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["agent"] = request.headers["User-agent"]
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(demo_data, "urlopen", fake_urlopen)
    assert demo_data._download_text("https://example.test/data", timeout=7) == "payload"
    assert captured == {"agent": demo_data._USER_AGENT, "timeout": 7}

    cache = tmp_path / demo_data.DEFAULT_OUTPUT_DIR_NAME
    _patch_demo_datapath(monkeypatch, cache)
    candidates = list(demo_data._local_data_candidates("demo.txt", local_dirs=[cache, tmp_path]))
    assert all(path != cache / "demo.txt" for path in candidates)
    assert len({path.resolve(strict=False) for path in candidates}) == len(candidates)

    module_path = tmp_path / "workdir" / "SSAPy-Toolkit" / "ssapy_toolkit" / "io" / "demo_data.py"
    monkeypatch.setattr(demo_data, "__file__", str(module_path))
    assert any(path.name == "demo.txt" for path in demo_data._local_data_candidates("demo.txt"))


def test_demo_parsing_3le_skips_when_optional_data_unavailable(monkeypatch):
    from demos.getting_started import demo_parsing_3le

    monkeypatch.setattr(demo_parsing_3le, "ensure_demo_data_file", lambda *args, **kwargs: None)

    out = demo_parsing_3le.main(verbose=False, fast=True, allow_download=False)

    assert out["skipped"] is True
    assert out["reason"] == "missing_data_file"
    assert out["tle_path"] is None


def test_demo_artemis_benchmark_skips_when_optional_data_unavailable(monkeypatch):
    from demos.benchmarks import demo_artemis_benchmark

    monkeypatch.setattr(demo_artemis_benchmark, "_find_csv", lambda allow_download=True: None)

    out = demo_artemis_benchmark.main(make_figures=False, fast=True, verbose=False, allow_download=False)

    assert out["skipped"] is True
    assert out["reason"] == "missing_data_file"
    assert out["csv_path"] is None


def test_demo_orekit_benchmark_skips_without_external_runtime(monkeypatch):
    from demos.benchmarks import demo_orekit_benchmark

    monkeypatch.setattr(demo_orekit_benchmark, "_run_orekit", lambda **kwargs: None)
    out = demo_orekit_benchmark.main(make_figures=False, fast=True, verbose=False, allow_install=False)

    assert out["skipped"] is True
    assert "unavailable" in out["reason"]
