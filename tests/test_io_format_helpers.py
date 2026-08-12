import json
import warnings
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning

from ssapy_toolkit.io import converter_json_hdf5, pprint_utils, workspace_io, xml_utils


def test_converter_json_hdf5_roundtrip_files_and_edge_keys(tmp_path):
    payload = {
        "plain/key": [1, True, None, "text"],
        "nested space": {"float": 1.5, "weird%key": {"a": 1}},
        "fallback": {"tuple": (1, 2)},
    }
    h5_path = tmp_path / "nested" / "payload.h5"
    json_path = tmp_path / "payload.json"
    out_json = tmp_path / "converted" / "out.json"

    assert converter_json_hdf5._percent_decode_name(converter_json_hdf5._percent_encode_name("a/b % c")) == "a/b % c"
    assert converter_json_hdf5._percent_decode_name("literal%nothex") == "literal%nothex"
    converter_json_hdf5.json_to_hdf5(payload, h5_path, root="root")
    loaded = converter_json_hdf5.hdf5_to_json(h5_path, root="root")
    assert loaded["plain/key"] == [1, True, None, "text"]
    assert loaded["nested space"]["weird%key"] == {"a": 1}

    json_path.write_text(json.dumps(payload), encoding="utf-8")
    converter_json_hdf5.json_file_to_hdf5(json_path, h5_path, root="data")
    converter_json_hdf5.hdf5_file_to_json(h5_path, out_json, root="data", pretty=False)
    assert json.loads(out_json.read_text(encoding="utf-8"))["plain/key"][0] == 1

    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("a", data=1)
        handle.create_dataset("b", data=2)
    with pytest.raises(KeyError):
        converter_json_hdf5.hdf5_to_json(h5_path, root="missing")


def test_xml_utils_roundtrip_special_types_and_raw_structures(tmp_path):
    path = tmp_path / "nested" / "payload.xml"
    when = datetime(2026, 1, 1, 12, 0, 0)
    time = Time("2026-01-01T12:00:00", scale="utc")
    payload = {
        "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "when": when,
        "time": time,
        "set": {"a", "b"},
        "tuple": (1, 2),
        "items": [{"name": "one"}, {"name": "two"}],
        "bytes": b"abc",
    }

    xml_utils.save_xml(path, {"payload": payload}, pretty=True)
    loaded = xml_utils.read_xml(path)
    np.testing.assert_array_equal(loaded["array"], payload["array"])
    assert loaded["when"] == when
    assert loaded["time"].isot == time.isot
    assert loaded["set"] == {"a", "b"}
    assert loaded["tuple"] == ("1", "2")
    assert loaded["items"][0]["name"] == "one"
    assert loaded["bytes"] == "abc"

    kept = xml_utils.load_xml(path, keep_root=True, decode_special=False)
    assert "payload" in kept
    assert kept["payload"]["array"]["@attrs"]["type"] == "ndarray"

    bad_shape = {"@attrs": {"type": "ndarray", "dtype": "int64", "shape": "3,3"}, "item": [1, 2]}
    flat = xml_utils._decode_special_struct(bad_shape)
    np.testing.assert_array_equal(flat, [1, 2])
    assert xml_utils._decode_special_struct({"@attrs": {"type": "datetime"}, "#text": "bad"}) == "bad"
    assert xml_utils._decode_special_struct({"@attrs": {"type": "astropy_time", "scale": "utc"}, "#text": "bad"}) == "bad"
    assert xml_utils._decode_special_struct({"@attrs": {"type": "unknown"}, "#text": "x"})["#text"] == "x"


def test_pprint_utils_hdf5_and_mapping_summaries(tmp_path, capsys):
    path = tmp_path / "summary.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["creator"] = "test"
        group = handle.create_group("group")
        group.attrs["kind"] = "demo"
        group.create_dataset("scalar", data=7)
        group.create_dataset("small", data=np.arange(4))
        group.create_dataset("large", data=np.arange(30).reshape(10, 3))

    with h5py.File(path, "r") as handle:
        obj, must_close = pprint_utils._open_hdf5_like(handle["group"])
        assert obj.name == "/group"
        assert not must_close
        assert "1 groups" not in pprint_utils._summarize_group(handle["group"])
        assert "preview" in pprint_utils._summarize_dataset(handle["group/small"])
        assert pprint_utils._summarize_attrs(handle) == "@attrs: ['creator']"
        assert pprint_utils._indent_for_name("/group/small") == "  "
        assert "..." in pprint_utils._preview_dataset_values(handle["group/large"], head=2, tail=2, small_limit=3)
        pprint_utils.pprint(handle["group"])
    assert "/group" in capsys.readouterr().out

    opened, must_close = pprint_utils._open_hdf5_like(path)
    assert must_close
    opened.close()
    with pytest.raises(TypeError):
        pprint_utils._open_hdf5_like(object())

    cyclic = {}
    cyclic["self"] = cyclic
    many = {f"k{i}": i for i in range(25)}
    pprint_utils.pprint({"arr": np.arange(25), "seq": list(range(8)), "bytes": b"abc", "none": None, "cycle": cyclic, "many": many})
    out = capsys.readouterr().out
    assert "dict" in out
    assert "preview" in pprint_utils._array_preview(np.arange(25))
    assert pprint_utils._clip("abcdef", 4) == "a..."
    assert pprint_utils._short_value(object()) == "object"
    pprint_utils.pprint(np.arange(3))
    assert "array" in capsys.readouterr().out


def test_pprint_utils_fallback_and_error_branches(tmp_path, capsys, monkeypatch):
    path = tmp_path / "summary_extra.h5"
    with h5py.File(path, "w") as handle:
        root_child = handle.create_group("child")
        root_child.attrs["kind"] = "branch"
        nested = root_child.create_group("nested")
        nested.attrs["note"] = "subgroup"
        ds = nested.create_dataset("values", data=np.arange(12).reshape(6, 2))
        ds.attrs["units"] = "km"
        scalar = handle.create_dataset("scalar", data=1)
        scalar.attrs["kind"] = "single"

    assert pprint_utils._indent_for_name("/") == ""
    with h5py.File(path, "r") as handle:
        assert "1 groups" in pprint_utils._summarize_group(handle["child"])
        assert pprint_utils._preview_dataset_values(handle["child/nested/values"], head=2, tail=0, small_limit=1).startswith("0")
        pprint_utils.pprint(handle["scalar"])
        pprint_utils._print_hdf5_summary(handle)
    out = capsys.readouterr().out
    assert "/scalar" in out
    assert "@attrs" in out

    class BadShapeDataset:
        name = "/bad"
        dtype = "float64"

        @property
        def shape(self):
            raise RuntimeError("no shape")

    class BadDtypeDataset:
        name = "/bad_dtype"
        shape = (1,)

        @property
        def dtype(self):
            raise RuntimeError("no dtype")

        def __getitem__(self, key):
            raise RuntimeError("no data")

    class BadAttrs:
        @property
        def attrs(self):
            raise RuntimeError("no attrs")

    class LenlessSequence:
        def __len__(self):
            raise RuntimeError("no len")

    class LenlessMapping(dict):
        def __len__(self):
            raise RuntimeError("no len")

    class KeylessMapping(dict):
        def keys(self):
            raise RuntimeError("no keys")

    class BadSortKey:
        def __str__(self):
            raise RuntimeError("no string")

    assert "shape=unknown" in pprint_utils._summarize_dataset(BadShapeDataset())
    assert "dtype=unknown" in pprint_utils._summarize_dataset(BadDtypeDataset())
    assert "preview unavailable" in pprint_utils._preview_dataset_values(BadDtypeDataset())
    assert pprint_utils._summarize_attrs(BadAttrs()) == ""
    assert pprint_utils._clip("abcdef", 3) == "abc"
    assert "scalar" in pprint_utils._array_preview(np.array(3))
    assert "values" in pprint_utils._array_preview(np.array([1, 2]))
    assert pprint_utils._short_value("abcdef") == '"abcdef"'
    assert pprint_utils._short_value(LenlessMapping()) == "dict(? keys)"
    assert pprint_utils._seq_preview(LenlessSequence()) == "LenlessSequence"

    pprint_utils._print_dict_summary(LenlessMapping())
    pprint_utils._print_dict_summary({"stop": {"here": 1}}, depth=4)
    pprint_utils._print_dict_summary(KeylessMapping())
    monkeypatch.setitem(pprint_utils._dict_defaults, "key_sort", True)
    monkeypatch.setitem(pprint_utils._dict_defaults, "max_items", 0)
    pprint_utils._print_dict_summary({BadSortKey(): "value"})
    assert "dict" in capsys.readouterr().out

    pprint_utils.pprint(tmp_path / "not_hdf5.txt")
    assert "not_hdf5" in capsys.readouterr().out


def test_pprint_utils_h5py_optional_error(monkeypatch):
    monkeypatch.setattr(pprint_utils, "h5py", None)
    with pytest.raises(RuntimeError, match="h5py not installed"):
        pprint_utils._open_hdf5_like("anything.h5")


def test_workspace_save_and_load_roundtrip(tmp_path, capsys):
    path = tmp_path / "nested" / "workspace.json"

    def save_it():
        scalar = 42
        array = np.array([1, 2, 3])
        frame = pd.DataFrame({"a": [1, 2]})
        series = pd.Series([3, 4], name="s")
        when = datetime(2026, 1, 1)
        aset = {"x", "y"}
        skipped = object()
        return workspace_io.save_workspace(path, exclude=["skipped"])

    result = save_it()
    assert result["__metadata__"]["num_variables"] >= 5
    assert "Workspace saved" in capsys.readouterr().out

    loaded = workspace_io.load_workspace(path, into_globals=False)
    assert loaded["scalar"] == 42
    np.testing.assert_array_equal(loaded["array"], [1, 2, 3])
    assert loaded["frame"].shape == (2, 1)
    assert loaded["series"].name == "s"
    assert loaded["when"] == datetime(2026, 1, 1)
    assert loaded["aset"] == {"x", "y"}
    assert "Workspace loaded" in capsys.readouterr().out


def test_workspace_astropy_types_and_global_injection(tmp_path):
    path = tmp_path / "workspace_astropy.json"

    class FakeRepresentation:
        name = "fake-representation"

    assert workspace_io._representation_type_name(FakeRepresentation) == "fake-representation"
    assert workspace_io._representation_type_name("fallback") == "fallback"

    def save_it():
        distance = 5.0 * u.km
        time = Time("2026-01-01T00:00:00", format="isot", scale="utc")
        coord = SkyCoord(ra=10.0 * u.deg, dec=-5.0 * u.deg, frame="icrs")
        table = Table({"name": ["a", "b"], "mag": [1.0, 2.0]}, meta={"kind": "demo"})
        unsupported = object()
        return workspace_io.save_workspace(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        saved = save_it()

    assert not [warning for warning in caught if issubclass(warning.category, AstropyDeprecationWarning)]
    assert "unsupported" in saved["__metadata__"]["skipped_variables"]

    loaded = workspace_io.load_workspace(path, into_globals=False)
    assert loaded["distance"].unit == u.km
    assert loaded["time"].isot.startswith("2026-01-01")
    assert loaded["coord"].frame.name == "icrs"
    assert loaded["table"].meta["kind"] == "demo"

    workspace_io.load_workspace(path, into_globals=True)
    assert "distance" in globals()
    for name in ("distance", "time", "coord", "table"):
        globals().pop(name, None)
