import csv
import importlib
import json
import types
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

from ssapy_toolkit.io import csv_utils, dict_to_from_hdf5, get_memory, hdf5_to_csv, hdf5_utils, io_utils, json_utils, pickle_utils
from ssapy_toolkit.io.guess_delimiter import guess_csv_delimiter

h5cache_module = importlib.import_module("ssapy_toolkit.io.h5cache")


def test_json_roundtrip_and_append_special_types(tmp_path):
    path = tmp_path / "nested" / "data.json"
    when = datetime(2026, 1, 2, 3, 4, 5)
    time = Time("2026-01-02T03:04:05", scale="utc")
    payload = {
        "array": np.array([[1, 2], [3, 4]]),
        "scalar": np.float64(1.25),
        "when": when,
        "time": time,
        "set": {"a", "b"},
        "tuple": (1, 2),
    }

    json_utils.save_json(path, payload)
    loaded = json_utils.read_json(path)
    np.testing.assert_array_equal(loaded["array"], payload["array"])
    assert loaded["scalar"] == 1.25
    assert loaded["when"] == when
    assert loaded["time"].isot == time.isot
    assert loaded["set"] == {"a", "b"}
    assert loaded["tuple"] == [1, 2]

    json_utils.append_json(path, {"tuple": "next", "new": [5]})
    loaded = json_utils.load_json(path)
    assert loaded["tuple"] == [1, 2, "next"]
    assert loaded["new"] == [5]

    list_path = tmp_path / "list.json"
    json_utils.save_json(list_path, [1])
    json_utils.append_json(list_path, [2, 3])
    assert json_utils.load_json(list_path) == [1, 2, 3]

    scalar_path = tmp_path / "scalar.json"
    json_utils.save_json(scalar_path, "old")
    json_utils.append_json(scalar_path, {"new": "value"})
    assert json_utils.load_json(scalar_path) == ["old", {"new": "value"}]

    with pytest.raises(TypeError):
        json_utils.save_json(tmp_path / "bad.json", {"bad": object()})


def test_hdf5_utils_nested_append_combine_and_verify(tmp_path, capsys):
    src1 = tmp_path / "nested" / "src1.h5"
    src2 = tmp_path / "src2.h5"
    out = tmp_path / "combined.h5"

    hdf5_utils.save_h5(src1, "group/a", np.array([1, 2]))
    hdf5_utils.save_h5(src1, "group/a", np.array([9]))
    assert "exists in file" in capsys.readouterr().out
    assert hdf5_utils.h5_key_exists(src1, "group/a")
    assert not hdf5_utils.h5_key_exists(src1, "missing")
    assert not hdf5_utils.h5_key_exists(tmp_path / "missing.h5", "x")

    hdf5_utils.append_h5(src1, "group/b", np.array([[1, 2]]))
    hdf5_utils.append_h5(src1, "group/b", np.array([3, 4]))
    np.testing.assert_array_equal(hdf5_utils.read_h5(src1, "group/b"), [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Incompatible"):
        hdf5_utils.append_h5(src1, "group/b", np.array([[1, 2, 3]]))

    hdf5_utils.overwrite_h5(src1, "scalar", 1.0)
    hdf5_utils.append_h5(src1, "scalar", 2.0)
    np.testing.assert_array_equal(hdf5_utils.read_h5(src1, "scalar"), [1.0, 2.0])
    assert hdf5_utils.read_h5(src1, "group") is None
    with pytest.raises(FileNotFoundError):
        hdf5_utils.read_h5(tmp_path / "none.h5", "x")

    hdf5_utils.save_h5(src2, "group/c", np.array([5]))
    hdf5_utils.combine_h5(out, [src1, src2, tmp_path / "missing.h5"], verbose=True, overwrite=True)
    assert set(hdf5_utils.h5_keys(out)) == {"group/a", "group/b", "group/c", "scalar"}
    assert set(hdf5_utils.h5_root_keys(out)) == {"group", "scalar"}
    assert sorted(hdf5_utils.read_h5_all(out)) == ["group/a", "group/b", "group/c", "scalar"]
    nested = hdf5_utils.read_h5_to_dict(out)
    np.testing.assert_array_equal(nested["group"]["c"], [5])

    assert hdf5_utils.verify_h5_file(out, mode="open")
    assert hdf5_utils.verify_h5_file(out, mode="structure")
    assert hdf5_utils.verify_h5_file(out, mode="full")
    assert not hdf5_utils.verify_h5_file(tmp_path / "missing.h5", verbose=True)
    with pytest.raises(ValueError, match="Invalid mode"):
        hdf5_utils.verify_h5_file(out, mode="deep")


def test_dict_hdf5_roundtrip_and_key_filter(tmp_path):
    path = tmp_path / "dict.h5"
    now = datetime(2026, 1, 1, 0, 0, 0)
    payload = {
        "nested": {"array": np.arange(3), "text": "hello"},
        "mixed": [1, "two", {"three": 3}],
        "bytes": b"abc",
        "when": now,
        "time": Time("2026-01-01T00:00:00", scale="utc"),
        "object": {"set": {1, 2}},
    }

    dict_to_from_hdf5.save_dict_to_hdf5(path, payload)
    loaded = dict_to_from_hdf5.load_dict_from_hdf5(path)
    np.testing.assert_array_equal(loaded["nested"]["array"], [0, 1, 2])
    assert loaded["nested"]["text"] == "hello"
    assert loaded["mixed"][1] == "two"
    assert loaded["bytes"] == b"abc"
    assert loaded["when"] == now
    assert loaded["time"].isot.startswith("2026-01-01")
    assert loaded["object"]["set"] == {1, 2}

    filtered = dict_to_from_hdf5.load_dict_from_hdf5(path, keys={"nested"})
    assert list(filtered) == ["nested"]

    with pytest.raises(TypeError):
        dict_to_from_hdf5.save_dict_to_hdf5(path, {"bad": object()}, pickle_objects=False)


def test_dict_hdf5_edge_types_empty_groups_and_errors(tmp_path):
    path = tmp_path / "edge.h5"
    time = Time("2026-01-01T00:00:00", scale="utc")
    payload = {
        "empty_list": [],
        "empty_dict": {},
        "numbers": [1, 2, 3],
        "date": datetime(2026, 1, 2).date(),
        "clock": datetime(2026, 1, 2, 3, 4, 5).time(),
        "time": time,
        "memory": memoryview(b"xyz"),
        "picked": {"set": {1, 2}},
    }

    dict_to_from_hdf5.save_dict_to_hdf5(path, payload)
    loaded = dict_to_from_hdf5.load_dict_from_hdf5(
        path,
        keys={"empty_list", "empty_dict", "date", "clock", "time", "memory", "picked"},
    )
    assert loaded["empty_list"] == []
    assert loaded["empty_dict"] == {}
    assert loaded["date"].isoformat() == "2026-01-02"
    assert loaded["clock"].isoformat() == "03:04:05"
    assert loaded["time"].isot.startswith("2026-01-01")
    assert loaded["memory"] == b"xyz"
    assert loaded["picked"] == {"set": {1, 2}}
    assert "numbers" not in loaded

    with pytest.raises(TypeError, match="string keys"):
        dict_to_from_hdf5.save_dict_to_hdf5(tmp_path / "badkey.h5", {1: "bad"})


def test_hdf5_to_csv_helpers_and_main(tmp_path, monkeypatch, capsys):
    h5_path = tmp_path / "sample.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("scalar", data=7)
        handle.create_dataset("group/vector", data=np.array([1, 2]))
        handle.create_dataset("group/matrix", data=np.array([[1, 2], [3, 4]]))
        handle.create_dataset("cube", data=np.zeros((1, 1, 1)))

    with h5py.File(h5_path, "r") as handle:
        assert [key for key, _ in hdf5_to_csv.iter_datasets(handle)] == ["/cube", "/group/matrix", "/group/vector", "/scalar"]
        assert hdf5_to_csv.dataset_to_python(handle["scalar"]) == 7

    assert hdf5_to_csv._stringify(None) == ""
    assert hdf5_to_csv._stringify(b"abc") == "abc"
    assert hdf5_to_csv._stringify(b"\xff") == "b'\\xff'"
    assert hdf5_to_csv.key_to_filename("/bad path:a") == "bad_path_a"
    grid = []
    hdf5_to_csv.place_cell(grid, 1, 2, "x")
    assert hdf5_to_csv.normalize_grid(grid) == [["", "", ""], ["", "", "x"]]

    with pytest.warns(UserWarning, match="Ignoring /cube"):
        out_dir = hdf5_to_csv.hdf5_to_csv_per_key(h5_path)
    assert (out_dir / "group_vector.csv").exists()

    with pytest.raises(SystemExit):
        hdf5_to_csv.main([])

    with pytest.warns(UserWarning, match="Ignoring /cube"):
        hdf5_to_csv.main([str(h5_path)])
    assert "Wrote:" in capsys.readouterr().out


def test_csv_and_io_utils(tmp_path, capsys):
    a = tmp_path / "nested" / "a.csv"
    b = tmp_path / "b.csv"
    csv_utils.save_csv(a, {"x": [1, 2], "y": [3.0, np.nan]})
    csv_utils.save_csv(b, pd.DataFrame({"x": [4], "y": [5.0]}))
    assert "Saved" in capsys.readouterr().out

    assert guess_csv_delimiter(a) == ","
    semicolon = tmp_path / "semicolon.csv"
    semicolon.write_text("x;y\n1;2\n", encoding="utf-8")
    assert guess_csv_delimiter(semicolon, delimiters=(";", ",")) == ";"
    assert csv_utils.read_csv_header(a) == ["x", "y"]
    header_path = tmp_path / "headers" / "header.csv"
    csv_utils.save_csv_header(header_path, ["left", "right"])
    assert csv_utils.read_csv_header(header_path) == ["left", "right"]
    assert csv_utils.read_csv(a, col="x", to_np=True).tolist() == [1, 2]
    assert csv_utils.read_csv(a, drop_nan=True).shape == (1, 2)

    combined = tmp_path / "combined" / "combined.csv"
    csv_utils.append_csv([a, b, tmp_path / "missing.csv"], save_path=combined)
    assert pd.read_csv(combined).shape == (3, 2)
    assert "ERRORED" in capsys.readouterr().out

    disk = tmp_path / "disk" / "disk.csv"
    csv_utils.append_csv_on_disk([a, b], disk)
    assert pd.read_csv(disk).shape == (3, 2)

    rows = tmp_path / "rows" / "rows.csv"
    csv_utils.append_dict_to_csv(rows, {"a": [1, 2], "b": [3, 4]})
    csv_utils.append_dict_to_csv(rows, np.array([[5, 6]]))
    csv_utils.save_csv_array_to_line(rows, ["tail", 7])
    csv_utils.save_csv_line(rows, pd.DataFrame({"a": [8], "b": [9]}))
    assert rows.exists()

    csv_utils._column_data = None
    assert bool(csv_utils.exists_in_csv(rows, "a", "1"))
    assert not csv_utils.exists_in_csv(tmp_path / "none.csv", "a", 1)

    images = tmp_path / "images"
    images.mkdir()
    (images / "plot10.png").write_text("x")
    (images / "plot2.jpg").write_text("x")
    (images / "note.txt").write_text("x")
    assert {p.name for p in map(Path, io_utils.get_image_paths(images))} == {"plot2.jpg", "plot10.png"}
    with pytest.raises(ValueError):
        io_utils.get_image_paths(tmp_path / "missing")

    assert io_utils.file_exists(str(images / "plot2"))
    assert io_utils.exists(images)
    made = tmp_path / "made"
    io_utils.mkdir(made)
    io_utils.mkdir(made)
    io_utils.rmfile(images / "note.txt")
    assert not (images / "note.txt").exists()
    assert io_utils.pd_flatten(["[1,2]", 4], factor=2) == [0.5, 1.0, 2.0]
    np.testing.assert_array_equal(io_utils.str_to_array("[1, 2, 3]"), [1, 2, 3])
    assert len(io_utils.allfiles(tmp_path)) >= 1


def test_pickle_and_memory_helpers_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "payload.pkl"
    payload = {"name": "ssatk", "values": [1, 2, 3]}

    pickle_utils.save_pickle(payload, path)

    assert pickle_utils.read_pickle(path) == payload

    class FakeMemoryInfo:
        rss = 2 * 1024**3

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def memory_info(self):
            return FakeMemoryInfo()

    monkeypatch.setattr(get_memory, "Process", FakeProcess)
    monkeypatch.setattr(get_memory.os, "getpid", lambda: 123)

    assert get_memory.get_memory_usage() == "Memory used: 2.00 GB"


def test_io_utils_directory_branches_and_numbered_image_sort(tmp_path, monkeypatch, capsys):
    missing_parent = tmp_path / "blocked" / "child"

    def raising_makedirs(*args, **kwargs):
        raise OSError("blocked")

    monkeypatch.setattr(io_utils.os, "makedirs", raising_makedirs)
    io_utils.mkdir(missing_parent)
    assert "Error creating directory" in capsys.readouterr().out

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a", encoding="utf-8")
    dest = tmp_path / "dest"
    io_utils.mvdir(src, dest)
    assert (dest / "a.txt").exists()
    io_utils.mvdir(dest, dest)
    assert "already exists" in capsys.readouterr().out

    io_utils.rmdir(tmp_path / "missing")
    assert "does not exist" in capsys.readouterr().out

    doomed = tmp_path / "doomed"
    doomed.mkdir()
    io_utils.rmdir(doomed)
    assert not doomed.exists()

    blocked = tmp_path / "blocked_delete"
    blocked.mkdir()
    monkeypatch.setattr(io_utils.shutil, "rmtree", lambda path: (_ for _ in ()).throw(OSError("blocked")))
    io_utils.rmdir(blocked)
    assert "Error deleting" in capsys.readouterr().out

    files = tmp_path / "files"
    files.mkdir()
    for name in ["plot_001_frame_2.png", "plot_001_frame_10.png", "plot_final.png", "skip.txt"]:
        (files / name).write_text("x", encoding="utf-8")
    names = [Path(path).name for path in io_utils.listdir(str(files), files_only=True, exclude="skip", do_sort=False)]
    assert set(names) == {"plot_001_frame_2.png", "plot_001_frame_10.png", "plot_final.png"}
    all_names = [Path(path).name for path in io_utils.listdir(str(files / "*.png"), files_only=False, do_sort=True)]
    assert all_names[:2] == ["plot_001_frame_2.png", "plot_001_frame_10.png"]

    images = io_utils.get_image_paths(files, sort_by_number=True)
    assert [Path(path).name for path in images[:2]] == ["plot_001_frame_2.png", "plot_001_frame_10.png"]
    unsorted_images = io_utils.get_image_paths(files, sort_by_number=False)
    assert len(unsorted_images) == 3

    frame = pd.DataFrame({"a": ["[1, 2]"], "b": ["[3, 4]"]})
    converted = io_utils.pdstr_to_arrays(frame)
    assert converted.shape == (1, 2)
    np.testing.assert_array_equal(converted[0, 0], [1.0, 2.0])


def test_h5cache_roundtrip_and_filtering(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SSATK_DATA_DIR", str(tmp_path))
    out = h5cache_module.h5cache(
        data={
            "plain": 1,
            "name/with/slash": "value",
            "array": np.arange(3),
            "bytes": b"abc",
            "nested": {"x": 2},
            "object_array": np.array([object()], dtype=object),
            "_private": 99,
        },
        filename="cache",
        add_timestamp=False,
    )
    loaded = h5cache_module.h5load(out, summary=True)
    assert loaded["plain"] == 1
    assert loaded["name/with/slash"] == "value"
    np.testing.assert_array_equal(loaded["array"], [0, 1, 2])
    assert loaded["bytes"] == b"abc"
    assert loaded["nested"] == {"x": 2}
    assert "object_array" in loaded
    assert "_private" not in loaded
    assert "[h5load] file" in capsys.readouterr().out

    with pytest.raises(TypeError):
        h5cache_module.h5cache(data=[1, 2], filename="bad", add_timestamp=False)
    with pytest.raises(TypeError):
        h5cache_module.h5load(object())


def test_h5cache_caller_frame_timestamp_and_legacy_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SSATK_DATA_DIR", str(tmp_path))
    assert h5cache_module._enc_key("a/b%c") == "a%2Fb%25c"
    assert h5cache_module._dec_key("a%2Fb%25c") == "a/b%c"
    assert h5cache_module._is_nonsaved_symbol(types)
    assert h5cache_module._is_nonsaved_symbol(test_h5cache_caller_frame_timestamp_and_legacy_files)
    assert h5cache_module._is_nonsaved_symbol(type)

    class Unpicklable:
        def __getstate__(self):
            raise RuntimeError("nope")

    def cache_caller_scope():
        local_value = np.array([4, 5, 6])
        _private = "skip"
        unpicklable = Unpicklable()
        return h5cache_module.h5cache(
            filename="frame_cache",
            exclude_names={"json"},
            only_picklable=True,
            add_timestamp=True,
        )

    out = Path(cache_caller_scope())
    assert out.name.startswith("frame_cache_")
    loaded = h5cache_module.h5load(out, summary=False)
    np.testing.assert_array_equal(loaded["local_value"], [4, 5, 6])
    assert "_private" not in loaded
    assert "unpicklable" not in loaded

    named = h5cache_module.h5cache(data={1: "one"}, filename="named_cache", add_timestamp=False)
    assert h5cache_module.h5load("named_cache", summary=False)["1"] == "one"
    assert Path(named).name == "named_cache.h5"

    root_path = tmp_path / "legacy_root.h5"
    with h5py.File(root_path, "w") as handle:
        handle.create_dataset("plain_array", data=np.arange(2))
        handle.create_dataset("plain_scalar", data=np.float64(2.5))
        handle.create_dataset("plain_bytes", data=np.bytes_("abc"))
        text = handle.create_dataset("text_bytes", data=np.bytes_("hello"))
        text.attrs["__kind__"] = "str"
    legacy = h5cache_module.h5load(root_path, summary=True, max_items=1)
    assert legacy["plain_array"].tolist() == [0, 1]
    assert legacy["plain_scalar"] == 2.5
    assert legacy["plain_bytes"] == b"abc"
    assert legacy["text_bytes"] == "hello"
    assert "truncated" in capsys.readouterr().out


def test_h5cache_error_branches(monkeypatch):
    original_h5py = h5cache_module.h5py
    monkeypatch.setattr(h5cache_module, "h5py", None)
    monkeypatch.setattr(h5cache_module, "_H5PY_IMPORT_ERROR", ImportError("missing"), raising=False)
    with pytest.raises(ImportError, match="h5py is required"):
        h5cache_module.h5cache(data={}, filename="x")
    with pytest.raises(ImportError, match="h5py is required"):
        h5cache_module.h5load("x")
    monkeypatch.setattr(h5cache_module, "h5py", original_h5py)
