from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import ssapy_toolkit as ssatk
from ssapy_toolkit.io.ssatk_save import ssatk_load, ssatk_save, supported_save_formats


def test_ssatk_save_load_hdf5_defaults_keys_and_roots(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SSATK_DATA_DIR", str(data_root))

    h5_path = ssatk_save({"orbit": {"r": np.arange(3.0), "name": "demo"}}, "runs/orbit.h5")

    assert h5_path == data_root / "runs" / "orbit.h5"
    loaded = ssatk_load("runs/orbit.h5")
    np.testing.assert_array_equal(loaded["orbit"]["r"], np.arange(3.0))
    assert loaded["orbit"]["name"] == "demo"

    single_path = ssatk_save(np.arange(4), "single.h5")
    np.testing.assert_array_equal(ssatk_load(single_path), np.arange(4))

    ssatk_save(np.arange(2), single_path, key="orbit/r", overwrite=True)
    np.testing.assert_array_equal(ssatk_load(single_path, key="orbit/r"), np.arange(2))
    with pytest.raises(FileExistsError, match="orbit/r"):
        ssatk_save(np.arange(2), single_path, key="orbit/r", overwrite=False)
    with pytest.raises(ValueError, match="Invalid HDF5 key"):
        ssatk_save(np.arange(2), "bad.h5", key="../bad")


def test_ssatk_save_load_numpy_json_csv_text_and_pickle(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SSATK_DATA_DIR", str(data_root))

    npy_path = ssatk_save(np.arange(3), "array.npy")
    np.testing.assert_array_equal(ssatk_load("array.npy"), np.arange(3))
    assert npy_path == data_root / "array.npy"

    npz_path = ssatk_save({"orbit": {"r": [1, 2], "v": [3, 4]}}, "arrays.npz")
    loaded_npz = ssatk_load(npz_path)
    assert set(loaded_npz) == {"orbit/r", "orbit/v"}
    np.testing.assert_array_equal(ssatk_load(npz_path, key="orbit/r"), [1, 2])

    json_path = ssatk_save({"arr": np.arange(2), "label": "ok"}, "payload.json")
    loaded_json = ssatk_load(json_path)
    np.testing.assert_array_equal(loaded_json["arr"], np.arange(2))
    assert loaded_json["label"] == "ok"

    frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    csv_path = ssatk_save(frame, "table.csv")
    pd.testing.assert_frame_equal(ssatk_load(csv_path), frame)

    text_path = ssatk_save(["alpha", "beta"], "notes.txt")
    assert ssatk_load(text_path) == "alpha\nbeta\n"

    pickle_path = ssatk_save({"items": [object()]}, tmp_path / "explicit.pkl")
    assert "items" in ssatk_load(pickle_path)


def test_ssatk_save_load_jsonl_npz_pickle_safety_and_top_level_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("SSATK_DATA_DIR", str(tmp_path / "data"))

    jsonl_path = ssatk_save(pd.DataFrame({"x": [1, 2]}), "records.jsonl")
    assert ssatk_load(jsonl_path) == [{"x": 1}, {"x": 2}]

    with pytest.raises(TypeError, match="object dtype"):
        ssatk_save(np.array([object()], dtype=object), "objects.npy")
    with pytest.raises(TypeError, match="object dtype"):
        ssatk_save({"obj": np.array([object()], dtype=object)}, "objects.npz")

    assert ".h5" in supported_save_formats()
    assert ssatk.ssatk_save is ssatk_save
    assert ssatk.ssatk_load is ssatk_load
    assert callable(ssatk.ssatk_save_cache)
    assert callable(ssatk.ssatk_load_cache)


def test_ssatk_save_figure_uses_figure_root(tmp_path, monkeypatch):
    fig_root = tmp_path / "figures"
    monkeypatch.setenv("SSATK_FIGURES_DIR", str(fig_root))

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = ssatk_save(fig, "quicklook.png")

    assert out == fig_root / "quicklook.png"
    assert out.exists()
