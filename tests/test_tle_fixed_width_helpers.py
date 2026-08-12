import numpy as np
import pytest
import importlib

read_3le_by_bit = importlib.import_module("ssapy_toolkit.io.read_3le_by_bit")


TLE1 = "1 25544U 98067A   20029.54791435  .00000742  00000-0  20455-4 0  9993"
TLE2 = "2 25544  51.6436  23.4361 0007417  71.2720  40.4325 15.49147159210616"


def test_cast_value_variants_and_errors():
    assert read_3le_by_bit._cast_value(" abc ", "str") == "abc"
    assert read_3le_by_bit._cast_value("42", "int") == 42
    assert np.isnan(read_3le_by_bit._cast_value("bad", "int"))
    assert read_3le_by_bit._cast_value(".25", "float") == 0.25
    assert np.isnan(read_3le_by_bit._cast_value("bad", "float"))
    assert np.isnan(read_3le_by_bit._cast_value(".", "float"))
    assert read_3le_by_bit._cast_value("16538-3", "tleexp") == pytest.approx(0.00016538)
    assert read_3le_by_bit._cast_value("00000-0", "tleexp") == 0.0
    assert np.isnan(read_3le_by_bit._cast_value("bad", "tleexp"))
    with pytest.raises(ValueError, match="Unknown kind"):
        read_3le_by_bit._cast_value("1", "bad")


def test_parse_fixed_width_records_and_selector(tmp_path):
    path = tmp_path / "fixed.txt"
    path.write_text("A001 1.5\nB002 2.5\nSKIP 9.0\n", encoding="utf-8")
    fields = [
        {"name": "tag", "start": 0, "end": 1, "type": "str"},
        {"name": "num", "start": 1, "end": 4, "type": "int"},
        {"name": "val", "start": 5, "end": 8, "type": "float"},
    ]
    df = read_3le_by_bit.parse_fixed_width_file(path, fields, line_selector=lambda line: line.startswith(("A", "B")))
    assert df.to_dict(orient="list") == {"tag": ["A", "B"], "num": [1, 2], "val": [1.5, 2.5]}

    tle_path = tmp_path / "tle.txt"
    tle_path.write_text(TLE1 + "\n" + TLE2 + "\n" + TLE1 + "\n", encoding="utf-8")
    record_fields = [
        {"name": "sat", "line": 0, "start": 2, "end": 7, "type": "int"},
        {"name": "inc", "line": 1, "start": 8, "end": 16, "type": "float"},
    ]
    df = read_3le_by_bit.parse_fixed_width_file(tle_path, record_fields, record_lines=2)
    assert df.shape == (1, 2)
    assert df.loc[0, "sat"] == 25544
    assert df.loc[0, "inc"] == pytest.approx(51.6436)

    bad_fields = [{"name": "bad", "line": 2, "start": 0, "end": 1, "type": "str"}]
    with pytest.raises(IndexError, match="line"):
        read_3le_by_bit.parse_fixed_width_file(tle_path, bad_fields, record_lines=2)


def test_read_3le_by_bit_schema(tmp_path):
    path = tmp_path / "tle.txt"
    path.write_text(TLE1 + "\n" + TLE2 + "\n", encoding="utf-8")
    df = read_3le_by_bit.read_3le_by_bit(path)
    assert df.shape[0] == 1
    assert df.loc[0, "satnum"] == 25544
    assert df.loc[0, "classification"] == "U"
    assert df.loc[0, "eccentricity"] == pytest.approx(0.0007417)
    assert df.loc[0, "mean_motion"] == pytest.approx(15.49147159)
