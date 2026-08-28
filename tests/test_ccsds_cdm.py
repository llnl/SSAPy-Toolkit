from dataclasses import replace
from io import StringIO

import numpy as np
import pytest
from astropy.time import Time

from ssapy_toolkit.io.ccsds_cdm import (
    ConjunctionDataMessage,
    format_cdm,
    read_cdm,
    write_cdm,
)

_COVARIANCE = """\
CR_R                          = 1.0 [m**2]
CT_R                          = 0.0 [m**2]
CT_T                          = 4.0 [m**2]
CN_R                          = 0.0 [m**2]
CN_T                          = 0.0 [m**2]
CN_N                          = 9.0 [m**2]
CRDOT_R                       = 0.0 [m**2/s]
CRDOT_T                       = 0.0 [m**2/s]
CRDOT_N                       = 0.0 [m**2/s]
CRDOT_RDOT                    = 0.01 [m**2/s**2]
CTDOT_R                       = 0.0 [m**2/s]
CTDOT_T                       = 0.0 [m**2/s]
CTDOT_N                       = 0.0 [m**2/s]
CTDOT_RDOT                   = 0.0 [m**2/s**2]
CTDOT_TDOT                    = 0.04 [m**2/s**2]
CNDOT_R                       = 0.0 [m**2/s]
CNDOT_T                       = 0.0 [m**2/s]
CNDOT_N                       = 0.0 [m**2/s]
CNDOT_RDOT                    = 0.0 [m**2/s**2]
CNDOT_TDOT                    = 0.0 [m**2/s**2]
CNDOT_NDOT                    = 0.09 [m**2/s**2]
"""

_DRAG_COVARIANCE = """\
CDRG_R                        = 1 [m**3/kg]
CDRG_T                        = 2 [m**3/kg]
CDRG_N                        = 3 [m**3/kg]
CDRG_RDOT                     = 4 [m**3/(kg*s)]
CDRG_TDOT                     = 5 [m**3/(kg*s)]
CDRG_NDOT                     = 6 [m**3/(kg*s)]
CDRG_DRG                      = 7 [m**4/kg**2]
"""

_SRP_COVARIANCE = """\
CSRP_R                        = 1 [m**3/kg]
CSRP_T                        = 2 [m**3/kg]
CSRP_N                        = 3 [m**3/kg]
CSRP_RDOT                     = 4 [m**3/(kg*s)]
CSRP_TDOT                     = 5 [m**3/(kg*s)]
CSRP_NDOT                     = 6 [m**3/(kg*s)]
CSRP_DRG                      = 7 [m**4/kg**2]
CSRP_SRP                      = 8 [m**4/kg**2]
"""

_THRUST_COVARIANCE = """\
CTHR_R                        = 1 [m**2/s**2]
CTHR_T                        = 2 [m**2/s**2]
CTHR_N                        = 3 [m**2/s**2]
CTHR_RDOT                     = 4 [m**2/s**3]
CTHR_TDOT                     = 5 [m**2/s**3]
CTHR_NDOT                     = 6 [m**2/s**3]
CTHR_DRG                      = 7 [m**3/(kg*s**2)]
CTHR_SRP                      = 8 [m**3/(kg*s**2)]
CTHR_THR                      = 9 [m**2/s**4]
"""


def _object(label, designator, frame="GCRF", epoch=None):
    epoch_line = "" if epoch is None else f"EPOCH                         = {epoch}\n"
    return f"""OBJECT                        = {label}
OBJECT_DESIGNATOR             = {designator}
CATALOG_NAME                  = SATCAT
OBJECT_NAME                   = TEST {label}
INTERNATIONAL_DESIGNATOR      = 2020-001A
EPHEMERIS_NAME                = TEST
COVARIANCE_METHOD             = CALCULATED
MANEUVERABLE                  = NO
REF_FRAME                     = {frame}
TIME_SYSTEM                   = UTC
{epoch_line}X                             = 7000.0 [km]
Y                             = 100.0 [km]
Z                             = 50.0 [km]
X_DOT                         = -0.1 [km/s]
Y_DOT                         = 7.5 [km/s]
Z_DOT                         = 0.2 [km/s]
USER_DEFINED                  = retained
{_COVARIANCE}"""


def _fixture(frame="GCRF", epoch=None):
    return f"""CCSDS_CDM_VERS                = 1.0
CREATION_DATE                 = 2025-01-01T00:00:00.000
ORIGINATOR                    = TEST
MESSAGE_ID                    = TEST-MESSAGE-1
COMMENT header comment
TCA                           = 2025-01-02T00:00:00.000
MISS_DISTANCE                 = 715 [m]
RELATIVE_SPEED                = 12.5 [m/s]
RELATIVE_POSITION_R           = 1 [m]
RELATIVE_POSITION_T           = 2 [m]
RELATIVE_POSITION_N           = 3 [m]
RELATIVE_VELOCITY_R           = 4 [m/s]
RELATIVE_VELOCITY_T           = 5 [m/s]
RELATIVE_VELOCITY_N           = 6 [m/s]
COLLISION_PROBABILITY         = 1.0e-4
CONJUNCTION_ID                = TEST-1
MESSAGE_EXTRA                 = preserved
{_object("OBJECT1", "ONE", frame, epoch)}
{_object("OBJECT2", "TWO", frame, epoch).replace("Y                             = 100.0", "Y                             = 101.0")}"""


def test_read_standard_layout_and_si_conversion():
    message = read_cdm(_fixture())

    assert message.object1.state.tolist() == [7_000_000.0, 100_000.0, 50_000.0, -100.0, 7_500.0, 200.0]
    assert message.object2.state[1] == 101_000.0
    assert message.object1.epoch == message.tca
    assert message.miss_distance_m == 715.0
    assert message.relative_position_rtn_m.flags.writeable is False
    assert message.extra_fields["MESSAGE_EXTRA"] == "preserved"
    assert message.object1.extra_fields["USER_DEFINED"] == "retained"


def test_gcrf_rtn_covariance_rotation_and_immutable_arrays():
    message = read_cdm(_fixture())
    obj = message.object1

    assert obj.covariance_ref_frame == "RTN"
    rotated = obj.position_covariance_gcrf()
    assert np.allclose(rotated, rotated.T, atol=1e-12)
    assert np.allclose(np.linalg.eigvalsh(rotated), [1.0, 4.0, 9.0], atol=1e-12)
    with pytest.raises(ValueError):
        obj.state[0] = 0.0
    with pytest.raises(TypeError):
        message.extra_fields["new"] = "value"


@pytest.mark.parametrize("frame", ["EME2000", "ITRF"])
def test_supported_frames_transform_to_gcrf(frame):
    message = read_cdm(_fixture(frame=frame))
    obj = message.object1
    assert obj.reference_frame == frame
    assert not np.allclose(obj.state_gcrf(), obj.state, rtol=0.0, atol=1e-8)


def test_ordinal_dates_and_matching_reference_frames():
    text = _fixture().replace(
        "2025-01-01T00:00:00.000", "2025-001T00:00:00.123456789"
    ).replace("2025-01-02T00:00:00.000", "2025-002T00:00:00.987654321")
    message = read_cdm(text)
    assert message.creation_date.utc.isot == "2025-01-01T00:00:00.123456789"
    assert message.tca.utc.isot == "2025-01-02T00:00:00.987654321"
    assert "2025-01-02T00:00:00.987654321" in format_cdm(message)

    mismatched = _fixture().replace(
        "REF_FRAME                     = GCRF",
        "REF_FRAME                     = EME2000",
        1,
    )
    with pytest.raises(ValueError, match="REF_FRAME values must match"):
        read_cdm(mismatched)


def test_optional_epoch_and_writer_round_trip():
    epoch = "2025-01-01T12:34:56.000"
    message = read_cdm(_fixture(epoch=epoch))
    text = format_cdm(message)
    for field in ("META_START", "META_STOP", "DATA_START", "DATA_STOP", "COVARIANCE_START", "EPOCH", "TIME_SYSTEM", "COV_REF_FRAME"):
        assert field not in text
    assert "[1]" not in text
    assert "COMMENT header comment" in text
    assert "COMMENT                        =" not in text
    result = read_cdm(text)
    assert result.object1.epoch == result.tca
    assert np.allclose(result.object2.covariance_rtn, message.object2.covariance_rtn)


def test_stream_path_and_overwrite_safety(tmp_path):
    message = read_cdm(_fixture())
    stream = StringIO()
    assert write_cdm(message, stream) is None
    assert read_cdm(StringIO(stream.getvalue())).object1.object_designator == "ONE"

    path = tmp_path / "message.cdm"
    assert write_cdm(message, path) == path
    with pytest.raises(FileExistsError):
        write_cdm(message, path)
    write_cdm(message, path, overwrite=True)


@pytest.mark.parametrize(
    "change, error",
    [
        ("OBJECT_DESIGNATOR             = ONE", "duplicate"),
        ("MISS_DISTANCE                 = -1 [m]", "nonnegative"),
        ("CR_R                          = -1 [m**2]", "positive semidefinite"),
        ("X                             = 7000 [mile]", "expected"),
        ("X                             = 7000", "expected"),
        ("REF_FRAME                     = TEME", "unsupported"),
    ],
)
def test_invalid_messages_are_rejected(change, error):
    text = _fixture()
    if change.startswith("OBJECT_DESIGNATOR"):
        text = text.replace("OBJECT_DESIGNATOR             = ONE\n", "OBJECT_DESIGNATOR             = ONE\n" + change + "\n", 1)
    elif change.startswith("MISS_DISTANCE"):
        text = text.replace("MISS_DISTANCE                 = 715 [m]", change)
    elif change.startswith("CR_R"):
        text = text.replace("CR_R                          = 1.0 [m**2]", change, 1)
    elif change.startswith("X "):
        text = text.replace("X                             = 7000.0 [km]", change, 1)
    elif change.startswith("REF_FRAME"):
        text = text.replace("REF_FRAME                     = GCRF", change, 1)
    else:
        text = text.replace("CR_R                          = 1.0 [m**2]", change, 1)
    with pytest.raises(ValueError, match=error):
        read_cdm(text)


def test_comments_and_covariance_extensions_follow_flattened_cdm_layout():
    text = _fixture().replace(
        "OBJECT                        = OBJECT1\n",
        "COMMENT object one\nOBJECT                        = OBJECT1\nCOMMENT object one data\n",
        1,
    ).replace(
        "CNDOT_NDOT                    = 0.09 [m**2/s**2]\n",
        "CNDOT_NDOT                    = 0.09 [m**2/s**2]\n"
        + _DRAG_COVARIANCE,
        1,
    )
    message = read_cdm(text)
    assert message.object1.comments == ("object one", "object one data")
    assert message.object1.extra_fields["CDRG_R"] == "1 [m**3/kg]"
    canonical = format_cdm(message)
    assert "COMMENT object one data" in canonical
    assert canonical.index("OBJECT                         = OBJECT1") < canonical.index(
        "COMMENT object one data"
    )
    assert canonical.index("CNDOT_NDOT") < canonical.index("CDRG_R")
    assert "[1]" not in canonical
    assert "COMMENT                        = object one data" not in canonical


def test_required_header_and_normative_object_values():
    with pytest.raises(ValueError, match="MESSAGE_ID"):
        read_cdm(_fixture().replace("MESSAGE_ID                    = TEST-MESSAGE-1\n", ""))

    with pytest.raises(ValueError, match="COVARIANCE_METHOD"):
        read_cdm(_fixture().replace("CALCULATED", "ESTIMATED", 1))

    with pytest.raises(ValueError, match="MANEUVERABLE"):
        read_cdm(_fixture().replace("MANEUVERABLE                  = NO", "MANEUVERABLE                  = UNKNOWN", 1))

    with pytest.raises(ValueError, match="COMMENT"):
        read_cdm(_fixture().replace("COMMENT header comment", "COMMENT = header comment"))

    for key, value in (
        ("ORIGINATOR", "BAD VALUE"),
        ("MESSAGE_ID", "BAD.VALUE"),
        ("OBJECT_DESIGNATOR", "BAD VALUE"),
        ("CATALOG_NAME", "BAD.VALUE"),
        ("INTERNATIONAL_DESIGNATOR", "BAD VALUE"),
    ):
        old_line = next(line for line in _fixture().splitlines() if line.startswith(key))
        with pytest.raises(ValueError, match=key):
            read_cdm(_fixture().replace(old_line, f"{key} = {value}", 1))


def test_standard_fields_cannot_move_between_sections():
    message = read_cdm(_fixture())
    with pytest.raises(ValueError, match="wrong section"):
        replace(message, extra_fields={"AREA_PC": "1 [m**2]"})
    with pytest.raises(ValueError, match="wrong section"):
        replace(message.object1, extra_fields={"TCA": "2025-01-02T00:00:00"})

    with pytest.raises(ValueError, match="out of order"):
        read_cdm(_fixture().replace("MESSAGE_EXTRA", "AREA_PC", 1))
    with pytest.raises(ValueError, match="out of order"):
        read_cdm(_fixture().replace("USER_DEFINED", "SCREEN_VOLUME_X", 1))


@pytest.mark.parametrize(
    "line, error",
    [
        ("SCREEN_VOLUME_SHAPE = CUBE", "SCREEN_VOLUME_SHAPE"),
        ("AREA_PC = nope [m**2]", "numeric"),
        ("RESIDUALS_ACCEPTED = 150 [%]", r"\[0, 100\]"),
        ("OBS_AVAILABLE = 1.5", "integer"),
        ("SOLAR_RAD_PRESSURE = MAYBE", "SOLAR_RAD_PRESSURE"),
    ],
)
def test_known_optional_fields_are_validated(line, error):
    anchor = "OBJECT                        = OBJECT1\n"
    if line.startswith("SCREEN_"):
        text = _fixture().replace(anchor, f"{line}\n{anchor}", 1)
    else:
        text = _fixture().replace(
            "REF_FRAME                     = GCRF\n",
            f"REF_FRAME                     = GCRF\n{line}\n",
            1,
        )
    with pytest.raises(ValueError, match=error):
        read_cdm(text)


def test_optional_covariance_rows_must_be_complete():
    partial = _fixture().replace(
        "CNDOT_NDOT                    = 0.09 [m**2/s**2]\n",
        "CNDOT_NDOT                    = 0.09 [m**2/s**2]\nCDRG_R = 1 [m**3/kg]\n",
        1,
    )
    with pytest.raises(ValueError, match="optional covariance rows must be complete"):
        read_cdm(partial)

    for incomplete_predecessors in (_SRP_COVARIANCE, _THRUST_COVARIANCE):
        text = _fixture().replace(
            "CNDOT_NDOT                    = 0.09 [m**2/s**2]\n",
            "CNDOT_NDOT                    = 0.09 [m**2/s**2]\n"
            + incomplete_predecessors,
            1,
        )
        with pytest.raises(ValueError, match="optional covariance rows must be complete"):
            read_cdm(text)

    complete = _fixture().replace(
        "CNDOT_NDOT                    = 0.09 [m**2/s**2]\n",
        "CNDOT_NDOT                    = 0.09 [m**2/s**2]\n"
        + _DRAG_COVARIANCE
        + _SRP_COVARIANCE
        + _THRUST_COVARIANCE,
        1,
    )
    assert read_cdm(format_cdm(read_cdm(complete))).object1.extra_fields["CTHR_THR"] == (
        "9 [m**2/s**4]"
    )


def test_writer_precision_respects_kvn_significant_digit_limit():
    message = read_cdm(_fixture())
    text = format_cdm(message)
    miss_distance = next(
        line.split("=", 1)[1].split("[", 1)[0].strip()
        for line in text.splitlines()
        if line.startswith("MISS_DISTANCE")
    )
    assert len(miss_distance.split("e", 1)[0].replace(".", "").lstrip("+-")) <= 16
    with pytest.raises(ValueError, match="1 through 15"):
        format_cdm(message, precision=16)
    with pytest.raises(ValueError, match="ASCII"):
        format_cdm(replace(message, originator="TÉST"))
    with pytest.raises(ValueError, match="printable ASCII"):
        format_cdm(replace(message, message_for="TEST\tX"))
    with pytest.raises(ValueError, match="printable ASCII"):
        format_cdm(replace(message, message_for="TEST\x01X"))
    with pytest.raises(ValueError, match="254"):
        format_cdm(replace(message, originator="X" * 255))


def test_constructor_validation():
    with pytest.raises(ValueError, match="nonnegative"):
        ConjunctionDataMessage(
            version="1.0",
            creation_date=Time("2025-01-01", scale="utc"),
            originator="TEST",
            message_id="TEST-MESSAGE-1",
            tca=Time("2025-01-01", scale="utc"),
            miss_distance_m=-1,
            object1=read_cdm(_fixture()).object1,
            object2=read_cdm(_fixture()).object2,
        )
