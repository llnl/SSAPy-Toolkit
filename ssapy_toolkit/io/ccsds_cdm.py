"""CCSDS Conjunction Data Message (CDM) KVN 1.0 interchange.

The public objects use SI Cartesian states and SI covariance values.  CDM
KVN state vectors are written in km and km/s; the primary RTN covariance is
written in m², m²/s, and m²/s².  CDM relative quantities use Object 2 minus
Object 1, matching :mod:`ssapy_toolkit.ssa`.

The mandatory six-state RTN covariance block is supported, and optional
CDRG/CSRP/CTHR covariance extensions are preserved.  Dynamic and alternate
XYZ covariance blocks are rejected instead of being silently truncated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TextIO

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    GCRS,
    CartesianDifferential,
    CartesianRepresentation,
    PrecessedGeocentric,
)
from astropy.time import Time

from ssapy_toolkit._paths import ensure_file_parent
from ssapy_toolkit.coordinates.satellite_frames import rtn_to_gcrf_matrix

__all__ = [
    "CDMObject",
    "ConjunctionDataMessage",
    "format_cdm",
    "read_cdm",
    "write_cdm",
]


_TIME_SCALES = {
    "UTC": "utc",
    "TAI": "tai",
    "GPS": "gps",
    "TT": "tt",
    "TDB": "tdb",
    "TCB": "tcb",
    "TCG": "tcg",
    "UT1": "ut1",
}
_SUPPORTED_FRAMES = {"GCRF", "EME2000"}
_MANEUVERABLE = {"YES", "NO", "UNKNOWN", "N/A"}

_STATE_KEYS = {"X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"}
_OBJECT_REQUIRED_KEYS = {
    "OBJECT",
    "OBJECT_DESIGNATOR",
    "CATALOG_NAME",
    "OBJECT_NAME",
    "INTERNATIONAL_DESIGNATOR",
    "EPHEMERIS_NAME",
    "COVARIANCE_METHOD",
    "MANEUVERABLE",
    "REF_FRAME",
}
_HEADER_KEYS = {
    "CCSDS_CDM_VERS",
    "CREATION_DATE",
    "ORIGINATOR",
    "MESSAGE_ID",
    "MESSAGE_FOR",
    "CLASSIFICATION",
}
_RELATIVE_KEYS = {
    "CONJUNCTION_ID",
    "TCA",
    "MISS_DISTANCE",
    "RELATIVE_SPEED",
    "RELATIVE_POSITION_R",
    "RELATIVE_POSITION_T",
    "RELATIVE_POSITION_N",
    "RELATIVE_VELOCITY_R",
    "RELATIVE_VELOCITY_T",
    "RELATIVE_VELOCITY_N",
    "COLLISION_PROBABILITY",
    "COLLISION_PROBABILITY_METHOD",
}

_COVARIANCE_FIELDS = (
    ("CR_R", 0, 0, "m**2"),
    ("CT_R", 1, 0, "m**2"),
    ("CT_T", 1, 1, "m**2"),
    ("CN_R", 2, 0, "m**2"),
    ("CN_T", 2, 1, "m**2"),
    ("CN_N", 2, 2, "m**2"),
    ("CRDOT_R", 3, 0, "m**2/s"),
    ("CRDOT_T", 3, 1, "m**2/s"),
    ("CRDOT_N", 3, 2, "m**2/s"),
    ("CRDOT_RDOT", 3, 3, "m**2/s**2"),
    ("CTDOT_R", 4, 0, "m**2/s"),
    ("CTDOT_T", 4, 1, "m**2/s"),
    ("CTDOT_N", 4, 2, "m**2/s"),
    ("CTDOT_RDOT", 4, 3, "m**2/s**2"),
    ("CTDOT_TDOT", 4, 4, "m**2/s**2"),
    ("CNDOT_R", 5, 0, "m**2/s"),
    ("CNDOT_T", 5, 1, "m**2/s"),
    ("CNDOT_N", 5, 2, "m**2/s"),
    ("CNDOT_RDOT", 5, 3, "m**2/s**2"),
    ("CNDOT_TDOT", 5, 4, "m**2/s**2"),
    ("CNDOT_NDOT", 5, 5, "m**2/s**2"),
)
_COVARIANCE_KEYS = {item[0] for item in _COVARIANCE_FIELDS}
_UNSUPPORTED_COVARIANCE_PREFIXES = (
    "CX_",
    "CY_",
    "CZ_",
)
_COVARIANCE_EXTENSION_PREFIXES = ("CDRG_", "CSRP_", "CTHR_")
_UNSUPPORTED_COVARIANCE_KEYS = {"ALT_COV_TYPE", "ALT_COV_REF_FRAME"}
_FORBIDDEN_STRUCTURE_KEYS = {
    "META_START",
    "META_STOP",
    "DATA_START",
    "DATA_STOP",
    "COVARIANCE_START",
    "COVARIANCE_STOP",
}
_NONSTANDARD_OBJECT_KEYS = {"EPOCH", "TIME_SYSTEM", "COV_REF_FRAME"}
_OBJECT_FIELD_KEYS = (
    _OBJECT_REQUIRED_KEYS
    | _STATE_KEYS
    | _COVARIANCE_KEYS
    | _NONSTANDARD_OBJECT_KEYS
    | _UNSUPPORTED_COVARIANCE_KEYS
    | _FORBIDDEN_STRUCTURE_KEYS
)
_KNOWN_KEYS = _HEADER_KEYS | _RELATIVE_KEYS | _OBJECT_FIELD_KEYS | _FORBIDDEN_STRUCTURE_KEYS


def _readonly_array(value, shape, name):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    array.flags.writeable = False
    return array


def _readonly_mapping(values):
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        values = dict(values)
    copied = {str(key): str(value) for key, value in values.items()}
    if len(copied) != len(values):
        raise ValueError("extra fields must have unique keys.")
    return MappingProxyType(copied)


def _time(value, scale):
    scale = str(scale).upper()
    if scale not in _TIME_SCALES:
        raise ValueError(f"unsupported CCSDS time system {scale!r}.")
    if isinstance(value, Time):
        result = value.copy()
        if str(result.scale).upper() != scale and scale != "GPS":
            result = Time(result, scale=_TIME_SCALES[scale])
    else:
        text = str(value).strip()
        text = text.removesuffix("Z")
        if scale == "GPS":
            # Astropy has no GPS civil-time scale.  GPS = TAI - 19 s.
            result = Time(text, format="isot", scale="tai") + 19.0 * u.s
        else:
            result = Time(text, format="isot", scale=_TIME_SCALES[scale])
    if not np.all(np.isfinite(np.asarray(result.jd))):
        raise ValueError("CDM times must be finite.")
    return result


def _utc_time(value):
    result = _time(value, "UTC")
    return result.utc


def _state_gcrf(state, frame, epoch):
    if frame == "GCRF":
        result = np.array(state, dtype=float, copy=True)
    else:
        representation = CartesianRepresentation(
            state[:3] * u.m,
            differentials=CartesianDifferential(state[3:] * u.m / u.s),
        )
        eme2000 = PrecessedGeocentric(
            representation,
            obstime=epoch,
            equinox=Time("J2000", scale="tt"),
        )
        transformed = eme2000.transform_to(GCRS(obstime=epoch))
        result = np.concatenate(
            (
                transformed.cartesian.xyz.to_value(u.m),
                transformed.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s),
            )
        )
    result = np.array(result, dtype=float, copy=True)
    result.flags.writeable = False
    return result


def _validate_covariance(covariance):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance_rtn must be finite with shape (6, 6).")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-10):
        raise ValueError("covariance_rtn must be symmetric.")
    # Published KVN entries are rounded independently.  Three parts per
    # million of the matrix scale covers the negative zero mode in the
    # authoritative Orekit fixture while still rejecting physical negatives.
    scale = float(np.max(np.abs(covariance)))
    tolerance = 3.0e-6 * max(scale, np.finfo(float).tiny)
    if np.min(np.linalg.eigvalsh(covariance)) < -tolerance:
        raise ValueError("covariance_rtn must be positive semidefinite.")
    result = np.array(0.5 * (covariance + covariance.T), copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class CDMObject:
    """One CDM object segment with SI state and RTN covariance."""

    object_designator: str
    catalog_name: str
    object_name: str
    international_designator: str
    ephemeris_name: str
    covariance_method: str
    maneuverable: str
    reference_frame: str
    epoch: Time
    state: np.ndarray
    covariance_rtn: np.ndarray
    time_system: str = "UTC"
    covariance_reference_frame: str = "RTN"
    comments: tuple[str, ...] = ()
    extra_fields: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        text_fields = (
            self.object_designator,
            self.catalog_name,
            self.object_name,
            self.international_designator,
            self.ephemeris_name,
            self.covariance_method,
        )
        if not all(isinstance(value, str) and value.strip() for value in text_fields):
            raise ValueError("CDM object identity and covariance fields must be non-empty strings.")
        frame = str(self.reference_frame).strip().upper()
        if frame not in _SUPPORTED_FRAMES:
            raise ValueError(f"unsupported CDM reference frame {frame!r}.")
        time_system = str(self.time_system).strip().upper()
        if time_system != "UTC":
            raise ValueError("only UTC object epochs are supported in CDM KVN 1.0.")
        maneuverable = str(self.maneuverable).strip().upper()
        if maneuverable not in _MANEUVERABLE:
            raise ValueError(f"invalid MANEUVERABLE value {maneuverable!r}.")
        if not isinstance(self.epoch, Time):
            raise TypeError("epoch must be an astropy.time.Time.")
        covariance_frame = str(self.covariance_reference_frame).strip().upper()
        if covariance_frame != "RTN":
            raise ValueError(
                "only the CCSDS RTN covariance frame is supported; "
                f"got {covariance_frame!r}."
            )
        epoch = _utc_time(self.epoch)
        state = _readonly_array(self.state, (6,), "state")
        covariance = _validate_covariance(self.covariance_rtn)
        comments = tuple(str(value) for value in self.comments)
        extras = _readonly_mapping(self.extra_fields)
        extra_keys = {key.strip().upper() for key in extras}
        if extra_keys & _OBJECT_FIELD_KEYS or any(
            key.startswith(_UNSUPPORTED_COVARIANCE_PREFIXES) for key in extra_keys
        ):
            raise ValueError("extra_fields cannot replace known CDM object fields.")
        object.__setattr__(self, "reference_frame", frame)
        object.__setattr__(self, "time_system", time_system)
        object.__setattr__(self, "covariance_reference_frame", covariance_frame)
        object.__setattr__(self, "maneuverable", maneuverable)
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "covariance_rtn", covariance)
        object.__setattr__(self, "comments", comments)
        object.__setattr__(self, "extra_fields", extras)

    @property
    def covariance_ref_frame(self) -> str:
        """CCSDS spelling for :attr:`covariance_reference_frame`."""

        return self.covariance_reference_frame

    def state_gcrf(self) -> np.ndarray:
        """Return the state in GCRF SI units and immutable storage."""

        return _state_gcrf(self.state, self.reference_frame, self.epoch)

    def position_covariance_gcrf(self) -> np.ndarray:
        """Rotate the RTN position covariance into GCRF coordinates."""

        state = self.state_gcrf()
        rotation = rtn_to_gcrf_matrix(state[:3], state[3:])
        covariance = rotation @ self.covariance_rtn[:3, :3] @ rotation.T
        covariance = np.array(0.5 * (covariance + covariance.T), copy=True)
        covariance.flags.writeable = False
        return covariance


@dataclass(frozen=True)
class ConjunctionDataMessage:
    """A CCSDS CDM 1.0 message in SI units."""

    version: str
    creation_date: Time
    originator: str
    tca: Time
    miss_distance_m: float
    object1: CDMObject
    object2: CDMObject
    conjunction_id: str | None = None
    relative_speed_m_s: float | None = None
    relative_position_rtn_m: np.ndarray | None = None
    relative_velocity_rtn_m_s: np.ndarray | None = None
    collision_probability: float | None = None
    collision_probability_method: str | None = None
    message_id: str | None = None
    message_for: str | None = None
    classification: str | None = None
    comments: tuple[str, ...] = ()
    extra_fields: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if str(self.version).strip() != "1.0":
            raise ValueError("only CCSDS CDM version 1.0 is supported.")
        if not isinstance(self.originator, str) or not self.originator.strip():
            raise ValueError("originator must be a non-empty string.")
        creation_date = _utc_time(self.creation_date)
        tca = _utc_time(self.tca)
        miss_distance = float(self.miss_distance_m)
        if not np.isfinite(miss_distance) or miss_distance < 0.0:
            raise ValueError("miss_distance_m must be finite and nonnegative.")
        if not isinstance(self.object1, CDMObject) or not isinstance(self.object2, CDMObject):
            raise TypeError("object1 and object2 must be CDMObject instances.")
        if not np.isclose(self.object1.epoch.jd, tca.jd, rtol=0.0, atol=1.0e-12):
            raise ValueError("object1 epoch must equal message TCA.")
        if not np.isclose(self.object2.epoch.jd, tca.jd, rtol=0.0, atol=1.0e-12):
            raise ValueError("object2 epoch must equal message TCA.")
        relative_speed = self.relative_speed_m_s
        if relative_speed is not None:
            relative_speed = float(relative_speed)
            if not np.isfinite(relative_speed) or relative_speed < 0.0:
                raise ValueError("relative_speed_m_s must be finite and nonnegative.")
        probability = self.collision_probability
        if probability is not None:
            probability = float(probability)
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("collision_probability must lie in [0, 1].")
        position = None if self.relative_position_rtn_m is None else _readonly_array(
            self.relative_position_rtn_m, (3,), "relative_position_rtn_m"
        )
        velocity = None if self.relative_velocity_rtn_m_s is None else _readonly_array(
            self.relative_velocity_rtn_m_s, (3,), "relative_velocity_rtn_m_s"
        )
        extras = _readonly_mapping(self.extra_fields)
        extra_keys = {key.strip().upper() for key in extras}
        if extra_keys & _KNOWN_KEYS or any(
            key.startswith(_UNSUPPORTED_COVARIANCE_PREFIXES) for key in extra_keys
        ):
            raise ValueError("extra_fields cannot replace known CDM message fields.")
        object.__setattr__(self, "version", "1.0")
        object.__setattr__(self, "creation_date", creation_date)
        object.__setattr__(self, "tca", tca)
        object.__setattr__(self, "miss_distance_m", miss_distance)
        object.__setattr__(self, "relative_speed_m_s", relative_speed)
        object.__setattr__(self, "collision_probability", probability)
        object.__setattr__(self, "relative_position_rtn_m", position)
        object.__setattr__(self, "relative_velocity_rtn_m_s", velocity)
        object.__setattr__(self, "comments", tuple(str(value) for value in self.comments))
        object.__setattr__(self, "extra_fields", extras)


def _split_source(source):
    if hasattr(source, "read"):
        text = source.read()
        return text.decode() if isinstance(text, bytes) else str(text)
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if isinstance(source, str):
        if "\n" in source or "\r" in source or source.lstrip().startswith("CCSDS_CDM_VERS"):
            return source
        return Path(source).read_text(encoding="utf-8")
    if hasattr(source, "__fspath__"):
        return Path(source).read_text(encoding="utf-8")
    raise TypeError("source must be CDM text, a path, or a text stream.")


def _parse_line(line):
    if "=" not in line:
        raise ValueError(f"invalid CDM KVN line: {line!r}")
    key, value = line.split("=", 1)
    key = key.strip().upper()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"invalid CDM KVN line: {line!r}")
    return key, value


def _parse_comment(line):
    value = line[len("COMMENT") :].lstrip()
    if value.startswith("="):
        value = value[1:].lstrip()
    if not value:
        raise ValueError(f"invalid CDM COMMENT line: {line!r}")
    return value


def _parse_unit(value, expected, key):
    value = value.strip()
    unit = None
    if value.endswith("]") and "[" in value:
        number, unit = value.rsplit("[", 1)
        value = number.strip()
        unit = unit[:-1].strip().replace("^", "**").replace(" ", "")
    if unit is not None and unit != expected:
        raise ValueError(f"{key} has unit [{unit}], expected [{expected}].")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{key} must be finite.")
    return parsed


def _parse_date(value, key, scale="UTC"):
    try:
        return _time(value, scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key} date.") from exc


def _parse_object(values, state_values, covariance_values, comments, extras, default_epoch):
    missing = (_OBJECT_REQUIRED_KEYS - {"OBJECT"}) - values.keys()
    missing |= _STATE_KEYS - state_values.keys()
    missing |= _COVARIANCE_KEYS - covariance_values.keys()
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"CDM object is missing required fields: {missing_text}.")
    time_system = values.get("TIME_SYSTEM", "UTC").upper()
    frame = values["REF_FRAME"].strip().upper()
    state = np.array(
        [
            _parse_unit(state_values["X"], "km", "X") * 1.0e3,
            _parse_unit(state_values["Y"], "km", "Y") * 1.0e3,
            _parse_unit(state_values["Z"], "km", "Z") * 1.0e3,
            _parse_unit(state_values["X_DOT"], "km/s", "X_DOT") * 1.0e3,
            _parse_unit(state_values["Y_DOT"], "km/s", "Y_DOT") * 1.0e3,
            _parse_unit(state_values["Z_DOT"], "km/s", "Z_DOT") * 1.0e3,
        ]
    )
    covariance = np.zeros((6, 6), dtype=float)
    for key, row, column, unit in _COVARIANCE_FIELDS:
        value = _parse_unit(covariance_values[key], unit, key)
        covariance[row, column] = covariance[column, row] = value
    return CDMObject(
        object_designator=values["OBJECT_DESIGNATOR"],
        catalog_name=values["CATALOG_NAME"],
        object_name=values["OBJECT_NAME"],
        international_designator=values["INTERNATIONAL_DESIGNATOR"],
        ephemeris_name=values["EPHEMERIS_NAME"],
        covariance_method=values["COVARIANCE_METHOD"],
        maneuverable=values["MANEUVERABLE"],
        reference_frame=frame,
        epoch=default_epoch,
        state=state,
        covariance_rtn=covariance,
        time_system=time_system,
        covariance_reference_frame=covariance_values.get("COV_REF_FRAME", "RTN"),
        comments=comments,
        extra_fields=extras,
    )


def read_cdm(source: str | Path | TextIO) -> ConjunctionDataMessage:
    """Read a CCSDS CDM KVN 1.0 message into SI arrays."""

    values = {"header": {}, "relative": {}, "object1": {}, "object2": {}}
    states = {"object1": {}, "object2": {}}
    covariances = {"object1": {}, "object2": {}}
    comments = {"header": [], "relative": [], "object1": [], "object2": []}
    extras = {"header": {}, "relative": {}, "object1": {}, "object2": {}}
    section = "header"
    object_order = []
    seen = {name: set() for name in values}
    for raw_line in _split_source(source).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("COMMENT") and (
            len(line) == len("COMMENT") or line[len("COMMENT")].isspace() or line[len("COMMENT")] == "="
        ):
            key, raw_value = "COMMENT", _parse_comment(line)
            if section not in comments:
                raise ValueError("COMMENT is out of order")
            comments[section].append(raw_value)
            continue
        key, raw_value = _parse_line(line)
        if key in _FORBIDDEN_STRUCTURE_KEYS:
            raise ValueError(
                f"{key} is not valid in the flattened CCSDS CDM KVN 1.0 layout."
            )
        if key == "OBJECT":
            if section not in {"relative", "object1"} or object_order == ["OBJECT1", "OBJECT2"]:
                raise ValueError("OBJECT markers must be OBJECT1 followed by OBJECT2.")
            marker = raw_value.strip().upper()
            expected = "OBJECT1" if not object_order else "OBJECT2"
            if marker != expected:
                raise ValueError(f"expected {expected}, got {marker}.")
            section = marker.lower()
            object_order.append(marker)
            seen[section].add(key)
            values[section][key] = marker
            continue
        if key in seen[section]:
            raise ValueError(f"duplicate CDM key {key}.")
        seen[section].add(key)
        if section == "header":
            if key in _RELATIVE_KEYS:
                section = "relative"
                seen[section].add(key)
                values[section][key] = raw_value
                continue
            if key in _HEADER_KEYS:
                values[section][key] = raw_value
            else:
                extras[section][key] = raw_value
        elif section == "relative":
            if key in _RELATIVE_KEYS:
                values[section][key] = raw_value
            elif key in _HEADER_KEYS:
                raise ValueError(f"CDM header key {key} is out of order.")
            else:
                extras[section][key] = raw_value
        elif section.startswith("object"):
            if key.startswith(_UNSUPPORTED_COVARIANCE_PREFIXES) or key in _UNSUPPORTED_COVARIANCE_KEYS:
                raise ValueError(f"unsupported CDM covariance field {key}.")
            if key in _STATE_KEYS:
                states[section][key] = raw_value
            elif key == "EPOCH":
                # Legacy/nonstandard input; CDM state vectors are at TCA.
                states[section][key] = raw_value
            elif key in _OBJECT_REQUIRED_KEYS or key == "TIME_SYSTEM":
                values[section][key] = raw_value
            elif key in _COVARIANCE_KEYS or key == "COV_REF_FRAME":
                covariances[section][key] = raw_value
            elif key in _HEADER_KEYS or key in _RELATIVE_KEYS:
                raise ValueError(f"CDM key {key} is out of order.")
            else:
                extras[section][key] = raw_value
        else:
            raise ValueError(f"unexpected CDM key {key}.")
    if object_order != ["OBJECT1", "OBJECT2"]:
        raise ValueError("CDM must contain OBJECT1 followed by OBJECT2.")
    header = values["header"]
    relative = values["relative"]
    for key in ("CCSDS_CDM_VERS", "CREATION_DATE", "ORIGINATOR"):
        if key not in header:
            raise ValueError(f"CDM header is missing required field {key}.")
    for key in ("TCA", "MISS_DISTANCE"):
        if key not in relative:
            raise ValueError(f"CDM relative metadata is missing required field {key}.")
    creation_date = _parse_date(header["CREATION_DATE"], "CREATION_DATE")
    time_system = values["object1"].get("TIME_SYSTEM", "UTC").upper()
    if values["object2"].get("TIME_SYSTEM", time_system).upper() != time_system:
        raise ValueError("OBJECT1 and OBJECT2 TIME_SYSTEM values conflict.")
    tca = _parse_date(relative["TCA"], "TCA", "UTC")
    miss_distance = _parse_unit(relative["MISS_DISTANCE"], "m", "MISS_DISTANCE")
    relative_speed = (
        None
        if "RELATIVE_SPEED" not in relative
        else _parse_unit(relative["RELATIVE_SPEED"], "m/s", "RELATIVE_SPEED")
    )
    position = None
    position_keys = ("RELATIVE_POSITION_R", "RELATIVE_POSITION_T", "RELATIVE_POSITION_N")
    if any(key in relative for key in position_keys):
        if not all(key in relative for key in position_keys):
            raise ValueError("all three relative position components are required.")
        position = np.array([_parse_unit(relative[key], "m", key) for key in position_keys])
    velocity = None
    velocity_keys = ("RELATIVE_VELOCITY_R", "RELATIVE_VELOCITY_T", "RELATIVE_VELOCITY_N")
    if any(key in relative for key in velocity_keys):
        if not all(key in relative for key in velocity_keys):
            raise ValueError("all three relative velocity components are required.")
        velocity = np.array([_parse_unit(relative[key], "m/s", key) for key in velocity_keys])
    probability = (
        None
        if "COLLISION_PROBABILITY" not in relative
        else _parse_unit(relative["COLLISION_PROBABILITY"], "1", "COLLISION_PROBABILITY")
    )
    return ConjunctionDataMessage(
        version=header["CCSDS_CDM_VERS"],
        creation_date=creation_date,
        originator=header["ORIGINATOR"],
        tca=tca,
        miss_distance_m=miss_distance,
        object1=_parse_object(
            values["object1"], states["object1"], covariances["object1"],
            comments["object1"], extras["object1"], tca
        ),
        object2=_parse_object(
            values["object2"], states["object2"], covariances["object2"],
            comments["object2"], extras["object2"], tca
        ),
        conjunction_id=relative.get("CONJUNCTION_ID"),
        relative_speed_m_s=relative_speed,
        relative_position_rtn_m=position,
        relative_velocity_rtn_m_s=velocity,
        collision_probability=probability,
        collision_probability_method=relative.get("COLLISION_PROBABILITY_METHOD"),
        message_id=header.get("MESSAGE_ID"),
        message_for=header.get("MESSAGE_FOR"),
        classification=header.get("CLASSIFICATION"),
        comments=tuple(comments["header"] + comments["relative"]),
        extra_fields={**extras["header"], **extras["relative"]},
    )

def _date_text(value, time_system="UTC"):
    if time_system == "GPS":
        return (value.tai - 19.0 * u.s).isot
    scale = _TIME_SCALES[time_system]
    return Time(value, scale=scale).isot


def _line(key, value, unit=None):
    suffix = f" [{unit}]" if unit else ""
    return f"{key:<30} = {value}{suffix}"


def _number(value, precision):
    return f"{float(value):.{precision}e}"


def _object_lines(label, obj, precision):
    lines = [
        _line("OBJECT", label),
        _line("OBJECT_DESIGNATOR", obj.object_designator),
        _line("CATALOG_NAME", obj.catalog_name),
        _line("OBJECT_NAME", obj.object_name),
        _line("INTERNATIONAL_DESIGNATOR", obj.international_designator),
        _line("EPHEMERIS_NAME", obj.ephemeris_name),
        _line("COVARIANCE_METHOD", obj.covariance_method),
        _line("MANEUVERABLE", obj.maneuverable),
        _line("REF_FRAME", obj.reference_frame),
    ]
    lines.extend(
        _line(key, value)
        for key, value in obj.extra_fields.items()
        if not key.upper().startswith(_COVARIANCE_EXTENSION_PREFIXES)
    )
    return [*lines, *(_line("COMMENT", comment) for comment in obj.comments)]


def _state_and_covariance_lines(obj, precision):
    lines = [
        _line("X", _number(obj.state[0] * 1.0e-3, precision), "km"),
        _line("Y", _number(obj.state[1] * 1.0e-3, precision), "km"),
        _line("Z", _number(obj.state[2] * 1.0e-3, precision), "km"),
        _line("X_DOT", _number(obj.state[3] * 1.0e-3, precision), "km/s"),
        _line("Y_DOT", _number(obj.state[4] * 1.0e-3, precision), "km/s"),
        _line("Z_DOT", _number(obj.state[5] * 1.0e-3, precision), "km/s"),
    ]
    lines.extend(_line(key, _number(obj.covariance_rtn[row, column], precision), unit)
                 for key, row, column, unit in _COVARIANCE_FIELDS)
    lines.extend(
        _line(key, value)
        for key, value in obj.extra_fields.items()
        if key.upper().startswith(_COVARIANCE_EXTENSION_PREFIXES)
    )
    return lines


def format_cdm(message: ConjunctionDataMessage, *, precision: int = 17) -> str:
    """Format a CDM as canonical CCSDS KVN text."""

    if not isinstance(message, ConjunctionDataMessage):
        raise TypeError("message must be a ConjunctionDataMessage.")
    if not isinstance(precision, int) or not 1 <= precision <= 17:
        raise ValueError("precision must be an integer from 1 through 17.")
    lines = [
        _line("CCSDS_CDM_VERS", message.version),
        _line("CREATION_DATE", _date_text(message.creation_date)),
        _line("ORIGINATOR", message.originator),
    ]
    for key, value in (("MESSAGE_ID", message.message_id), ("MESSAGE_FOR", message.message_for), ("CLASSIFICATION", message.classification)):
        if value is not None:
            lines.append(_line(key, value))
    lines.extend(_line("COMMENT", comment) for comment in message.comments)
    relative = [
        ("CONJUNCTION_ID", message.conjunction_id, None),
        ("TCA", _date_text(message.tca), None),
        ("MISS_DISTANCE", _number(message.miss_distance_m, precision), "m"),
        ("RELATIVE_SPEED", None if message.relative_speed_m_s is None else _number(message.relative_speed_m_s, precision), "m/s"),
    ]
    if message.relative_position_rtn_m is not None:
        relative.extend((key, _number(value, precision), "m") for key, value in zip(("RELATIVE_POSITION_R", "RELATIVE_POSITION_T", "RELATIVE_POSITION_N"), message.relative_position_rtn_m))
    if message.relative_velocity_rtn_m_s is not None:
        relative.extend((key, _number(value, precision), "m/s") for key, value in zip(("RELATIVE_VELOCITY_R", "RELATIVE_VELOCITY_T", "RELATIVE_VELOCITY_N"), message.relative_velocity_rtn_m_s))
    relative.extend(
        (
            ("COLLISION_PROBABILITY", None if message.collision_probability is None else _number(message.collision_probability, precision), None),
            ("COLLISION_PROBABILITY_METHOD", message.collision_probability_method, None),
        )
    )
    lines.extend(_line(key, value, unit) for key, value, unit in relative if value is not None)
    lines.extend(_line(key, value) for key, value in message.extra_fields.items())
    lines.extend(_object_lines("OBJECT1", message.object1, precision))
    lines.extend(_state_and_covariance_lines(message.object1, precision))
    lines.extend(_object_lines("OBJECT2", message.object2, precision))
    lines.extend(_state_and_covariance_lines(message.object2, precision))
    return "\n".join(lines) + "\n"


def write_cdm(message, destination, *, precision: int = 17, overwrite: bool = False):
    """Write a CDM to a text stream or path."""

    text = format_cdm(message, precision=precision)
    if hasattr(destination, "write"):
        destination.write(text)
        return None
    path = Path(destination)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path = ensure_file_parent(path)
    path.write_text(text, encoding="utf-8")
    return path
