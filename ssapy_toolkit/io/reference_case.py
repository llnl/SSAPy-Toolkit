"""Portable trajectory exports for independent propagation comparisons."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

__all__ = [
    "ReferenceCase",
    "ReferenceCaseFiles",
    "compare_reference_case",
    "read_reference_case",
    "write_reference_case",
]


@dataclass(frozen=True)
class ReferenceCaseFiles:
    """Files written for a portable SSATK reference case."""

    metadata_path: Path
    ephemeris_path: Path


@dataclass(frozen=True)
class ReferenceCase:
    """Trajectory and metadata read from an SSATK reference case."""

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    metadata: Mapping[str, object]


def read_reference_case(source) -> ReferenceCase:
    """Read an SSATK JSON/OEM reference case into SI Cartesian arrays.

    ``source`` may be either the sidecar JSON or the CCSDS OEM file. If only
    the OEM is available, ``t`` starts at zero because its original time
    origin is stored only in the JSON sidecar.
    """

    source = Path(source)
    if source.suffix.lower() == ".json":
        metadata_path = source
        metadata = json.loads(source.read_text(encoding="utf-8"))
        ephemeris_path = source.parent / metadata["files"]["ephemeris"]
    elif source.suffix.lower() == ".oem":
        ephemeris_path = source
        metadata_path = source.with_suffix(".json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
    else:
        raise ValueError("source must be an SSATK .json or CCSDS .oem file.")
    if not ephemeris_path.exists():
        raise FileNotFoundError(ephemeris_path)
    epochs, values = _read_oem_records(ephemeris_path)
    epoch = _epoch_utc(metadata["epoch"]) if metadata.get("epoch") else epochs[0]
    time_origin = float(metadata.get("time_origin_s", 0.0))
    times = time_origin + np.array([(item - epoch).total_seconds() for item in epochs])
    if metadata.get("sample_count") not in (None, len(values)):
        raise ValueError("OEM sample count does not match reference-case metadata.")
    if not np.all(np.diff(times) > 0.0) and len(times) > 1:
        raise ValueError("OEM epochs must be strictly increasing.")
    return ReferenceCase(
        t=times,
        r=values[:, :3] * 1.0e3,
        v=values[:, 3:] * 1.0e3,
        metadata=metadata,
    )


def compare_reference_case(trajectory, reference, *, time_tolerance=1e-9):
    """Compare a trajectory with reference states at the reference epochs."""
    actual_t, actual_r, actual_v = _trajectory_arrays(trajectory)
    reference = read_reference_case(reference) if isinstance(reference, (str, Path)) else reference
    if not all(hasattr(reference, name) for name in ("t", "r", "v")):
        raise TypeError("reference must be a ReferenceCase or .json/.oem path.")
    reference_t, reference_r, reference_v = _trajectory_arrays(reference)
    time_tolerance = float(time_tolerance)
    if not np.isfinite(time_tolerance) or time_tolerance < 0.0:
        raise ValueError("time_tolerance must be non-negative.")
    for name in ("reference_frame", "center_name"):
        actual_metadata = getattr(trajectory, "metadata", {})
        reference_metadata = getattr(reference, "metadata", {})
        actual_value = actual_metadata.get(name) if isinstance(actual_metadata, Mapping) else None
        reference_value = reference_metadata.get(name) if isinstance(reference_metadata, Mapping) else None
        if actual_value is not None and reference_value is not None and actual_value != reference_value:
            raise ValueError(f"trajectory and reference {name} metadata do not match.")
    if reference_t[0] < actual_t[0] - time_tolerance or reference_t[-1] > actual_t[-1] + time_tolerance:
        raise ValueError("reference epochs must overlap the trajectory time span.")
    sample_t = np.clip(reference_t, actual_t[0], actual_t[-1])
    positions = np.array([np.interp(sample_t, actual_t, actual_r[:, i]) for i in range(3)]).T
    velocities = np.array([np.interp(sample_t, actual_t, actual_v[:, i]) for i in range(3)]).T
    position_residual = np.linalg.norm(positions - reference_r, axis=1)
    velocity_residual = np.linalg.norm(velocities - reference_v, axis=1)
    return {
        "max_position_m": float(position_residual.max()),
        "rms_position_m": float(np.sqrt(np.mean(position_residual**2))),
        "final_position_m": float(position_residual[-1]),
        "max_velocity_m_s": float(velocity_residual.max()),
        "rms_velocity_m_s": float(np.sqrt(np.mean(velocity_residual**2))),
        "final_velocity_m_s": float(velocity_residual[-1]),
        "sample_count": int(reference_t.size),
    }


def write_reference_case(
    trajectory,
    output_dir,
    *,
    epoch,
    case_name: str = "ssatk_reference_case",
    center_name: str = "Earth",
    reference_frame: str = "GCRF",
    time_system: str = "UTC",
    force_models: Iterable[str] = (),
    constants: Mapping[str, object] | None = None,
    integrator: Mapping[str, object] | None = None,
    precision: int = 17,
    overwrite: bool = False,
) -> ReferenceCaseFiles:
    """Write a trajectory as CCSDS OEM 2.0 plus a JSON case description.

    ``trajectory`` must expose ``t``, ``r``, and ``v`` arrays. Times are
    interpreted as seconds from the first sample; OEM positions and velocities
    are written in km and km/s, while the JSON sidecar records SI states too.
    The sidecar is intentionally explicit so another propagator can reproduce
    the case without inferring frames, constants, or tolerances from a plot.
    """

    times, positions, velocities = _trajectory_arrays(trajectory)
    epoch_utc = _epoch_utc(epoch)
    case_name = _case_name(case_name)
    if not isinstance(time_system, str) or time_system.upper() != "UTC":
        raise ValueError("write_reference_case currently supports only UTC.")
    if not isinstance(reference_frame, str) or not reference_frame.strip():
        raise ValueError("reference_frame must be a non-empty string.")
    if not isinstance(center_name, str) or not center_name.strip():
        raise ValueError("center_name must be a non-empty string.")
    if not isinstance(precision, int) or not 1 <= precision <= 17:
        raise ValueError("precision must be an integer from 1 through 17.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"{case_name}.json"
    ephemeris_path = output_dir / f"{case_name}.oem"
    if not overwrite and (metadata_path.exists() or ephemeris_path.exists()):
        raise FileExistsError(f"reference case already exists: {case_name}")

    elapsed = times - times[0]
    models = [force_models] if isinstance(force_models, str) else [str(model) for model in force_models]
    metadata = {
        "format": "ssatk-reference-case",
        "schema_version": 1,
        "case_name": case_name,
        "epoch": _format_epoch(epoch_utc),
        "time_system": "UTC",
        "center_name": center_name,
        "reference_frame": reference_frame,
        "state_units": {"time": "s", "position": "m", "velocity": "m/s"},
        "state_order": ["x", "y", "z", "vx", "vy", "vz"],
        "time_origin_s": float(times[0]),
        "time_span_s": float(elapsed[-1]),
        "sample_count": int(times.size),
        "force_models": models,
        "constants": dict(constants or {}),
        "integrator": dict(integrator or {}),
        "numeric_precision_digits": precision,
        "initial_state": np.concatenate((positions[0], velocities[0])).tolist(),
        "final_state": np.concatenate((positions[-1], velocities[-1])).tolist(),
        "files": {"ephemeris": ephemeris_path.name},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ephemeris_path.write_text(
        _oem_text(
            case_name,
            epoch_utc,
            elapsed,
            positions,
            velocities,
            center_name=center_name,
            reference_frame=reference_frame,
            precision=precision,
        ),
        encoding="utf-8",
    )
    return ReferenceCaseFiles(metadata_path, ephemeris_path)


def _read_oem_records(path):
    records = []
    in_data = False
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text == "DATA_START":
            in_data = True
            continue
        if text == "DATA_STOP":
            in_data = False
            continue
        if not in_data or not text or text.startswith("#"):
            continue
        fields = text.split()
        if len(fields) != 7:
            raise ValueError(f"invalid CCSDS OEM state record: {text!r}")
        try:
            epoch = _epoch_utc(fields[0])
            state = np.asarray(fields[1:], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid CCSDS OEM state record: {text!r}") from exc
        if not np.all(np.isfinite(state)):
            raise ValueError("CCSDS OEM state records must be finite.")
        records.append((epoch, state))
    if not records:
        raise ValueError("CCSDS OEM contains no state records.")
    return [item[0] for item in records], np.vstack([item[1] for item in records])


def _trajectory_arrays(trajectory):
    try:
        times = np.asarray(trajectory.t, dtype=float)
        positions = np.asarray(trajectory.r, dtype=float)
        velocities = np.asarray(trajectory.v, dtype=float)
    except AttributeError as exc:
        raise TypeError("trajectory must expose t, r, and v arrays.") from exc
    if times.ndim != 1 or times.size == 0:
        raise ValueError("trajectory.t must be a non-empty 1-D array.")
    if positions.shape != (times.size, 3) or velocities.shape != (times.size, 3):
        raise ValueError("trajectory r and v arrays must each have shape (N, 3).")
    if not np.all(np.isfinite(np.concatenate((times, positions.ravel(), velocities.ravel())))):
        raise ValueError("trajectory arrays must contain only finite values.")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("trajectory.t must be strictly increasing.")
    return times, positions, velocities


def _epoch_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    else:
        raise TypeError("epoch must be an ISO-8601 string or timezone-aware datetime.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("epoch must include a timezone, such as the UTC suffix 'Z'.")
    return parsed.astimezone(timezone.utc)


def _format_epoch(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _case_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("case_name must contain only letters, digits, '_', '.', or '-'.")
    return value


def _number(value: float, precision: int) -> str:
    return f"{float(value):.{precision}e}"


def _oem_text(
    case_name,
    epoch,
    elapsed,
    positions,
    velocities,
    *,
    center_name,
    reference_frame,
    precision,
) -> str:
    start = _format_epoch(epoch)
    stop = _format_epoch(epoch + timedelta(seconds=float(elapsed[-1])))
    lines = [
        "CCSDS_OEM_VERS = 2.0",
        f"CREATION_DATE = {start}",
        "ORIGINATOR = SSATK",
        "META_START",
        f"OBJECT_NAME = {case_name}",
        "OBJECT_ID = UNKNOWN",
        f"CENTER_NAME = {center_name}",
        f"REF_FRAME = {reference_frame}",
        "TIME_SYSTEM = UTC",
        f"START_TIME = {start}",
        f"USEABLE_START_TIME = {start}",
        f"USEABLE_STOP_TIME = {stop}",
        "META_STOP",
        "DATA_START",
        "# EPOCH X Y Z X_DOT Y_DOT Z_DOT (km, km/s)",
    ]
    for offset, position, velocity in zip(elapsed, positions, velocities):
        sample_epoch = _format_epoch(epoch + timedelta(seconds=float(offset)))
        state = [_number(value * 1.0e-3, precision) for value in (*position, *velocity)]
        lines.append(" ".join([sample_epoch, *state]))
    lines.extend(("DATA_STOP", ""))
    return "\n".join(lines)
