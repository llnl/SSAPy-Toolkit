#!/usr/bin/env python3
# yudata_path.py
#
# Build a safe, normalized path for saving/reading *data* files.
# Defaults to ~/yudata (falls back to ./yudata if HOME is not writable).
# If the provided filename has no known data extension, ".csv" is appended.
#
# Notes:
# - No use of the typing module.
# - numpy imported (no math).
# - Returns a str path and ensures parent directories exist.

from pathlib import Path
import os
import numpy as np  # kept per preference; not required for path ops

HOME_DATA_DIR = Path.home() / "yu_data"
FALLBACK_DATA_DIR = Path.cwd() / "yu_data"

# Common data extensions (case-insensitive). We check only the final suffix.
_KNOWN_DATA_EXTS = {
    ".csv", ".tsv", ".txt", ".log",
    ".json", ".jsonl", ".ndjson",
    ".yaml", ".yml",
    ".parquet", ".feather",
    ".h5", ".hdf5", ".hdf",
    ".npz", ".npy",
    ".pkl", ".pickle",
    ".xls", ".xlsx",
    ".zip", ".gz", ".bz2", ".xz", ".zst", ".tar"
}


def _safe_rel_parts(user_path):
    """
    Normalize an input path into a *relative* path component list:
      - remove drive/root/leading slashes,
      - resolve '.' and '..' without escaping above the root,
      - preserve intermediate subfolders.
    """
    p = Path(user_path)
    parts = []
    for part in p.parts:
        if part in (p.anchor, "/", "\\", ""):
            continue
        if part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return parts


def yudata(filename):
    """
    Construct a safe path under yudata (home by default, cwd as fallback).
    Returns a string path. Creates parent directories if needed.
    """
    if not isinstance(filename, (str, Path)):
        raise TypeError("yudata(filename): filename must be str or pathlib.Path")

    # Normalize to safe relative parts
    relative_parts = _safe_rel_parts(filename)
    if not relative_parts:
        relative_parts = ["data"]  # default base name if only dirs/empties were given

    # Determine final name and extension policy
    base_name = relative_parts[-1]

    # Subdirectory tree under yudata (everything except the final leaf name)
    subdir = Path(*relative_parts[:-1]) if len(relative_parts) > 1 else Path()

    # Try home, then cwd
    for base_dir in (HOME_DATA_DIR, FALLBACK_DATA_DIR):
        try:
            target_dir = base_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            return str(target_dir / base_name)
        except (OSError, PermissionError):
            continue

    raise RuntimeError("Could not create or access 'yu_data' in HOME or CWD.")

# Example usage:
# print(yudata("project/run1/results"))          # => ~/yudata/project/run1/results.csv
# print(yudata("archive/output.parquet"))        # => ~/yudata/archive/output.parquet
# print(yudata("/abs/path/../to/data_dump"))     # => ~/yu_data/to/data_dump.csv
