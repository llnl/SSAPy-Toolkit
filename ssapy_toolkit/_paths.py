"""Shared path normalization helpers."""

from __future__ import annotations

from pathlib import Path


def safe_relative_parts(path: str | Path) -> list[str]:
    """Normalize a user path into safe relative path components."""
    user_path = Path(path)
    parts: list[str] = []
    for part in user_path.parts:
        if part in (user_path.anchor, "/", "\\", ""):
            continue
        if part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return parts
