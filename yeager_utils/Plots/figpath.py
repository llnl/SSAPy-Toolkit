from pathlib import Path

HOME_FIG_DIR = Path.home() / "yu_figures"
FALLBACK_DIR = Path.cwd() / "yu_figures"

# Known extensions that should be treated as real file types
_KNOWN_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif",
    ".svg", ".svgz", ".pdf", ".ps", ".eps",
    ".mp4", ".mov", ".avi", ".mpeg", ".mpg", ".webm"
}

def figpath(filename):
    """
    Return a target file path for `filename` inside ~/yu_figures (fallback ./yu_figures).

    Rules:
      - Directory parts in `filename` are ignored; only the base name is used.
      - A trailing dot-segment is treated as an extension ONLY if it matches a known extension
        (case-insensitive). Otherwise, the dot stays in the name and '.jpg' is appended.
      - If no extension is present, default to '.jpg'.
    """
    # Use only the base name; ignore any directories the caller passed
    name = Path(filename).name

    p = Path(name)
    suffix = p.suffix  # last dot segment
    suffix_lc = suffix.lower()

    if suffix and suffix_lc in _KNOWN_EXTS:
        base_name = p.stem  # remove that real extension
        ext = suffix  # preserve caller's case
    else:
        base_name = name  # keep entire name (including any dots) as the stem
        ext = ".jpg"

    # Pick a writable directory
    for d in (HOME_FIG_DIR, FALLBACK_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
            target = d / f"{base_name}{ext}"
            return str(target)
        except (OSError, PermissionError):
            continue

    raise RuntimeError("Could not create or access a figure directory (~/yu_figures or ./yu_figures).")
