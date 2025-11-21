from pathlib import Path

HOME_FIG_DIR = Path.home() / "yu_figures"
FALLBACK_DIR = Path.cwd() / "yu_figures"

# Known extensions treated as real output formats (case-insensitive)
_KNOWN_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif",
    ".svg", ".svgz", ".pdf", ".ps", ".eps",
    ".mp4", ".mov", ".avi", ".mpeg", ".mpg", ".webm",
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

def _safe_rel_parts(p: Path):
    """
    Normalize a user path into a safe relative path:
      - strip any drive / root / leading slashes,
      - collapse '.' and '..' without allowing traversal outside the root,
      - keep intermediate subfolders intact.
    """
    parts = []
    for part in p.parts:
        # Skip anchors/roots (e.g., 'C:\\', '/', '//server')
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

def figpath(filename):
    """
    Build a path under yu_figures that *respects requested subfolders*.

    Rules:
      - The path is always rooted under ~/yu_figures (fallback: ./yu_figures).
      - Subfolders in `filename` are preserved and created as needed.
      - The final extension is honored only if it matches a known output type.
        Otherwise, the name is treated as a stem and '.jpg' is appended.
      - Absolute paths and '..' are normalized to stay under yu_figures.

    Examples
    --------
    figpath("plot")                          -> ~/yu_figures/plot.jpg
    figpath("tests/burn_to_dv")              -> ~/yu_figures/tests/burn_to_dv.jpg
    figpath("tests/burn_to_dv.png")          -> ~/yu_figures/tests/burn_to_dv.png
    figpath("/abs/path/ignored/name.svg")    -> ~/yu_figures/name.svg
    figpath("weird/name.foo")                -> ~/yu_figures/weird/name.foo.jpg
    """
    if not isinstance(filename, (str, Path)):
        raise TypeError("figpath(filename): filename must be str or pathlib.Path")

    # Normalize to a safe relative path (no drive, no leading slash, no traversal)
    user_p = Path(filename)
    rel_parts = _safe_rel_parts(user_p)
    if not rel_parts:
        rel_parts = ["figure"]

    # Determine basename + extension policy
    basename = rel_parts[-1]
    suffix = Path(basename).suffix  # last dot segment on the filename
    suffix_lc = suffix.lower()

    if suffix and suffix_lc in _KNOWN_EXTS:
        final_name = basename  # keep caller's extension (preserve case)
    else:
        # treat entire basename (including dots) as stem; append default
        final_name = f"{basename}.jpg"

    # Choose base directory (home first, then CWD)
    for base in (HOME_FIG_DIR, FALLBACK_DIR):
        try:
            # Construct full subdir path and ensure it exists
            subdir = Path(*rel_parts[:-1]) if len(rel_parts) > 1 else Path()
            target_dir = base / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            return str(target_dir / final_name)
        except (OSError, PermissionError):
            continue

    raise RuntimeError("Could not create or access yu_figures in HOME or CWD.")
