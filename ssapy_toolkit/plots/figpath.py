import os
from pathlib import Path

from ssapy_toolkit._paths import safe_relative_parts

DEFAULT_FIG_DIR_NAME = "ssatk_figures"
SSATK_FIGURES_ENV = "SSATK_FIGURES_DIR"
HOME_FIG_DIR = Path.home() / DEFAULT_FIG_DIR_NAME

__all__ = ["ssatk_path", "figpath"]

# You can keep this around if you like, but it's no longer used for extension logic.
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


def ssatk_path(filename="figure"):
    """
    Build a path under the SSATK figure directory.

    Rules:
      - The path is rooted under ~/ssatk_figures by default.
      - Set SSATK_FIGURES_DIR to choose an explicit alternate root.
      - Subfolders in `filename` are preserved and created as needed.
      - The basename is used exactly as given (no automatic extension added).
      - Absolute paths and '..' are normalized to stay under the output root.

    Examples
    --------
    ssatk_path("plot")                          -> ~/ssatk_figures/plot
    ssatk_path("demo_gallery/figures/burn_to_dv")              -> ~/ssatk_figures/demo_gallery/figures/burn_to_dv
    ssatk_path("demo_gallery/figures/burn_to_dv.png")          -> ~/ssatk_figures/demo_gallery/figures/burn_to_dv.png
    ssatk_path("/abs/path/ignored/name.svg")    -> ~/ssatk_figures/abs/path/ignored/name.svg
    ssatk_path("weird/name.foo")                -> ~/ssatk_figures/weird/name.foo
    """
    if not isinstance(filename, (str, Path)):
        raise TypeError("ssatk_path(filename): filename must be str or pathlib.Path")

    # Normalize to a safe relative path (no drive, no leading slash, no traversal)
    rel_parts = safe_relative_parts(filename)
    if not rel_parts:
        rel_parts = ["figure"]

    # Use the basename exactly as provided (no auto extension)
    basename = rel_parts[-1]
    final_name = basename

    base = _figure_root()
    try:
        # Construct full subdir path and ensure it exists
        subdir = Path(*rel_parts[:-1]) if len(rel_parts) > 1 else Path()
        target_dir = base / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        return str(target_dir / final_name)
    except (OSError, PermissionError) as exc:
        raise RuntimeError(
            f"Could not create or access {base}. Set {SSATK_FIGURES_ENV} "
            "to an explicit writable output directory."
        ) from exc


def _figure_root():
    override = os.environ.get(SSATK_FIGURES_ENV)
    if override:
        return Path(override).expanduser()
    return HOME_FIG_DIR


figpath = ssatk_path
