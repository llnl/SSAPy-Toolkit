from pathlib import Path

HOME_FIG_DIR = Path.home() / "yu_figures"
FALLBACK_DIR = Path.cwd() / "yu_figures"

def figpath(filename, fmt=None, clean_conflicts=True):
    """
    Return a target file path for `filename` inside ~/Figures (fallback ./figures).

    Behavior
    --------
    - If `fmt` is None and `filename` has an extension, that extension is used.
    - If `fmt` is None and `filename` has no extension, default to ".jpg".
    - If `fmt` is provided, it overrides any extension in `filename`.

    Parameters
    ----------
    filename : str or Path
        Desired base name (directory part is ignored; file goes into the chosen fig dir).
    fmt : str or None
        Extension to use (with or without leading dot), e.g. ".png" or "png". Optional.
    clean_conflicts : bool
        If True, remove files in the chosen directory that share the same stem
        but a different extension.

    Returns
    -------
    str
        Path to the target file as a string.

    Raises
    ------
    RuntimeError
        If neither ~/Figures nor ./figures can be created or accessed.
    """
    filename = Path(filename)
    base_name = filename.stem

    if fmt is None:
        ext = filename.suffix if filename.suffix else ".jpg"
    else:
        ext = fmt if str(fmt).startswith(".") else f".{fmt}"

    for d in (HOME_FIG_DIR, FALLBACK_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
            target = d / f"{base_name}{ext}"

            if clean_conflicts:
                for f in d.iterdir():
                    if f.is_file() and f.stem == base_name and f.suffix != ext:
                        try:
                            f.unlink()
                        except Exception:
                            pass  # ignore failures cleaning unrelated files

            return str(target)
        except (OSError, PermissionError):
            continue

    raise RuntimeError("Could not create or access a figure directory (~/Figures or ./figures).")
