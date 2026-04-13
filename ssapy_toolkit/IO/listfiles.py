import os
import glob
import re


def _natural_key(s: str):
    """
    Split a string into a list of ints and lowercased text to enable natural sorting.
    Example: 'frame_10.png' -> ['frame_', 10, '.png']
    """
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", s)]


def list_files(
    *patterns: str,
    sort: bool = True,
) -> list[str]:
    """
    Expand path patterns (with shell-style wildcards) into a list of files.

    Examples
    --------
    list_files("path/to/some/files/filenames_*_numbers.jpg")
    list_files("frames_*.png", "extra_frames/*.png")
    list_files("root/**/frame_*.png")  # recursive with **

    Parameters
    ----------
    patterns : str
        One or more path patterns, may include *, ?, or **.
    sort : bool
        If True, apply natural sort (e.g. frame_2 < frame_10).

    Returns
    -------
    list[str]
        Expanded file paths as strings.
    """
    files: list[str] = []

    for pat in patterns:
        # Ensure string and expand ~
        pat = os.path.expanduser(str(pat))

        # glob supports ** with recursive=True
        matches = glob.glob(pat, recursive=True)
        files.extend(matches)

    # Deduplicate while preserving order
    files = list(dict.fromkeys(files))

    if sort and files:
        files.sort(key=_natural_key)

    return files
