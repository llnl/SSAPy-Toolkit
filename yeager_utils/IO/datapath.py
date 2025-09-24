import os
from pathlib import Path

ENV_VAR = "YEAGER_UTILS_DATA_DIRS"   # colon-separated list of dirs
CONFIG_FILE = Path.home() / ".config" / "yeager_utils" / "datadirs.txt"


def _candidate_dirs(user_dirs=None):
    # 1) explicit arg wins
    if user_dirs:
        return [Path(p).expanduser() for p in user_dirs]

    # 2) env var (colon-separated on all OSes)
    env = os.environ.get(ENV_VAR, "")
    if env.strip():
        return [Path(p).expanduser() for p in env.split(":") if p.strip()]

    # 3) config file: one path per line, comments with #
    if CONFIG_FILE.exists():
        try:
            lines = CONFIG_FILE.read_text(encoding="utf-8").splitlines()
            paths = []
            for line in lines:
                line = line.split("#", 1)[0].strip()
                if line:
                    paths.append(Path(line).expanduser())
            if paths:
                return paths
        except Exception:
            pass  # fall through to defaults

    # 4) sensible defaults (no personal paths!)
    defaults = [
        Path.home() / "Data",
        Path.cwd() / "data",
    ]
    return defaults


def datapath(filename, dirs=None, must_exist=False):
    """
    Returns a target file path for `filename` in the first accessible data directory.

    Search order:
      1) `dirs` argument (list of paths)
      2) env var YEAGER_UTILS_DATA_DIRS (colon-separated list)
      3) ~/.config/yeager_utils/datadirs.txt (one path per line, '#' comments OK)
      4) Defaults: ~/Data, ./data

    If `must_exist=True`, only returns a path if the file already exists.
    Otherwise, it will create the chosen directory if missing.
    """
    for d in _candidate_dirs(dirs):
        try:
            d.mkdir(parents=True, exist_ok=True)
            target = d / filename
            if must_exist and not target.exists():
                continue
            return str(target)
        except (OSError, PermissionError):
            continue

    # Do not reveal private paths; keep the error generic.
    raise RuntimeError("Could not create or access any configured data directory.")
