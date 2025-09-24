import os
from pathlib import Path

ENV_VAR = "YEAGER_UTILS_FIG_DIRS"   # colon-separated list of dirs
CONFIG_FILE = Path.home() / ".config" / "yeager_utils" / "figdirs.txt"


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
        Path.home() / "Figures",
        Path.home() / "Pictures",
        Path.cwd() / "figures",
    ]
    return defaults


def figpath(filename, fmt=".jpg", dirs=None, clean_conflicts=True):
    """
    Returns a target file path for `filename` in the first writable figure directory.

    Search order:
      1) `dirs` argument (list of paths)
      2) env var YEAGER_UTILS_FIG_DIRS (colon-separated list)
      3) ~/.config/yeager_utils/figdirs.txt (one path per line, '#' comments OK)
      4) Defaults: ~/Figures, ~/Pictures, ./figures

    It will create the chosen directory if missing. If clean_conflicts=True,
    it removes files that share the same base name but different extension.
    """
    base_name = Path(filename).stem  # strip any existing extension

    for d in _candidate_dirs(dirs):
        try:
            d.mkdir(parents=True, exist_ok=True)
            target = d / f"{base_name}{fmt}"

            if clean_conflicts:
                for f in d.iterdir():
                    if f.is_file() and f.stem == base_name and f.suffix != fmt:
                        try:
                            f.unlink()
                        except Exception:
                            pass  # ignore failure to remove unrelated files

            return str(target)
        except (OSError, PermissionError):
            continue

    # Do not reveal private paths; keep the error generic.
    raise RuntimeError("Could not create or access any configured figure directory.")
