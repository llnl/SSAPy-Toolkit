## Summary

-

## Repository Policy Checklist

- [ ] Changes preserve the existing file layout (`ssapy_toolkit/`, `tests/`, `demos/`, `docs/`, `scripts/`, `.github/`).
- [ ] Package-code changes include tests under `tests/`, or this PR only changes docs/CI/metadata.
- [ ] New user-facing workflows include a runnable demo under `demos/`, or no demo is needed because:
- [ ] No generated outputs, downloaded data, images, notebooks with embedded outputs, binary media, or large artifacts are committed.
- [ ] Persistent data needed by this work is stored in SSAPy-Data or documented as an external input.

## Validation

- [ ] `pytest tests`
- [ ] `python -m ssapy_toolkit.run_all_demos --no-open` when demos or user-facing workflows changed
- [ ] `python scripts/check_repository_policy.py`
