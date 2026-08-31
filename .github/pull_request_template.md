## Summary

-

## Repository Policy Checklist

- [ ] If any part of this change was produced with an automated agent or LLM assistance, [AGENTS.md](../AGENTS.md) was followed and this PR states what was verified by running it.
- [ ] Changes preserve the existing file layout (`ssapy_toolkit/`, `tests/`, `demos/`, `docs/`, `scripts/`, `.github/`).
- [ ] Package-code changes include tests under `tests/`, or this PR only changes docs/CI/metadata.
- [ ] New user-facing workflows include a runnable demo under `demos/`, or no demo is needed because:
- [ ] No generated outputs, downloaded data, images, notebooks with embedded outputs, binary media, or large artifacts are committed.
- [ ] Persistent data needed by this work is stored in SSAPy-Data or documented as an external input.

## Validation

- [ ] `pytest tests`
- [ ] `python -m ssapy_toolkit.run_all_demos --no-open` when demos or user-facing workflows changed
- [ ] `python scripts/check_repository_policy.py`
- [ ] No existing test, tolerance, workflow, or policy check was weakened to make this pass.
- [ ] Every number quoted in this PR came from a command I ran on this change.
