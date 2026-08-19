# API Coverage Audit

This branch adds an executable coverage audit for SSAPy-Toolkit functions before
PR review.

## Scope

The default audit treats the public package surface as:

- top-level functions in `ssapy_toolkit` modules whose names do not start with
  `_`;
- methods on public classes whose names do not start with `_`.

The exhaustive audit adds:

- private functions and methods whose names start with `_`;
- nested function definitions and closures;
- package-scoped branch coverage thresholds.

`if __name__ == "__main__"` self-test helpers are intentionally excluded.
Dependency-fallback definitions are audited only when the fallback branch is
actually defined in the current environment. This prevents the audit from
requiring dead fallback code when the primary implementation imports correctly.

## Validation Standard

Every audited function body must execute under coverage. Tests then check one
of:

- a numerical oracle from geometry, orbital mechanics, or coordinate algebra;
- a round-trip or norm-preservation invariant;
- a structural plot oracle, such as expected trace/artist types and counts;
- a controlled failure mode for optional heavy dependencies.

This is stronger than statement coverage alone:
`scripts/audit_public_api_coverage.py` verifies that each audited
function/method has at least one executable body line hit by the test run.

## Commands

```bash
python3 -m coverage erase
python3 -m coverage run --branch -m pytest -q
python3 -m coverage json -i -o /tmp/ssatk_branch_coverage.json
python3 scripts/audit_public_api_coverage.py \
  --coverage-json /tmp/ssatk_branch_coverage.json \
  --min-hit-pct 95 \
  --require-branch-data \
  --write-unhit /tmp/ssatk_public_functions_body_unhit.tsv
python3 scripts/audit_public_api_coverage.py \
  --coverage-json /tmp/ssatk_branch_coverage.json \
  --include-private \
  --include-nested \
  --min-hit-pct 90 \
  --min-branch-pct 65 \
  --require-branch-data \
  --write-unhit /tmp/ssatk_all_functions_unhit.tsv \
  --write-missing-branches /tmp/ssatk_missing_branches.tsv
```

Current audited result from this branch:

- `459 passed, 16 skipped`
- `public_functions=681`
- `body_hit=681`
- `body_unhit=0`
- `body_hit_pct=100.0`
- `all_functions_including_nested=1457`
- `all_functions_body_hit=1457`
- `all_functions_body_unhit=0`
- `all_functions_body_hit_pct=100.0`
- `package_branches=6666`
- `package_branch_hit_pct=73.6`

CI enforces the public audit at 95% body-hit and the exhaustive audit at 90%
body-hit / 65% branch-hit on Python 3.10. Those thresholds are intentionally
below the current measured baseline so normal line-number or dependency-version
drift does not create brittle failures.
