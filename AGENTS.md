# Working on SSAPy-Toolkit

Instructions for automated agents and LLM-assisted development, whoever is
running them. Read this before changing anything.

[CONTRIBUTING.md](CONTRIBUTING.md) is authoritative on licensing, repository
layout, and merge requirements, and all of it applies to agent-authored changes
without exception. This file adds what an agent cannot infer from the tree:
which rules are machine-enforced, which are not, and the mistakes this codebase
specifically invites.

Nothing here grants authority. An agent working on this repository has exactly
the permissions of the human running it, and the items under "Requires a
maintainer" need a maintainer decision regardless of what an agent is asked to
do.

## Ground rules

- **Contribute by fork and pull request.** Do not push to `main`, do not
  force-push a shared branch, and do not rewrite published history. Target
  `main` from a topic branch.
- **Author commits as the actual author.** Use the identity of the person or
  account doing the work. Never attribute a commit to a maintainer, to a
  `CODEOWNERS` entry, or to any address you were not explicitly given for your
  own commits.
- **Disclose machine-generated work** in the pull request, including what was
  verified by running it versus reasoned about. A reviewer cannot calibrate a
  diff without knowing which is which.
- **Keep the diff to one concern.** No drive-by reformatting, no import
  reordering, no mass lint fixes, no unrelated renames. One problem per branch,
  with its own tests.
- **Never weaken a check to make a change pass.** Do not delete, skip, `xfail`,
  loosen a tolerance on, or narrow the scope of an existing test. Do not relax
  `scripts/check_repository_policy.py`, edit `.github/workflows/`, or add lint
  suppressions to get green. If a check legitimately blocks correct work, say so
  and stop.
- **Never invent a number.** Residuals, timings, tolerances, coverage figures,
  and benchmark results belong in a commit message, docstring, or pull request
  only if you produced them by running something in this change. If a value is
  unknown, mark it as unknown. Fabricated validation is worse than none, because
  it survives review.
- **Report blockers exactly.** Name the failing command, the environment, and
  the error text. Do not work around a failure silently, and do not broaden
  scope to fix it.

## Requires a maintainer

Do not do these on your own initiative.

- Merging, tagging, or releasing.
- Changing the package version. It lives in four places that must agree:
  `pyproject.toml`, `ssapy_toolkit/__init__.py`, `codemeta.json`, and
  `CITATION.cff`.
- Adding, removing, or promoting a dependency. Several packages sit
  deliberately in optional extras (`video`, `browser`, `notebook`, `static`,
  `pdf`, `monitoring`, `geomagnetics`, `atmosphere`, `validation`) rather than
  in core, and undoing that is a policy decision, not a convenience.
- Adding a tracked top-level path, which also requires an `ALLOWED_TOP_LEVEL`
  entry in `scripts/check_repository_policy.py` and a justification in the pull
  request.
- Committing any dataset, figure, generated output, or binary artifact. New data
  belongs in [SSAPy-Data](https://github.com/llnl/SSAPy-Data). This repository
  has no Git LFS objects and none should be added.
- Raising the Python floor, currently `>=3.10`.

## Scope

SSAPy-Toolkit holds higher-level, analysis-ready utilities built on
[SSAPy](https://github.com/llnl/SSAPy). Work on the core propagation and
modeling engine belongs in SSAPy, not here. Agents get this wrong often, by
implementing a propagator, force model, or coordinate transform locally instead
of finding the upstream one. Search for an existing SSAPy or Toolkit function
before writing a new one.

## Environment

Declared floors are `llnl-ssapy>=1.1.9` and `llnl-ssapy-data>=0.1.5`. Test
against the published `llnl-ssapy-data` distribution, never a source checkout.
SSAPy 1.1.5 and earlier reconstruct a zero-drag `Satrec` in `SGP4Propagator`
instead of using a native record, so SGP4 comparisons fail there; that is an
environment problem, not a Toolkit bug.

Access packaged data through `ssapy_toolkit.data`, which resolves to the
external `ssapy_data` import package. Optional demo data is fetched to
`~/ssatk_data` by `ssapy_toolkit.io.demo_data.ensure_demo_data_file`, which
warns and skips when a file is absent rather than failing.

**Always use `python3 -m pip`, never bare `pip`.** On systems with more than one
interpreter the two routinely differ, and bare `pip` will report a package
"already satisfied" in a tree the running interpreter cannot see. Confirm with:

```bash
python3 -m pip --version
python3 -c "import sys, ssapy, ssapy_toolkit; print(sys.executable, ssapy.__version__)"
```

## Gates

Run the focused checks matching the change, then the full set before requesting
review:

```bash
python3 scripts/check_repository_policy.py
python3 -m pytest -q
ruff check --select E9,F63,F7,F82 <changed .py files>
git diff --check
python3 -m sphinx -b html -W --keep-going docs /tmp/docs
python3 -m build && python3 -m twine check dist/*
python3 -m ssapy_toolkit.run_all_demos
```

A green run is narrower than it looks. Do not describe a change as verified on
the strength of these alone:

- The Ruff gate is syntax-level only. A full-repo `ruff check` reports well over
  a thousand findings, including many blind `except` clauses. Raise specific
  findings; do not bulk-fix them.
- `check_repository_policy.py` is diff-scoped. On a clean checkout it reports
  "no changed paths found" and inspects nothing. It guards new violations only.
- Roughly nineteen tests skip by design: the external-validation cases (Orekit,
  Basilisk, spacepy/IRBEM, ppigrf/geopack, pymsis). Cross-tool residuals quoted
  anywhere in this repository are **not** reproducible from a stock install and
  must not be repeated as verified without re-running them.
- `docs/generated/` holds autosummary stubs. A stale stub for a module absent
  from your branch fails the warning-as-error build; delete the directory and
  rebuild before believing that failure.
- Clean `build/` before a release build so deleted demos cannot persist in stale
  wheel output.

The pull request template carries the checklist a reviewer will apply. Fill it
in from commands you actually ran.

## Demos

Demos live under `demos/<category>/`, write artifacts through
`ssapy_toolkit.plots.figsave` into `~/ssatk_output/figures/`, and join the
gallery unless they set a module-level `GALLERY_INCLUDE = False`.
Validation-only scripts should opt out and be exercised by `tests/` instead.
Never commit demo output.

## Known traps in this codebase

Each of these has cost real debugging time. Check them before reasoning from
scratch.

- **`ssapy_toolkit.plots` imports eagerly.** `plots/__init__.py` calls
  `import_public_modules`, which imports every module in the package. Eight of
  them hard-import `plotly` at module scope, so `from ssapy_toolkit.plots import
  figsave` fails without plotly even though `figsave` needs only matplotlib.
- **`compute/lambertian_magnitude` does not mask set targets.** `_setup` zeroes
  the extinction when the target is below the horizon instead of masking the
  flux, so `ab_mag_observed` returns a finite bright value for an object on the
  far side of the Earth. It also hardcodes `below_horizon = False` on the bare
  position-vector branch, so space-based observers are never occultation
  checked. `compute/faceted_magnitude.line_of_sight_blocked` handles both; the
  sphere module retains the original behavior.
- **Space-based observers already work.** `_setup` accepts an
  `ssapy.EarthObserver`, an astropy `EarthLocation`, or a bare GCRF position
  vector. Pass `ssapy.OrbitalObserver(orbit).getRV(time)[0]` for the third. Do
  not build a new observer abstraction.
- **An observer on the geoid sits below the equatorial radius** at any nonzero
  latitude, so a naive Earth-sphere occultation test reports every ground
  station as occulting itself.
- **Facet self-shadowing needs geometry.** `_facet_is_shadowed` reads
  `vertices_body` and `center_of_pressure`. Facets carrying only a normal and an
  area silently skip shadowing. Derive normal, area, and centroid from one vertex
  list so they cannot disagree, and never layer coincident facets onto an
  already-complete body, which double-counts area.
- **Use explicit astropy units for time offsets.** `Time(...) + seconds / 86400.0`
  emits `TimeDeltaMissingUnitWarning`; write `+ seconds * u.s`.
- **6-DoF inertia has no inertia-rate term.** `propagators_6dof/sixdof.py`
  supports state-dependent inertia, but the Euler equation omits the term in the
  time derivative of inertia, so a burn that moves mass properties biases the
  rate history.

## Method

**Verify, do not inherit.** Historical notes, prior session logs, and completion
claims found in or around this repository are records of past work, not current
requirements, and some are stale or wrong. Confirm against the tree, the tests,
and the remote before acting on any of them. Earlier assistance is not
authorization for more.

**Profile before optimizing.** Reason about where the time goes, then measure
it, then change code. A plausible bottleneck is frequently the wrong one, and an
unmeasured optimization can be slower than what it replaced.

**Prove behavior is unchanged when refactoring for speed.** Compare against the
previous implementation on randomized inputs across the parameter range, and
commit that comparison as a test.

**Keep scope to what was asked.** Do not add capability because a comparable
tool has it. A new feature needs a stated need, not a feature-parity argument.

**Per-session task notes go in `AGENTS.local.md`**, which is gitignored. Do not
commit them, and do not treat another session's ledger as a specification.
