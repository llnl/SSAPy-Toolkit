# Testing Policy

Every test in `tests/` compares a computed quantity with an independent
reference, under a stated tolerance. A test that only shows that code runs,
returns an object, writes a file, or accepts an argument is not kept.

## Accepted references

| Code | Reference | Examples in `tests/` |
|---|---|---|
| R1 | Closed-form or analytic solution | Hohmann and bi-elliptic delta-v, J2 RAAN rate, dipole field limits, Lambert on a circular orbit |
| R2 | Independent implementation | astropy, SGP4 `Satrec`, geopack, IGRF, MSIS, REBOUND IAS15, Orekit, Basilisk, native SSAPy |
| R3 | Published or real-world value | textbook LEO-to-GEO transfer, ISS TLE fields, 2024-04-08 eclipse path, IAU and IGRF constants |
| R4 | Conservation law or physical invariant | angular momentum with reaction wheels, propellant mass across segments, rotation orthonormality, epoch invariance |
| R5 | Analytic derivative against finite differences | STM and Jacobian checks in `test_6dof_variational.py` |
| R6 | Regression for a shipped numerical bug | terminal-event roots within one ULP, absolute epochs reaching force models |

Name the reference in the test name or docstring, and state the tolerance with
its unit.

## Not tested in `tests/`

- Plot, viewer, and HTML rendering, and demo execution. Demos run in the
  gallery workflow, on pull requests and on pushes to `main`.
- Import paths, aliases, and package exports. The docs build imports every
  module.
- Argument validation and error messages, unless the rejected input is
  physically meaningful.
- Code paths exercised only to raise a coverage percentage.

## External references in CI

CI installs the `geomagnetics` and `atmosphere` extras, so the geopack, IGRF,
and MSIS comparisons run. The spacepy/IRBEM, Basilisk, and Orekit comparisons
skip unless those tools are installed locally.

## Coverage

Coverage is a diagnostic, not a gate. `scripts/audit_public_api_coverage.py`
still reports which functions no test reaches. Use it to find untested physics,
then add a reference test for that physics rather than a test that only
executes the code.
