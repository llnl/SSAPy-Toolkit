# Contributing to SSAPy Toolkit

All contributions to SSAPy Toolkit must be made under the BSD 3-Clause License.

Contributions are welcome via pull request targeting the `main` branch of the
[SSAPy Toolkit](https://github.com/llnl/SSAPy-Toolkit) repository. Pull requests
are reviewed by the project maintainers and must pass the repository's CI checks
(tests and linting) before they can be merged.

Work that primarily concerns the core propagation and modeling engine should be
contributed to the [SSAPy](https://github.com/llnl/SSAPy) repository instead;
SSAPy Toolkit is for higher-level, analysis-ready utilities and workflows built
on top of SSAPy.

By contributing, you agree to abide by the project's
[Code of Conduct](CODE_OF_CONDUCT.md).

## Repository scope and file layout

SSAPy Toolkit is a source-code repository. Keep generated artifacts, analysis
outputs, downloaded data products, figures, screenshots, animations, notebooks
with embedded outputs, and other binary media out of the repository. If a change
requires persistent data, put that data in
[SSAPy-Data](https://github.com/llnl/SSAPy-Data) or document an external
download/source instead of committing it here.

Use the existing top-level structure for new work:

| Path | Purpose |
| --- | --- |
| `ssapy_toolkit/` | Importable package code. |
| `tests/` | Automated regression and behavior tests. |
| `demos/` | Runnable, lightweight demonstrations of user-facing workflows. |
| `docs/` | Narrative documentation and API documentation. |
| `scripts/` | Maintainer/development utilities, not importable package code. |
| `.github/` | GitHub Actions, issue/PR templates, and ownership policy. |

Do not introduce new top-level directories or project-level configuration files
unless the pull request explains why the existing layout cannot support the
change and updates the repository policy check if needed.

## Tests and demos

Pull requests that change package behavior must include automated tests under
`tests/`. Add or update tests for bug fixes, new functions, changed defaults,
and changed error handling. Documentation-only and CI-only changes do not need
new tests, but the pull request should state that explicitly.

New user-facing workflows, plotting utilities, data-ingest utilities, command
line behavior, or analysis recipes must also add or update a runnable demo under
`demos/`. Demos should be small Python or Markdown examples that generate their
outputs locally. Do not commit generated demo outputs; the gallery workflow
builds media artifacts from source during CI.

Before requesting review, run the focused checks that match the change:

```bash
pip install -e .[dev]
pytest tests
python -m ssapy_toolkit.run_all_demos --no-open
python scripts/check_repository_policy.py
```

## Review and merge requirements

Each pull request must:

- Preserve the repository layout described above.
- Include tests for package-code changes or explain why no behavior changed.
- Include a demo update for new user-facing workflows or explain why no demo is
  needed.
- Avoid committed data, generated outputs, images, binary media, and large local
  artifacts.
- Pass the GitHub Actions checks before merge.

The repository policy workflow blocks pull requests that add disallowed file
types, binary files, large files, or unexpected top-level paths. If a blocked
file is genuinely required, prefer moving it to SSAPy-Data. If a repository
policy exception is still necessary, make the exception explicit in the pull
request and update `scripts/check_repository_policy.py` in the same change.
