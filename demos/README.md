# Demo Gallery

> Run all demos locally and build a presentation-style gallery.

---

## Running from a cloned repo

From the top of the repository:

```bash
python -m ssapy_toolkit.run_all_demos
```

If you installed the repo in editable mode, you can also use the console script:

```bash
ssapy-demo-gallery
```

Example with explicit paths:

```bash
python -m ssapy_toolkit.run_all_demos --demos-dir demos --output ./demo_gallery_output
```

---

## Running after `pip install ssapy_toolkit`

If `ssapy_toolkit` is installed and the console entry point is available, the
command auto-detects the packaged demos and can be run from any directory:

```bash
ssapy-demo-gallery
```

Or run it as a module:

```bash
python -m ssapy_toolkit.run_all_demos
```

---

## Useful options

| Flag | Description |
|------|-------------|
| `--open` | Open the generated report after it is written |
| `--no-clean` | Don't clean the output directory before running |
| `--output ./demo_gallery_output` | Write results to a custom output directory |
| `--demos-dir demos` | Specify the demos directory explicitly |

---

## Output

The gallery runner creates an **HTML report** at
`~/ssatk_figures/demo_gallery/index.html` by default and saves generated demo
artifacts beside it in the selected output directory. The default output root is
the user's home directory, not the source checkout; set `SSATK_FIGURES_DIR` or
pass `--output` for an explicit alternate location.

---

## Demo vs. test policy

- Gallery demos should create useful artifacts: figures, animations, logs, or
  other files that help a user understand the workflow.
- Validation-only examples should set `GALLERY_INCLUDE = False` and be covered
  by `pytest` instead of appearing in the default gallery.
- Duplicate gallery demos should be merged into the clearest user workflow; any
  deleted behavior should move into direct tests when it improves coverage.
- Optional-data demos should skip gracefully or report actionable missing-data
  messages when a package, local cache, or network source is unavailable.
