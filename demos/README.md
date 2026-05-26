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

If `ssapy_toolkit` is installed and the console entry point is available:

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
| `--no-open` | Don't automatically open the generated report |
| `--no-clean` | Don't clean the output directory before running |
| `--output ./demo_gallery_output` | Write results to a custom output directory |
| `--demos-dir demos` | Specify the demos directory explicitly |

---

## Output

The gallery runner creates an **HTML report** and saves generated demo artifacts in the selected output directory.
