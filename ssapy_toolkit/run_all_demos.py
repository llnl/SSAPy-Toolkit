from __future__ import annotations
import argparse
import webbrowser
from pathlib import Path
from ssapy_toolkit.plots.figpath import figpath
from .demo_gallery import run_all_demos

def default_output_dir() -> Path:
    return Path(figpath("demo_gallery/index.html")).expanduser().resolve().parent

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all SSAPy demos and build a local gallery report."
    )
    parser.add_argument(
        "--demos-dir",
        default="demos",
        help="Directory containing demo scripts (default: ./demos)",
    )
    parser.add_argument(
        "--output",
        default=str(default_output_dir()),
        help="Output directory for generated demo artifacts and report "
             "(default: whatever figpath uses for demo_errors)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML report in a browser after it is written",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the existing output directory before running",
    )
    args = parser.parse_args()

    demos_dir = Path(args.demos_dir).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not demos_dir.exists():
        raise SystemExit(f"Demo directory not found: {demos_dir}")

    results = run_all_demos(
        demos_dir=demos_dir,
        output_root=output_root,
        clean=not args.no_clean,
    )

    success = sum(r.status == "success" for r in results)
    failed = sum(r.status == "failed" for r in results)
    report = output_root / "index.html"

    print()
    print("Demo gallery complete")
    print(f"  demos  : {demos_dir}")
    print(f"  output : {output_root}")
    print(f"  report : {report}")
    print(f"  success: {success}")
    print(f"  failed : {failed}")

    # report is already written by run_all_demos before we get here
    if args.open and report.exists():
        try:
            webbrowser.open(report.as_uri())
        except Exception:
            pass

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
