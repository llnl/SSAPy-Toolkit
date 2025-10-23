#!/usr/bin/env python
# run_all_here.py
# Find and run all Python scripts in THIS file's directory, one by one.
# Exits immediately on the first failure.

import os
import sys
import argparse
import subprocess
from pathlib import Path

def find_scripts(base_dir, include_hidden, pattern):
    scripts = []
    for p in sorted(base_dir.glob(pattern)):
        name = p.name
        if not p.is_file():
            continue
        if name == "__init__.py":
            continue
        if name.startswith("_") and not include_hidden:
            continue
        scripts.append(p)
    return scripts

def main():
    parser = argparse.ArgumentParser(
        description="Run all Python scripts in this file's directory, sequentially. Quits on first error."
    )
    parser.add_argument(
        "--pattern",
        default="*.py",
        help="Glob pattern to match scripts (default: *.py).",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include files starting with an underscore.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-script timeout in seconds (default: no timeout).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would run without executing.",
    )

    args = parser.parse_args()

    this_file = Path(__file__).resolve()
    base_dir = this_file.parent

    scripts = find_scripts(
        base_dir=base_dir,
        include_hidden=args.include_hidden,
        pattern=args.pattern,
    )

    # Exclude this script itself
    scripts = [p for p in scripts if p.resolve() != this_file]

    if not scripts:
        print("No matching scripts found.")
        return 0

    if args.dry_run:
        print("Dry run. The following scripts would be executed in order:")
        for p in scripts:
            print(f" - {p.name}")
        return 0

    print(f"Found {len(scripts)} scripts in {base_dir}")
    print("-" * 60)

    try:
        for i, script in enumerate(scripts, 1):
            print(f"[{i}/{len(scripts)}] Running: {script.name}")
            try:
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    cwd=str(base_dir),
                    timeout=args.timeout,
                )
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 124  # conventional timeout code
                print(f"Timed out after {args.timeout} seconds.")

            print(f"Exit code: {code}")
            print("-" * 60)

            if code != 0:
                print(f"Failure detected. Stopping on {script.name} with exit code {code}.")
                return code

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        return 130  # 128 + SIGINT

    print("All scripts succeeded.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
