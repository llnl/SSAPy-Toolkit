"""
import_cost.py — where does `import ssapy_toolkit.plots` spend its time?

Runs the target import under -X importtime, then reports the most
expensive modules by self time and traces which of your modules is
responsible for pulling in a given dependency.

    python scripts/import_cost.py
    python scripts/import_cost.py --target ssapy_toolkit.plots --trace bqplot igrf

Self time is the module's own cost, excluding its children, so the
figures do not double count. The chain shows the actual import path, which
is the thing grep cannot tell you: a dependency can arrive through any
number of intermediaries.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

LINE = re.compile(r"^import time:\s+(\d+)\s*\|\s+(\d+)\s*\|(\s*)(\S.*)$")


def measure(target):
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", f"import {target}"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"import failed:\n{proc.stderr[-2000:]}")

    entries = []
    for line in proc.stderr.splitlines():
        m = LINE.match(line)
        if m:
            self_us, cum_us, indent, name = m.groups()
            entries.append(dict(self_us=int(self_us), cum_us=int(cum_us),
                                depth=len(indent) // 2, name=name.strip()))
    return entries, proc.stdout


def parent_chain(entries, index):
    """
    importtime prints post-order, so a module's parent is the next later
    line one level shallower.
    """
    chain = []
    depth = entries[index]["depth"]
    for e in entries[index + 1:]:
        if e["depth"] == depth - 1:
            chain.append(e["name"])
            depth -= 1
            if depth == 0:
                break
    return chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="ssapy_toolkit.plots")
    ap.add_argument("--trace", nargs="*", default=["bqplot", "igrf"],
                    help="substrings to trace back to their importer")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    entries, stdout = measure(args.target)
    total = max(e["cum_us"] for e in entries) / 1e6
    print(f"import {args.target}: {total:.2f} s, {len(entries)} modules\n")

    if stdout.strip():
        print("printed to stdout during import (a library should not do this):")
        for line in stdout.strip().splitlines()[:5]:
            print(f"    {line}")
        print()

    print(f"most expensive by SELF time (excludes children):")
    for e in sorted(entries, key=lambda e: -e["self_us"])[:args.top]:
        print(f"  {e['self_us'] / 1000:8.1f} ms  {e['name']}")

    for needle in args.trace:
        hits = [(i, e) for i, e in enumerate(entries)
                if needle.lower() in e["name"].lower()]
        print(f"\n--- '{needle}' ---")
        if not hits:
            print("  not imported")
            continue
        top = max(hits, key=lambda p: p[1]["cum_us"])
        i, e = top
        print(f"  {e['name']}: {e['cum_us'] / 1000:.0f} ms cumulative "
              f"({len(hits)} related modules)")
        chain = parent_chain(entries, i)
        if chain:
            print("  imported by: " + " <- ".join(chain[:6]))
        own = sum(x["self_us"] for _, x in hits) / 1000
        print(f"  self time across all '{needle}' modules: {own:.0f} ms")


if __name__ == "__main__":
    main()
