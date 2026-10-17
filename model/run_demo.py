#!/usr/bin/env python3
"""Capacity Allocator - generate synthetic data, solve, and write the three CSVs.

    python run_demo.py                            # every scenario, Gurobi, ../output
    python run_demo.py --scenario ib_surge
    python run_demo.py --out /tmp/x
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from capacity_allocator import config as cfg, run, scenarios

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "output"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scenario",
        nargs="+",
        default=["all"],
        metavar="NAME",
        help=f"one or more of: {', '.join(scenarios.ORDER)} (default: all)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="artifact directory")
    args = ap.parse_args(argv)

    print(f"Capacity Allocator - site {cfg.SITE}, {cfg.HORIZON_HOURS}h horizon")
    print(f"Output: {args.out}")
    print("All data is synthetic.\n")

    try:
        results = run.run_all(args.scenario, args.out)
    except RuntimeError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    written, preserved = run.write_data(results, args.out)

    nv, nc = results[0].after.backend.report_size()
    print()
    print(f"Model: {nv} variables, {nc} constraints, {len(results[0].ladder)} objective tiers")
    print(f"Data:      {', '.join(p.name for p in written)} in {args.out / 'data'}")
    if preserved:
        # These rows were not produced by this run; say so.
        print(
            f"           kept {len(preserved)} scenario(s) from a previous run: "
            f"{', '.join(sorted(preserved))}"
        )
    print("Present:   presentation.ipynb")

    unresolved = [r for r in results if not r.resolved]
    if unresolved:
        print()
        print("Scenarios the model could NOT fix, and the ceiling it reported:")
        for r in unresolved:
            print(f"  {r.scenario.name:18} {r.limit}")
        print("  (a diagnostic, not an attribution - ceilings interact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
