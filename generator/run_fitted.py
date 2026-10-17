#!/usr/bin/env python3
"""Run the demo model on days drawn from the fitted distributions.

    python run_fitted.py                     # the 13 curated seeds
    python run_fitted.py --verified 25       # 25 days, unbiased
    python run_fitted.py --cases             # just the pinned cases
    python run_fitted.py --scan 200 --select # which constraints co-occur (dev step)
    python run_fitted.py --out ../output
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from capacity_allocator import config as cfg  # noqa: E402
from capacity_allocator import data, model, run as demo_run  # noqa: E402

from mock_inputs import generate, seeds, to_demo  # noqa: E402


def solve_day_parts(
    seed: int
) -> tuple[demo_run.Result, data.Params, demo_run.Solved]:
    """Solve one fitted day; return the `Result` plus its params and solved 'after' side."""
    t0 = time.time()
    md = generate.generate(seed=seed)
    p = data.build(to_demo.to_demo_frames(md))
    sides = {}
    for label, freeze in (("before", True), ("after", False)):
        backend = model.make_backend(f"fitted_{seed}_{label}")
        m = model.build(backend, p, freeze_moves=freeze)
        sides[label] = demo_run.Solved(
            backend, m, model.solve_lexicographic(backend, m.tiers)
        )
    result = demo_run.fitted_result(
        seed, p, sides["before"], sides["after"], time.time() - t0
    )
    return result, p, sides["after"]


def solve_day(seed: int) -> demo_run.Result:
    """Generate, convert and solve one fitted day as a `Result`."""
    return solve_day_parts(seed)[0]


def summarise(r: demo_run.Result) -> dict:
    """One row of `fitted_days.csv`."""
    seed = int(r.scenario.name.removeprefix("fitted_"))
    lad = r.ladder
    breached = lad[(lad["kind"] == "slack") & (lad["slack_actual"] > demo_run.EPS)]
    resolved = breached[breached["slack_optimal"] <= demo_run.EPS]
    top = breached.iloc[0] if len(breached) else None
    return {
        "seed": seed,
        "moves": r.moves_total,
        "limit": r.limit,
        "tiers_breached": len(breached),
        "tiers_resolved": len(resolved),
        "top_tier": top["tier"] if top is not None else None,
        "top_before": round(float(top["slack_actual"]), 1) if top is not None else 0.0,
        "top_after": round(float(top["slack_optimal"]), 1) if top is not None else 0.0,
        "limit_counts": Counter(r.hours["ob_capacity_limited_by"]),
        "ob_capacity_actual": float(r.hours["ob_capacity_actual_units"].sum()),
        "ob_capacity_optimal": float(r.hours["ob_capacity_optimal_units"].sum()),
    }


# --- Co-occurrence scan ---

def day_families(r: demo_run.Result) -> dict[str, set[str]]:
    """Constraint families the planned day breached, by tier kind."""
    return demo_run.breached_families(r.ladder, side="slack_actual")


def _partition_breaches(r: demo_run.Result, family: str) -> set[str]:
    """Shift partitions of one family that breached, a subset of {"now", "later"}."""
    lad = r.ladder
    hit = lad[(lad["slack_actual"] > demo_run.EPS)]
    found = set()
    for tier in hit["tier"]:
        if demo_run.family_of(tier) != family:
            continue
        for suffix in demo_run.SHIFT_SUFFIXES:
            if tier.endswith(suffix):
                found.add(suffix.lstrip("_"))
    return found


def scan(seed_list: list[int]) -> pd.DataFrame:
    """Solve every seed and record breached families, one row per day."""
    rows = []
    for i, seed in enumerate(seed_list, 1):
        r = solve_day(seed)
        fams = day_families(r)
        operational = fams.get("slack", set())
        rows.append(
            {
                "seed": seed,
                "moves": r.moves_total,
                "limit": r.limit,
                "n_operational": len(operational),
                "operational": "|".join(sorted(operational)),
                "guardrail": "|".join(sorted(fams.get("guardrail", set()))),
                "ob_bl_max_partitions": "|".join(sorted(_partition_breaches(r, "ob_bl_max"))),
                "ob_bl_min_partitions": "|".join(sorted(_partition_breaches(r, "ob_bl_min"))),
                "storage_partitions": "|".join(sorted(_partition_breaches(r, "storage"))),
                "cpt_partitions": "|".join(sorted(_partition_breaches(r, "cpt"))),
            }
        )
        if i % 25 == 0 or i == len(seed_list):
            print(f"  scanned {i}/{len(seed_list)} days")
    return pd.DataFrame(rows)


def cooccurrence(scanned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise family co-occurrence, and the families-per-day distribution."""
    per_day = [set(s.split("|")) - {""} for s in scanned["operational"]]
    families = sorted({f for day in per_day for f in day})

    pairs = []
    for i, a in enumerate(families):
        for b in families[i + 1:]:
            both = sum(1 for day in per_day if a in day and b in day)
            pairs.append(
                {
                    "family_a": a,
                    "family_b": b,
                    "days_both": both,
                    "days_a": sum(1 for day in per_day if a in day),
                    "days_b": sum(1 for day in per_day if b in day),
                    "share_both": round(both / max(len(per_day), 1), 4),
                }
            )
    pair_table = pd.DataFrame(pairs).sort_values(
        ["days_both", "family_a"], ascending=[False, True]
    ).reset_index(drop=True)

    spread = (
        scanned["n_operational"].value_counts().sort_index()
        .rename_axis("families_breached").reset_index(name="days")
    )
    spread["share"] = (spread["days"] / max(len(scanned), 1)).round(4)
    return pair_table, spread


# --- Pattern selection ---

PATTERNS = {
    "ob_low_ib_high": (
        "outbound backlog below its band while inbound storage is high - the two pull "
        "staffing in opposite directions"
    ),
    "late_only_eod": (
        "end-of-day target missed and outbound backlog high LATER but not NOW - fine "
        "early, wrong by the end of the day"
    ),
    "broad": "three or more distinct constraint families breached in one day",
}


def matches(scanned: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Scan rows matching a named pattern, best first."""
    fams = scanned["operational"].str.split("|")

    if pattern == "ob_low_ib_high":
        hit = scanned[fams.apply(lambda f: "ob_bl_min" in f and "storage" in f)]
        return hit.sort_values("n_operational", ascending=False)

    if pattern == "late_only_eod":
        hit = scanned[
            fams.apply(lambda f: "eod_max" in f and "ob_bl_max" in f)
            & (scanned["ob_bl_max_partitions"] == "later")
        ]
        return hit.sort_values("n_operational", ascending=False)

    if pattern == "broad":
        return scanned[scanned["n_operational"] >= 3].sort_values(
            "n_operational", ascending=False
        )

    raise SystemExit(f"unknown pattern {pattern!r}; choose from {', '.join(PATTERNS)}")


def closest(scanned: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Nearest misses, reported instead of (never as) a match."""
    fams = scanned["operational"].str.split("|")
    if pattern == "ob_low_ib_high":
        score = fams.apply(lambda f: ("ob_bl_min" in f) + ("storage" in f))
    elif pattern == "late_only_eod":
        score = (
            fams.apply(lambda f: ("eod_max" in f) + ("ob_bl_max" in f))
            + (scanned["ob_bl_max_partitions"] == "later").astype(int)
        )
    else:
        score = scanned["n_operational"]
    return scanned.assign(_score=score).sort_values(
        ["_score", "n_operational"], ascending=False
    ).drop(columns="_score")


# --- Pinned cases ---

CASES_PATH = Path(__file__).resolve().parent.parent / "model" / "fitted_cases.toml"


def _read_toml(path: Path) -> dict:
    """Parse TOML; `tomllib` is stdlib only from Python 3.11, so fall back to `tomli` / `toml`."""
    for name in ("tomllib", "tomli"):
        try:
            mod = __import__(name)
        except ImportError:
            continue
        with path.open("rb") as f:
            return mod.load(f)
    try:
        import toml
    except ImportError as exc:
        raise SystemExit(
            f"cannot read {path.name}: this interpreter has no TOML parser. It is "
            f"stdlib (`tomllib`) from Python 3.11; on older ones `pip install tomli`.\n"
            f"  interpreter: {sys.executable}"
        ) from exc
    return toml.loads(path.read_text())


def load_cases() -> list[dict]:
    """The pinned multi-constraint days, in manifest order."""
    if not CASES_PATH.exists():
        raise SystemExit(
            f"no case manifest at {CASES_PATH}. Run `--scan N --select` first, then "
            f"record the chosen seeds there."
        )
    cases = _read_toml(CASES_PATH).get("case", [])
    if not cases:
        raise SystemExit(f"{CASES_PATH} has no [[case]] entries")
    return cases


def _scan_mode(args) -> int:
    """`--scan N`, optionally with `--select`."""
    lo, hi = seeds.VERIFIED_RANGE
    seed_list = list(range(lo, min(lo + args.scan, hi)))

    print(f"Co-occurrence scan: {len(seed_list)} days from the verified range {lo}-{hi}")
    print("Reporting what co-occurs, not searching for a hoped-for pattern.")
    print()
    scanned = scan(seed_list)
    pair_table, spread = cooccurrence(scanned)

    print("\n--- distinct OPERATIONAL constraint families breached per day ---")
    print(f"  {'families':>10} {'days':>6} {'share':>8}")
    for row in spread.itertuples():
        print(f"  {row.families_breached:>10} {row.days:>6} {row.share:>7.1%}")
    print(f"  mean {scanned['n_operational'].mean():.2f} families per day")

    print("\n--- which families co-occur (operational only, top 12 pairs) ---")
    print(f"  {'family A':<16} {'family B':<16} {'both':>6} {'A':>5} {'B':>5} {'share':>8}")
    for row in pair_table.head(12).itertuples():
        print(
            f"  {row.family_a:<16} {row.family_b:<16} {row.days_both:>6} "
            f"{row.days_a:>5} {row.days_b:>5} {row.share_both:>7.1%}"
        )

    print("\n--- the two bottom guardrails, for contrast ---")
    print("  Not in the pair table: a guardrail measures distance from a target INSIDE")
    print("  the constraint, so it is non-zero on almost every real day and would pair")
    print("  with everything at the other family's own rate.")
    for fam in ("ob_bl_centre", "storage_buffer"):
        n = scanned["guardrail"].str.split("|").apply(lambda f, k=fam: k in f).sum()
        print(f"  {fam:<16} non-zero on {n:>4} / {len(scanned)} days  ({n / len(scanned):.1%})")

    if args.select:
        _select_mode(scanned)

    if args.out:
        (args.out / "data").mkdir(parents=True, exist_ok=True)
        path = args.out / "data" / "fitted_cooccurrence.csv"
        pair_table.assign(days_scanned=len(scanned)).to_csv(path, index=False)
        spread_path = args.out / "data" / "fitted_families_per_day.csv"
        spread.assign(days_scanned=len(scanned)).to_csv(spread_path, index=False)
        print(f"\nWrote {path.name} and {spread_path.name} to {args.out / 'data'}")
    return 0


def _print_tier_slack(seed: int) -> None:
    """Per-tier before/after slack for one day."""
    r = solve_day(seed)
    lad = r.ladder
    interesting = lad[(lad["slack_actual"] > demo_run.EPS) | (lad["slack_optimal"] > demo_run.EPS)]
    print(f"      {'tier':<22} {'kind':<10} {'before':>12} {'after':>12}  outcome")
    for t in interesting.itertuples():
        print(
            f"      {t.tier:<22} {t.kind:<10} {t.slack_actual:>12,.0f} "
            f"{t.slack_optimal:>12,.0f}  {t.outcome}"
        )


def _select_mode(scanned: pd.DataFrame) -> None:
    """Report the best seeds per pattern, or that none matched."""
    print("\n=== pattern selection ===")
    for pattern, why in PATTERNS.items():
        hit = matches(scanned, pattern)
        print(f"\n--- {pattern}: {why}")
        if hit.empty:
            near = closest(scanned, pattern).head(3)
            print(f"  NO MATCH in {len(scanned)} days.")
            print("  Closest available:")
            for row in near.itertuples():
                print(
                    f"    seed {row.seed}  {row.n_operational} families: "
                    f"{row.operational or '-'}"
                )
            continue
        print(f"  {len(hit)} / {len(scanned)} days match. Best first:")
        for row in hit.head(5).itertuples():
            print(
                f"    seed {row.seed}  {row.n_operational} families: {row.operational}"
                f"   moves={row.moves}  limit={row.limit}"
            )
        print(f"\n    per-tier detail for the best match, seed {hit.iloc[0].seed}:")
        _print_tier_slack(int(hit.iloc[0].seed))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--verified",
        type=int,
        metavar="N",
        help=(
            "run the first N seeds of the verified range instead of the curated set. "
            "Unbiased, so this is what the limit-mix comparison should use."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        nargs="+",
        metavar="SEED",
        help="run these specific seeds",
    )
    ap.add_argument(
        "--seeds",
        metavar="LIST",
        help=(
            "comma-separated seeds, e.g. --seeds 90017,90042. With --out each lands in "
            "moves/review/detail.csv as scenario=fitted_<seed>. Rerunning a seed replaces "
            "its rows, so this is idempotent."
        ),
    )
    ap.add_argument(
        "--cases",
        action="store_true",
        help=(
            "regenerate the seeds pinned in model/fitted_cases.toml. Seconds, not a scan."
        ),
    )
    ap.add_argument(
        "--scan",
        type=int,
        metavar="N",
        help=(
            "solve N days and report which constraint families co-occur. A development "
            "step, not a presentation step."
        ),
    )
    ap.add_argument(
        "--select",
        action="store_true",
        help="after --scan, name the best seed per pattern, or say none matched.",
    )
    ap.add_argument("--out", type=Path, default=None, help="write CSVs here")
    args = ap.parse_args(argv)

    if args.scan:
        return _scan_mode(args)

    plot_frames = bool(args.cases or args.seeds or args.seed)

    if args.cases:
        cases = load_cases()
        seed_list = [c["seed"] for c in cases]
        sample = f"pinned cases from {CASES_PATH.name}"
    elif args.seeds:
        seed_list = [int(s_.strip()) for s_ in args.seeds.split(",") if s_.strip()]
        sample = "specified"
    elif args.seed:
        seed_list, sample = list(args.seed), "specified"
    elif args.verified:
        lo, hi = seeds.VERIFIED_RANGE
        seed_list = list(range(lo, min(lo + args.verified, hi)))
        sample = "verified range, unbiased"
    else:
        seed_list = seeds.curated()
        sample = "curated"

    unverified = [s_ for s_ in seed_list if not seeds.is_verified(s_)]

    print(f"Fitted days - the model on {len(seed_list)} independent days")
    print(f"Seed sample: {sample}.  Range verified to solve: {seeds.VERIFIED_RANGE}")
    print("Distributions fitted to real output; the site and every value are synthetic.")
    if unverified:
        print(
            f"\n  WARNING: {len(unverified)} seed(s) outside the verified range: "
            f"{unverified}.\n  These have not been checked and may not solve."
        )
    print()

    rows, limit_total, drift, results = [], Counter(), [], []
    for i, seed in enumerate(seed_list):
        result, _, _ = solve_day_parts(seed)
        results.append(result)
        r = summarise(result)
        limit_total.update(r.pop("limit_counts"))
        rows.append(r)
        note = ""
        if seed in seeds.DEMO_SEEDS:
            want = seeds.expected(seed)
            if (r["moves"], r["limit"]) != (want.moves, want.limit):
                note = (
                    f"   DRIFT: recorded {want.moves} moves / {want.limit}"
                )
                drift.append(seed)
        print(
            f"  seed {seed}  {r['moves']:4d} moves  "
            f"{r['tiers_resolved']}/{r['tiers_breached']} tiers cleared  "
            f"limit={r['limit']}{note}"
        )
    df = pd.DataFrame(rows)

    if drift:
        print(
            f"\n  {len(drift)} curated seed(s) no longer produce their recorded outcome: "
            f"{drift}.\n  The model's behaviour has changed. Re-verify before presenting, "
            f"and update mock_inputs/seeds.py."
        )

    print("\n--- what the model recommended ---")
    print(f"  days with NO movement recommended   {int((df.moves == 0).sum()):3d} / {len(df)}")
    print(
        f"  movements per day                   mean {df.moves.mean():6.1f}  "
        f"median {df.moves.median():6.1f}  max {df.moves.max():4d}"
    )
    print(f"  tiers breached per day              mean {df.tiers_breached.mean():6.2f}")
    print(
        f"  of those, cleared                   mean {df.tiers_resolved.mean():6.2f}  "
        f"({df.tiers_resolved.sum() / max(df.tiers_breached.sum(), 1):.0%} overall)"
    )
    gain = (df.ob_capacity_optimal - df.ob_capacity_actual) / df.ob_capacity_actual
    print(
        f"  outbound capacity moved             mean {gain.mean():+.1%}  "
        f"p10 {gain.quantile(.1):+.1%}  p90 {gain.quantile(.9):+.1%}"
    )

    print("\n--- which ceiling the model ended up on, per day ---")
    for limit, n in df.limit.value_counts().items():
        print(f"  {limit:26} {n:3d} / {len(df)}  ({n / len(df):.0%})")

    if args.out:
        (args.out / "data").mkdir(parents=True, exist_ok=True)
        if args.cases:
            pd.DataFrame(
                [
                    {
                        "seed": c["seed"],
                        "scenario": f"fitted_{c['seed']}",
                        "pattern": c.get("pattern", ""),
                        "title": c.get("title", ""),
                        "description": " ".join(c.get("description", "").split()),
                    }
                    for c in load_cases()
                ]
            ).to_csv(args.out / "data" / "fitted_cases.csv", index=False)
        if plot_frames:
            appended, _kept = demo_run.append_data(results, args.out)
            print(
                f"\nAppended {len(results)} fitted day(s) to "
                f"{', '.join(pth.name for pth in appended)} as "
                f"{', '.join(r.scenario.name for r in results)}"
            )
        else:
            print(
                f"\nSummary only - {sample} is a distribution run. Use --cases or "
                f"--seeds to publish per-hour frames for plotting."
            )
        df.to_csv(args.out / "data" / "fitted_days.csv", index=False)
        print(f"\nWrote fitted_days.csv to {args.out / 'data'}")

    print("\nRead this as a behaviour check, not an accuracy claim. The inputs are drawn")
    print("from fitted distributions, so the model is being asked the right SHAPE of")
    print("question, not a real one. And the limit column is a diagnostic, not an")
    print("attribution: ceilings interact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
