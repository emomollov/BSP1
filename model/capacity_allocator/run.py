"""Orchestration: generate -> solve twice -> dataframes -> publish."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config as cfg
from . import data, model, scenarios
from .data import Params
from .scenarios import Scenario

# --- solved model -> dataframes ---

#: Reasons outbound capacity could not be higher in a given hour.
LIMIT_OUTSIDE_SHIFT = "outside_shift"
LIMIT_NO_WORK = "no_work_available"
LIMIT_MACHINE = "machine"
LIMIT_PEOPLE = "cross_trained_people"
LIMIT_NOT_NEEDED = "no_more_capacity_needed"
LIMIT_UNCONSTRAINED = "not_constrained"

#: Tolerance for "at its bound", in units.
EPS = 0.1


@dataclass
class Solved:
    """One side of the before/after pair."""

    backend: model.Backend
    model: model.Model
    ladder: model.LadderResult

    @property
    def params(self) -> Params:
        """Read through the model so the two cannot disagree."""
        return self.model.params

    def v(self, var) -> float:
        return self.backend.value(var)


def build_hours(p: Params, before: Solved, after: Solved) -> pd.DataFrame:
    """Per-hour results, as optimal/actual pairs."""
    rows = []
    for h in p.hours:
        bl_window = p.bl_window[h]
        # NaN where the forward window runs off the horizon.
        windowed = len(bl_window) >= cfg.BL_WINDOW_HOURS
        rate_opt = (
            after.v(after.model.S_BL[h]) / len(bl_window) if windowed else math.nan
        )
        rate_act = (
            before.v(before.model.S_BL[h]) / len(bl_window) if windowed else math.nan
        )

        rows.append(
            {
                "site": cfg.SITE,
                "hour": h,
                "clock": cfg.hour_label(h),
                "shift_tier": p.tier_of_hour[h],
                "workable": h in p.workable,
                # --- outbound capacity ---
                "ob_capacity_optimal_units": after.v(after.model.Hcap[(cfg.OB, h)]),
                "ob_capacity_actual_units": before.v(before.model.Hcap[(cfg.OB, h)]),
                "ob_capacity_ceiling_units": p.ob_machine[h],
                "ob_capacity_ceiling_observed": p.ob_machine_observed[h],
                "ob_tph": p.tph.get((cfg.OB, h), 0.0),
                # --- outbound people ---
                "ob_people_optimal": _heads(p, after, cfg.OB, h),
                "ob_people_actual": _heads_planned(p, cfg.OB, h),
                "ob_people_pool": _pool_at(p, cfg.OB, h),
                # --- outbound backlog ---
                "ob_backlog_optimal_units": after.v(after.model.U[(cfg.OB, h)]),
                "ob_backlog_actual_units": before.v(before.model.U[(cfg.OB, h)]),
                "ob_arrivals_units": p.arrivals.get((cfg.OB, h), 0.0),
                "ob_throughput_optimal_units": after.v(after.model.Y[(cfg.OB, h)]),
                "ob_throughput_actual_units": before.v(before.model.Y[(cfg.OB, h)]),
                # --- outbound backlog in hours of cover (same denominator both sides) ---
                "ob_bl_capacity_per_hour_optimal": rate_opt,
                "ob_bl_capacity_per_hour_actual": rate_act,
                "ob_bl_optimal_hours": _safe_div(
                    after.v(after.model.U[(cfg.OB, h)]), rate_opt
                ),
                "ob_bl_actual_hours": _safe_div(
                    before.v(before.model.U[(cfg.OB, h)]), rate_act
                ),
                "ob_bl_min_hours": p.bl_min_hours[h],
                "ob_bl_max_hours": p.bl_max_hours[h],
                # Band midpoint, which the `ob_bl_centre` tier aims at.
                "ob_bl_centre_hours": 0.5 * (p.bl_min_hours[h] + p.bl_max_hours[h]),
                "ob_bl_band_observed": windowed,
                "ob_bl_breach_optimal_units": after.v(after.model.v_minus[h]),
                "ob_bl_breach_actual_units": before.v(before.model.v_minus[h]),
                # Distance from the centre, minimised by the bottom-guardrail tier.
                "ob_bl_centre_optimal_units": _centre_dev(after, h),
                "ob_bl_centre_actual_units": _centre_dev(before, h),
                # --- inbound ---
                "ib_capacity_optimal_units": after.v(after.model.Hcap[(cfg.IB, h)]),
                "ib_capacity_actual_units": before.v(before.model.Hcap[(cfg.IB, h)]),
                "ib_people_optimal": _heads(p, after, cfg.IB, h),
                "ib_people_actual": _heads_planned(p, cfg.IB, h),
                "ib_backlog_optimal_units": after.v(after.model.U[(cfg.IB, h)]),
                "ib_backlog_actual_units": before.v(before.model.U[(cfg.IB, h)]),
                "ib_storage_ceiling_units": p.ib_storage,
                "ib_storage_optimal_pct": after.v(after.model.U[(cfg.IB, h)]) / p.ib_storage,
                "ib_storage_actual_pct": before.v(before.model.U[(cfg.IB, h)]) / p.ib_storage,
                "ib_storage_breach_optimal_units": after.v(after.model.o_storage[h]),
                "ib_storage_breach_actual_units": before.v(before.model.o_storage[h]),
                # Overflow above the buffer line, not the ceiling.
                "ib_storage_buffer_limit": cfg.THRESHOLDS["storage_buffer"],
                "ib_storage_buffer_optimal_units": after.v(after.model.o_storage_buffer[h]),
                "ib_storage_buffer_actual_units": before.v(before.model.o_storage_buffer[h]),
                # --- returns ---
                "rt_capacity_optimal_units": after.v(after.model.Hcap[(cfg.RT, h)]),
                "rt_capacity_actual_units": before.v(before.model.Hcap[(cfg.RT, h)]),
                "rt_people_optimal": _heads(p, after, cfg.RT, h),
                "rt_people_actual": _heads_planned(p, cfg.RT, h),
                "rt_backlog_optimal_units": after.v(after.model.U[(cfg.RT, h)]),
                "rt_backlog_actual_units": before.v(before.model.U[(cfg.RT, h)]),
                # --- support ---
                "sp_people_optimal": _heads(p, after, cfg.SP, h),
                "sp_people_actual": _heads_planned(p, cfg.SP, h),
                # --- total outbound units (end-of-day numerator) ---
                "ob_total_optimal_units": after.v(after.model.U_total[h]),
                "ob_total_actual_units": before.v(before.model.U_total[h]),
                # --- why not higher ---
                "ob_capacity_limited_by": classify_limit(p, after, h),
            }
        )
    return pd.DataFrame(rows)


def build_cpts(p: Params, before: Solved, after: Solved) -> pd.DataFrame:
    """Per-cut-off deadline risk."""
    tol = cfg.THRESHOLDS["cpt_risk"]
    rows = []
    for c in p.cpts:
        cum_opt = after.v(after.model.H_cum[c])
        cum_act = before.v(before.model.H_cum[c])
        rows.append(
            {
                "site": cfg.SITE,
                "cpt_index": c,
                "cpt_hour": p.cpt_hour[c],
                "clock": cfg.hour_label(p.cpt_hour[c]),
                "shift_tier": p.cpt_tier[c],
                "demand_units": p.cpt_demand[c],
                "ob_cum_capacity_optimal_units": cum_opt,
                "ob_cum_capacity_actual_units": cum_act,
                "cpt_risk_tolerance": tol,
                "cpt_risk_optimal_pct": _safe_div(p.cpt_demand[c], tol * cum_opt),
                "cpt_risk_actual_pct": _safe_div(p.cpt_demand[c], tol * cum_act),
                "cpt_breach_optimal_units": after.v(after.model.g_cpt[c]),
                "cpt_breach_actual_units": before.v(before.model.g_cpt[c]),
            }
        )
    return pd.DataFrame(rows)


def build_moves(p: Params, before: Solved, after: Solved) -> pd.DataFrame:
    """Recommended movements per work block."""
    rows = []
    for q in p.quarters:
        row = {
            "site": cfg.SITE,
            "quarter": q,
            "event_type": "shift_start" if q % 2 == 1 else "block_start",
        }
        for d in cfg.DEPTS:
            row[f"people_actual_{d}"] = p.planned_heads.get((q, d), 0)
            row[f"people_optimal_{d}"] = round(after.v(after.model.P[(q, d)]))
            row[f"people_delta_{d}"] = row[f"people_optimal_{d}"] - row[f"people_actual_{d}"]
        total = 0
        for i in cfg.DEPTS:
            for j in cfg.DEPTS:
                if i == j:
                    continue
                n = round(after.v(after.model.X[(q, i, j)]))
                row[f"move_{i}_to_{j}"] = n
                total += n
        row["moves_total"] = total
        rows.append(row)
    return pd.DataFrame(rows)


def build_ladder(before: Solved, after: Solved) -> pd.DataFrame:
    """The objective ladder, before and after."""
    rows = []
    for i, name in enumerate(before.ladder.order):
        b = before.ladder.attained[name]
        a = after.ladder.attained.get(name, math.nan)
        kind = before.ladder.kinds.get(name, "slack")
        rows.append(
            {
                "priority": i + 1,
                "tier": name,
                "label": before.ladder.labels[name],
                "kind": kind,
                "slack_actual": round(b, 1),
                "slack_optimal": round(a, 1),
                "relieved": round(b - a, 1),
                "outcome": _outcome(b, a, kind),
            }
        )
    return pd.DataFrame(rows)


#: Suffixes the ladder uses to split one constraint family across shift partitions.
SHIFT_SUFFIXES = ("_now", "_later")


def family_of(tier: str) -> str:
    """Tier name -> constraint family, by stripping the shift-partition suffix."""
    for suffix in SHIFT_SUFFIXES:
        if tier.endswith(suffix):
            return tier[: -len(suffix)]
    return tier


def breached_families(ladder: pd.DataFrame, side: str = "slack_actual") -> dict[str, set[str]]:
    """Families with a non-zero slack, grouped by tier `kind`."""
    hit = ladder[ladder[side] > EPS]
    return {
        kind: {family_of(t) for t in group["tier"]}
        for kind, group in hit.groupby("kind")
    }


def _outcome(b: float, a: float, kind: str) -> str:
    """Label a tier's before/after pair in the vocabulary its `kind` deserves.

    Preferences and guardrails are never "unresolved": a movement cost is the price of the
    recommendation, and a guardrail is a distance from a target, rarely zero.
    """
    if kind == "preference":
        return "preference"
    if kind == "guardrail":
        if abs(a) <= EPS:
            return "at target"
        if a < b - EPS:
            return "risk reduced"
        if a > b + EPS:
            return "traded for a higher tier"
        return "unchanged"
    if abs(b) <= EPS:
        # Clear before, not necessarily after: a lower tier may break to relieve a higher one.
        return "clear" if abs(a) <= EPS else "broken to relieve a higher tier"
    if abs(a) <= EPS:
        return "resolved"
    if a < b - EPS:
        return "reduced"
    if a > b + EPS:
        return "worsened to relieve a higher tier"
    return "unresolved"


# --- the limit classifier ---


def classify_limit(p: Params, after: Solved, h: float) -> str:
    """Name the ceiling outbound capacity sits on in this hour (a diagnostic, not an attribution).

    Checks run from most physical to least, so order matters.
    """
    m = after.model
    if h not in p.workable:
        return LIMIT_OUTSIDE_SHIFT

    cap = after.v(m.Hcap[(cfg.OB, h)])

    if p.ob_machine_observed[h] and cap >= p.ob_machine[h] - EPS:
        return LIMIT_MACHINE

    # Everyone who could be on outbound already is (P at the pool ceiling in every block).
    quarters_here = [q for q in p.quarters if p.cover.get((h, q), 0.0) > 0]
    if quarters_here and all(
        after.v(m.P[(q, cfg.OB)]) >= p.movable_pool[(q, cfg.OB)] - EPS
        for q in quarters_here
    ):
        return LIMIT_PEOPLE

    # Capacity exceeds the work available.
    if after.v(m.Y[(cfg.OB, h)]) < cap - EPS:
        return LIMIT_NO_WORK

    # Means "no constraint wanted more capacity here", NOT "no more was available".
    if not _any_breach(after, h):
        return LIMIT_NOT_NEEDED

    return LIMIT_UNCONSTRAINED


def _any_breach(after: Solved, h: float) -> bool:
    """Would any constraint have wanted more outbound capacity in this hour?"""
    m, p = after.model, after.params

    if h in m.v_minus and after.v(m.v_minus[h]) > EPS:
        return True

    # A breaching cut-off counts only inside its own window.
    for c in m.g_cpt:
        if p.alpha.get((c, h), 0.0) > 0 and after.v(m.g_cpt[c]) > EPS:
            return True

    return False


LIMIT_RESOLVED = "resolved"


def dominant_limit(hours: pd.DataFrame) -> str:
    """The ceiling holding back the hours that still breach after the solve."""
    breaching = hours[
        (hours["ob_bl_breach_optimal_units"] > EPS)
        | (hours["ib_storage_breach_optimal_units"] > EPS)
    ]
    if breaching.empty:
        return LIMIT_RESOLVED
    counts = breaching["ob_capacity_limited_by"].value_counts()
    counts = counts.drop(labels=[LIMIT_OUTSIDE_SHIFT], errors="ignore")
    if counts.empty:
        return LIMIT_RESOLVED
    return str(counts.index[0])


# --- helpers ---


def _centre_dev(s: Solved, h: float) -> float:
    """Deviation of outbound backlog from its band centre, or NaN where undefined."""
    var = s.model.d_centre.get(h)
    return math.nan if var is None else s.v(var)


def _heads(p: Params, s: Solved, dept: str, h: float) -> float:
    """Coverage-weighted headcount on a department in an hour."""
    return sum(
        s.v(s.model.P[(q, dept)]) * p.cover[(h, q)] for q in p.quarters if (h, q) in p.cover
    )


def _heads_planned(p: Params, dept: str, h: float) -> float:
    return sum(
        p.planned_heads.get((q, dept), 0) * p.cover[(h, q)]
        for q in p.quarters
        if (h, q) in p.cover
    )


def _pool_at(p: Params, dept: str, h: float) -> float:
    """Everyone who could work on this department in this hour."""
    return sum(
        p.movable_pool[(q, dept)] * p.cover[(h, q)]
        for q in p.quarters
        if (h, q) in p.cover
    )


def _safe_div(num: float, den: float) -> float:
    """NaN on a non-positive or absent denominator; zero would read as a measurement."""
    if den is None or (isinstance(den, float) and math.isnan(den)) or den <= 0:
        return math.nan
    return num / den


def round_for_publication(df: pd.DataFrame) -> pd.DataFrame:
    """Round units to 1dp and ratios to 3dp (BL bands move in 0.25-hour steps, 1dp would blur a breach)."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind != "f":
            continue
        if col.endswith(("_pct", "_hours", "_tph", "_per_hour", "_per_hour_optimal", "_per_hour_actual")):
            out[col] = out[col].round(3)
        else:
            out[col] = out[col].round(1)
    return out


# --- orchestration ---


@dataclass
class Result:
    """Everything one scenario produces."""

    scenario: Scenario
    params: Params
    before: Solved
    after: Solved
    hours: pd.DataFrame
    cpts: pd.DataFrame
    moves: pd.DataFrame
    ladder: pd.DataFrame
    seconds: float

    @property
    def moves_total(self) -> int:
        return int(self.moves["moves_total"].sum())

    @property
    def limit(self) -> str:
        return dominant_limit(self.hours)

    @property
    def resolved(self) -> bool:
        return self.limit == LIMIT_RESOLVED


def run_scenario(scn: Scenario, out_dir: Path | None = None) -> Result:
    """Generate one scenario's data and solve it twice."""
    t0 = time.time()
    frames = data.generate(scn)
    if out_dir is not None:
        data.write(frames, out_dir / "inputs" / scn.name)

    p = data.build(frames)

    sides = {}
    for label, freeze in (("before", True), ("after", False)):
        backend = model.make_backend(f"{scn.name}_{label}")
        m = model.build(backend, p, freeze_moves=freeze)
        ladder = model.solve_lexicographic(backend, m.tiers)
        sides[label] = Solved(backend, m, ladder)

    return _result(scn, p, sides["before"], sides["after"], time.time() - t0)


def _result(scn: Scenario, p: Params, before: Solved, after: Solved, seconds: float) -> Result:
    return Result(
        scenario=scn,
        params=p,
        before=before,
        after=after,
        hours=build_hours(p, before, after),
        cpts=build_cpts(p, before, after),
        moves=build_moves(p, before, after),
        ladder=build_ladder(before, after),
        seconds=seconds,
    )


def fitted_result(
    seed: int,
    p: Params,
    before: Solved,
    after: Solved,
    seconds: float,
) -> Result:
    """Wrap a fitted day's solve as a `Result`, built exactly as `run_scenario` builds one."""
    scn = Scenario(
        name=f"fitted_{seed}",
        headline=f"An unengineered day (seed {seed})",
    )
    return _result(scn, p, before, after, seconds)


# --- emission: reshaped at emit time, so `Result` frames keep the model's shape ---


def _scenario_rank(names: pd.Series) -> pd.Series:
    """Sort key putting scenarios in `scenarios.ORDER`, with fitted days last (ordered by name)."""
    rank = {n: i for i, n in enumerate(scenarios.ORDER)}
    return names.map(rank).fillna(len(rank))


#: Per-file sort keys, applied after scenario rank and name.
_SORT_WITHIN = {
    "moves": ["quarter", "_from", "_to"],
    "review": ["priority"],
    "detail": ["hour"],
}


def _sort_published(stem: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Order a published frame from its own columns, so it works on frames read back from disk."""
    out = frame.copy()
    if stem == "moves" and len(out):
        pos = {cfg.display(d): i for i, d in enumerate(cfg.DEPTS)}
        out["_from"] = out["from"].map(pos)
        out["_to"] = out["to"].map(pos)
    out["_scn"] = _scenario_rank(out["scenario"])
    out = out.sort_values(["_scn", "scenario"] + _SORT_WITHIN[stem], kind="stable")
    return out.drop(columns=[c for c in ("_scn", "_from", "_to") if c in out])


def _quarter_start_hours(p: Params) -> dict[int, float]:
    """First model hour each work block covers, read from the coverage matrix the solve used."""
    starts: dict[int, float] = {}
    for (h, q), frac in p.cover.items():
        if frac > 0:
            starts[q] = min(starts.get(q, math.inf), h)
    return starts


#: `from`/`to` carry display names and `hour` a clock label; departments sort in `config.DEPTS` order.
MOVES_COLUMNS = ["scenario", "quarter", "event_type", "hour", "from", "to", "people"]


def moves_long(results: list[Result]) -> pd.DataFrame:
    """Every non-zero recommended movement, one per row (the wide `r.moves` is mostly zeros)."""
    order = {d: i for i, d in enumerate(cfg.DEPTS)}
    rows = []
    for r in results:
        starts = _quarter_start_hours(r.params)
        for mv in r.moves.itertuples():
            q = int(mv.quarter)
            for i in cfg.DEPTS:
                for j in cfg.DEPTS:
                    if i == j:
                        continue
                    people = int(getattr(mv, f"move_{i}_to_{j}"))
                    if not people:
                        continue
                    rows.append(
                        {
                            "scenario": r.scenario.name,
                            "quarter": q,
                            "event_type": mv.event_type,
                            "hour": cfg.hour_label(starts[q]),
                            "from": cfg.display(i),
                            "to": cfg.display(j),
                            "people": people,
                            "_from": order[i],
                            "_to": order[j],
                        }
                    )
    frame = pd.DataFrame(rows, columns=MOVES_COLUMNS)
    return _sort_published("moves", frame).reset_index(drop=True)


def review(results: list[Result]) -> pd.DataFrame:
    """The objective ladder for every scenario, with the scenario verdict repeated on each row."""
    rows = []
    for r in results:
        for t in r.ladder.itertuples():
            rows.append(
                {
                    "scenario": r.scenario.name,
                    "moves_total": r.moves_total,
                    "limited_by": r.limit,
                    "resolved": r.resolved,
                    "priority": int(t.priority),
                    "tier": t.tier,
                    "label": t.label,
                    "kind": t.kind,
                    "slack_actual": t.slack_actual,
                    "slack_optimal": t.slack_optimal,
                    "relieved": t.relieved,
                    "outcome": t.outcome,
                }
            )
    return _sort_published("review", pd.DataFrame(rows)).reset_index(drop=True)


def detail_for(r: Result) -> pd.DataFrame:
    """One scenario's hourly trajectory with cut-off risk and day-level targets joined on (unrounded)."""
    hours = r.hours.copy()
    hours.insert(0, "scenario", r.scenario.name)

    # Two cut-offs in one hour would silently duplicate hour rows in the join.
    dupes = r.cpts["cpt_hour"][r.cpts["cpt_hour"].duplicated()].tolist()
    if dupes:
        raise ValueError(
            f"{r.scenario.name}: more than one carrier cut-off in hour(s) {dupes}. "
            f"detail.csv joins cut-offs onto hours 1:1 and cannot represent that; give "
            f"the cut-offs their own file, or aggregate them, before publishing."
        )

    # `clock`/`shift_tier` derive from the cut-off hour (with a different boundary rule), so keep them renamed.
    cpts = r.cpts.drop(columns=["site"]).rename(
        columns={"clock": "cpt_clock", "shift_tier": "cpt_shift_tier"}
    )
    out = hours.merge(cpts, how="left", left_on="hour", right_on="cpt_hour")

    # Per-scenario inputs, so data; the end-of-day hour is site geometry and stays in `config`.
    out["eod_capacity_per_hour"] = r.params.eod_capacity_per_hour
    out["eod_soft_max_hours"] = r.params.eod_soft_max
    out["eod_hard_min_hours"] = r.params.eod_hard_min
    return out


def published_frames(results: list[Result]) -> dict[str, pd.DataFrame]:
    """The three published tables, built and sorted in `scenarios.ORDER` regardless of `--scenario` order."""
    detail = pd.concat([detail_for(r) for r in results], ignore_index=True)
    return {
        "moves": moves_long(results),
        "review": review(results),
        "detail": _sort_published(
            "detail", round_for_publication(detail)
        ).reset_index(drop=True),
    }


def _publish(
    results: list[Result], out_dir: Path, drop: set[str]
) -> tuple[list[Path], set[str]]:
    """Write `results` into the three CSVs, replacing rows whose scenario is in `drop`; returns paths and kept scenarios."""
    data = out_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    fresh = published_frames(results)
    written, preserved = [], set()

    for stem, frame in fresh.items():
        path = data / f"{stem}.csv"
        if path.exists():
            kept = pd.read_csv(path)
            kept = kept[~kept["scenario"].isin(drop)]
            preserved |= set(kept["scenario"])
            frame = pd.concat([kept, frame], ignore_index=True)
        _sort_published(stem, frame).to_csv(path, index=False)
        written.append(path)
    return written, preserved


def write_data(results: list[Result], out_dir: Path) -> tuple[list[Path], set[str]]:
    """Publish an engineered-scenario run.

    Replaces rows for every engineered scenario (so no stale rows survive a partial run)
    and leaves everything else, i.e. appended fitted days, untouched.
    """
    return _publish(results, out_dir, drop=set(scenarios.ORDER))


def append_data(results: list[Result], out_dir: Path) -> tuple[list[Path], set[str]]:
    """Add these results to the published CSVs, replacing only rows of the same scenario."""
    return _publish(results, out_dir, drop={r.scenario.name for r in results})


def run_all(names: list[str], out_dir: Path) -> list[Result]:
    """Solve every requested scenario, writing generated inputs only; publishing is `write_data`."""
    results = []
    for scn in scenarios.resolve(names):
        r = run_scenario(scn, out_dir)
        results.append(r)
        status = "resolved" if r.resolved else f"limited by {r.limit}"
        print(
            f"  {scn.name:18} {r.seconds:5.1f}s  {r.moves_total:3d} moves  {status}"
        )
    return results
