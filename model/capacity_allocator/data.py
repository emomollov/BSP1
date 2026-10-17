"""Data boundary: the synthetic generator that writes the eight input CSVs, and `build`, which turns them into `Params`.

The generative chain is ordered so the baseline is in-band by construction:

    1. quarters          pure geometry from config.SHIFTS
    2. tph               shaped per department per hour
    3. headcount         shaped per quarter per origin per skill group, plus absenteeism
    4. planned capacity  DERIVED: tph x coverage-weighted available headcount
    5. arrivals          shaped, scenario-overridable
    6. backlog           DERIVED: the recursion run at planned capacity
    7. cut-off demand    DERIVED from cumulative planned capacity, scenario-overridable
    8. targets           DERIVED from the backlog trajectory

Step 4 is what makes the replay identity exact: planned capacity IS tph x heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config as cfg
from .scenarios import BASELINE, Scenario

# --- Baseline magnitudes: plausible, leave every constraint slack, not calibrated ---

#: Decimal places for published values; derived series are kept fine enough that the
#: recursion accumulates no visible residue.
TPH_DP = 2
UNIT_DP = 1
DERIVED_DP = 3

#: Units per person per hour, (trough, peak).
TPH_RANGE = {
    cfg.OB: (104.0, 126.0),
    cfg.IB: (72.0, 88.0),
    cfg.RT: (44.0, 52.0),
}

#: People per work block, by origin and skill group. The skill set always includes the
#: origin; only the cross-trained rows can move, the native-only row is the floor.
POOL = {
    # origin: {skills: people}
    cfg.OB: {
        (cfg.OB,): 44,
        (cfg.IB, cfg.OB): 18,
        (cfg.IB, cfg.OB, cfg.SP): 12,
        (cfg.OB, cfg.SP): 8,
    },
    cfg.IB: {
        (cfg.IB,): 14,
        (cfg.IB, cfg.OB): 12,
        (cfg.IB, cfg.OB, cfg.SP): 7,
        (cfg.IB, cfg.RT): 4,
    },
    cfg.RT: {
        (cfg.RT,): 5,
        (cfg.OB, cfg.RT): 5,
        (cfg.OB, cfg.RT, cfg.SP): 3,
    },
}

#: Per-quarter staffing multiplier (1-based quarter); the night block runs lighter.
QUARTER_STAFFING = {1: 1.00, 2: 1.00, 3: 0.95, 4: 0.95, 5: 0.86, 6: 0.86}

#: Flat absenteeism per department.
ABSENTEEISM = {cfg.OB: 0.06, cfg.IB: 0.05, cfg.RT: 0.04}

#: Arrivals per hour, (trough, peak); peaks mid-horizon, when staffing is heaviest.
ARRIVALS_RANGE = {
    cfg.OB: (5_800.0, 8_400.0),
    cfg.IB: (2_000.0, 2_500.0),
    cfg.RT: (400.0, 520.0),
}

#: Seed backlog at the first horizon hour, in units.
SEED_BACKLOG = {cfg.OB: 28_000.0, cfg.IB: 17_500.0, cfg.RT: 6_000.0}

#: Non-pickable units carried alongside the pickable backlog.
OB_NON_PICKABLE = 9_000.0

#: Outbound backlog band, in hours of forward cover.
BL_BAND_HOURS = (1.5, 4.0)

#: Share of cumulative planned outbound capacity due out by each cut-off; below the risk
#: tolerance so baseline leaves headroom.
CPT_DEMAND_SHARE = 0.84


# --- Shaping ---------------------------------------------------------------


def _largest_remainder(weights: dict, total: int) -> dict:
    """Split `total` across `weights` in proportion, summing to `total` exactly."""
    pool_total = sum(weights.values())
    if pool_total <= 0:
        return {k: 0 for k in weights}
    
    exact = {k: v * total / pool_total for k, v in weights.items()}
    out = {k: int(math.floor(v)) for k, v in exact.items()}
    deficit = total - sum(out.values())
    by_frac = sorted(weights, key=lambda k: exact[k] - out[k], reverse=True)
    for i in range(deficit):
        out[by_frac[i % len(by_frac)]] += 1
    return out


def _shaped(hours: list[float], low: float, high: float) -> dict[float, float]:
    """One cosine bump across the horizon: `low` at both ends, `high` in the middle.

    Flat columns hide bugs, e.g. a clipped forward window or a ceiling binding on one hour.
    """
    if len(hours) == 1:
        return {hours[0]: (low + high) / 2.0}
    n = len(hours) - 1
    return {
        h: low + (high - low) * 0.5 * (1.0 - math.cos(2.0 * math.pi * i / n))
        for i, h in enumerate(hours)
    }


# --- Geometry --------------------------------------------------------------


def build_quarters() -> list[dict]:
    """Work blocks (numbered decision points) and break closures across the horizon."""
    segments: list[dict] = []
    quarter = 0
    for shift_name, start, brk_start, brk_end, end in cfg.SHIFTS:
        for block_start, block_end in ((start, brk_start), (brk_end, end)):
            quarter += 1
            segments.append(
                {
                    "shift": shift_name,
                    "quarter": quarter,
                    "kind": "work",
                    "start": block_start,
                    "end": block_end,
                }
            )
        segments.append(
            {
                "shift": shift_name,
                "quarter": None,
                "kind": "closed",
                "start": brk_start,
                "end": brk_end,
            }
        )
    return sorted(segments, key=lambda s: (s["start"], s["kind"] != "work"))


def horizon_hours() -> list[float]:
    """Integer model hours in the horizon."""
    return [float(h) for h in range(int(cfg.HORIZON_START), int(cfg.HORIZON_END))]


def coverage(quarters: list[dict], hours: list[float]) -> dict[tuple[float, int], float]:
    """`W_seg`: the fraction of each hour covered by each work block."""
    cover: dict[tuple[float, int], float] = {}
    for seg in quarters:
        if seg["kind"] != "work":
            continue
        for h in hours:
            overlap = max(0.0, min(h + 1.0, seg["end"]) - max(h, seg["start"]))
            if overlap > 0:
                cover[(h, seg["quarter"])] = overlap
    return cover


# --- Simulation: the generative chain, without ceilings or targets ---------


@dataclass
class Sim:
    """One scenario's simulated day, before ceilings and targets are attached."""

    hours: list[float]
    quarters: list[dict]
    cover: dict[tuple[float, int], float]
    work_quarters: list[int]
    tph: dict[str, dict[float, float]]
    headcount: pd.DataFrame
    planned_heads: dict[tuple[int, str], int]
    planned_cap: dict[str, dict[float, float]]
    arrivals: dict[str, dict[float, float]]
    backlog: dict[str, dict[float, float]]
    throughput: dict[str, dict[float, float]]
    ob_total: dict[float, float]


def _simulate(scn: Scenario) -> Sim:
    """Run the generative chain for one scenario (ceilings and targets come from `_reference`)."""
    hours = horizon_hours()
    quarters = build_quarters()
    cover = coverage(quarters, hours)
    work_quarters = [s["quarter"] for s in quarters if s["kind"] == "work"]

    # --- tph ---------------------------------------------------------------
    # TPH and arrivals are the only true inputs, so they are rounded here, at generation,
    # keeping published inputs and trajectory in exact agreement.
    tph = {
        dept: {h: round(v, TPH_DP) for h, v in _shaped(hours, low, high).items()}
        for dept, (low, high) in TPH_RANGE.items()
    }

    # --- headcount ---------------------------------------------------------
    # Long format, one row per (quarter, origin, skill group).
    hc_rows = []
    for q in work_quarters:
        staffing = QUARTER_STAFFING.get(q, 1.0)
        for origin, groups in POOL.items():
            absent = ABSENTEEISM[origin]
            native_key = (origin,)

            # `xt_pool_share` removes flexibility, not headcount: stripped people join the
            # native-only group, so staffing and planned capacity are unchanged.
            stranded = 0
            sized: dict[tuple[str, ...], int] = {}
            for skills, people in groups.items():
                scaled = int(round(people * staffing))
                if len(skills) > 1:
                    kept = int(round(scaled * scn.xt_pool_share))
                    stranded += scaled - kept
                    sized[skills] = kept
                else:
                    sized[skills] = scaled
            sized[native_key] = sized.get(native_key, 0) + stranded

            # Absenteeism hits the origin total once, then is split by largest remainder;
            # per-group rounding would make the total depend on grouping, so
            # `pool_exhausted` would differ from baseline in headcount.
            available = _largest_remainder(sized, round(sum(sized.values()) * (1.0 - absent)))

            for skills, people_int in sized.items():
                hc_rows.append(
                    {
                        "site": cfg.SITE,
                        "quarter": q,
                        "origin": origin,
                        "skills": "+".join(skills),
                        "movable": len(skills) > 1,
                        "people": people_int,
                        "absenteeism": absent,
                        "available": available[skills],
                    }
                )
    headcount = pd.DataFrame(hc_rows)

    planned_heads: dict[tuple[int, str], int] = {}
    for (q, origin), grp in headcount.groupby(["quarter", "origin"]):
        planned_heads[(int(q), str(origin))] = int(grp["available"].sum())

    # --- planned capacity (DERIVED) ----------------------------------------
    # capacity[dept, h] = tph[dept, h] * coverage-weighted available heads on that dept.
    planned_cap: dict[str, dict[float, float]] = {d: {} for d in cfg.PRODUCING}
    for dept in cfg.PRODUCING:
        for h in hours:
            heads = sum(
                planned_heads.get((q, dept), 0) * cover.get((h, q), 0.0) for q in work_quarters
            )
            planned_cap[dept][h] = tph[dept][h] * heads

    # --- arrivals ----------------------------------------------------------
    arrivals = {dept: _shaped(hours, low, high) for dept, (low, high) in ARRIVALS_RANGE.items()}
    for h in scn.ob_surge_hours:
        if h in arrivals[cfg.OB]:
            arrivals[cfg.OB][h] *= scn.ob_arrival_mult
    for h in scn.ib_surge_hours:
        if h in arrivals[cfg.IB]:
            arrivals[cfg.IB][h] *= scn.ib_arrival_mult
    # Rounded after the surge multiplier.
    arrivals = {d: {h: round(v, UNIT_DP) for h, v in s.items()} for d, s in arrivals.items()}

    # --- backlog trajectory (DERIVED) --------------------------------------
    # throughput = min(capacity, backlog + arrivals): nothing unarrived can be processed.
    backlog: dict[str, dict[float, float]] = {}
    throughput: dict[str, dict[float, float]] = {}
    for dept in cfg.PRODUCING:
        b = {hours[0]: SEED_BACKLOG[dept]}
        y = {}
        for i, h in enumerate(hours):
            available = b[h] + arrivals[dept][h]
            y[h] = min(planned_cap[dept][h], available)
            if i + 1 < len(hours):
                b[hours[i + 1]] = available - y[h]
        backlog[dept] = b
        throughput[dept] = y

    # Total outbound units (end-of-day numerator), fed by the same throughput as the pickable series.
    ob_total = {hours[0]: SEED_BACKLOG[cfg.OB] + OB_NON_PICKABLE}
    for i, h in enumerate(hours[:-1]):
        ob_total[hours[i + 1]] = ob_total[h] + arrivals[cfg.OB][h] - throughput[cfg.OB][h]

    return Sim(
        hours=hours,
        quarters=quarters,
        cover=cover,
        work_quarters=work_quarters,
        tph=tph,
        headcount=headcount,
        planned_heads=planned_heads,
        planned_cap=planned_cap,
        arrivals=arrivals,
        backlog=backlog,
        throughput=throughput,
        ob_total=ob_total,
    )


@lru_cache(maxsize=1)
def _reference() -> Sim:
    """The cached BASELINE simulation, from which every ceiling and target is derived.

    Deriving them from the scenario instead would let `ib_surge` raise its own storage
    ceiling and hide its breach; the same applies to the end-of-day target.
    """
    return _simulate(BASELINE)


# --- Generation ------------------------------------------------------------


def generate(scn: Scenario) -> dict[str, pd.DataFrame]:
    """Build the input frames for one scenario."""
    sim = _simulate(scn)
    ref = _reference()
    hours = sim.hours

    # --- ceilings, from the BASELINE ---------------------------------------
    # `ob_machine_share` scales planned capacity; inside a scenario's peak window it
    # applies there and `ob_machine_offpeak_share` outside.
    def _share(h: float) -> float:
        if not scn.ob_machine_peak_hours:
            return scn.ob_machine_share
        return (
            scn.ob_machine_share
            if h in scn.ob_machine_peak_hours
            else scn.ob_machine_offpeak_share
        )

    ob_machine = {h: ref.planned_cap[cfg.OB][h] * _share(h) for h in hours}
    for h in scn.unobserved_machine_hours:
        if h in ob_machine:
            ob_machine[h] = float("nan")

    ib_storage = max(ref.backlog[cfg.IB].values()) * scn.ib_storage_share

    # --- frames ------------------------------------------------------------
    shifts_df = pd.DataFrame(
        [
            {
                "site": cfg.SITE,
                "shift": s["shift"],
                "quarter": s["quarter"],
                "interval_kind": s["kind"],
                "start_hour": s["start"],
                "end_hour": s["end"],
            }
            for s in sim.quarters
        ]
    )

    tph_df = pd.DataFrame(
        [
            {"site": cfg.SITE, "hour": h, "dept": dept, "tph": sim.tph[dept][h]}
            for dept in cfg.PRODUCING
            for h in hours
        ]
    )

    ob_plan_df = pd.DataFrame(
        [
            {
                "site": cfg.SITE,
                "hour": h,
                "ob_capacity_plan": round(sim.planned_cap[cfg.OB][h], DERIVED_DP),
                "ob_arrivals": sim.arrivals[cfg.OB][h],
                "ob_pickable_backlog_plan": round(sim.backlog[cfg.OB][h], DERIVED_DP),
                "ob_units_total_plan": round(sim.ob_total[h], DERIVED_DP),
                "ob_max_machine": (
                    float("nan") if math.isnan(ob_machine[h]) else round(ob_machine[h], DERIVED_DP)
                ),
                "bl_min_hours": BL_BAND_HOURS[0],
                "bl_max_hours": BL_BAND_HOURS[1],
            }
            for h in hours
        ]
    )

    ib_plan_df = pd.DataFrame(
        [
            {
                "site": cfg.SITE,
                "hour": h,
                "ib_capacity_plan": round(sim.planned_cap[cfg.IB][h], DERIVED_DP),
                "ib_arrivals": sim.arrivals[cfg.IB][h],
                "ib_backlog_plan": round(sim.backlog[cfg.IB][h], DERIVED_DP),
                "ib_max_storage": round(ib_storage, DERIVED_DP),
            }
            for h in hours
        ]
    )

    returns_plan_df = pd.DataFrame(
        [
            {
                "site": cfg.SITE,
                "hour": h,
                "rt_capacity_plan": round(sim.planned_cap[cfg.RT][h], DERIVED_DP),
                "rt_arrivals": sim.arrivals[cfg.RT][h],
                "rt_backlog_plan": round(sim.backlog[cfg.RT][h], DERIVED_DP),
            }
            for h in hours
        ]
    )

    cpts_df = _build_cpts(scn, hours, ref.planned_cap[cfg.OB])

    # --- targets, from the BASELINE ----------------------------------------
    # Denominator: average hourly outbound capacity; a fixed input, which keeps the constraint linear.
    eod_denominator = sum(ref.planned_cap[cfg.OB].values()) / len(hours)
    ref_eod_hours = ref.ob_total[float(cfg.EOD_HOUR)] / eod_denominator
    eod_guidance_df = pd.DataFrame(
        [
            {
                "site": cfg.SITE,
                "eod_hour": cfg.EOD_HOUR,
                "eod_capacity_per_hour": round(eod_denominator, DERIVED_DP),
                "eod_soft_max_hours": round(ref_eod_hours * scn.eod_soft_max_share, 3),
                "eod_hard_min_hours": round(ref_eod_hours * scn.eod_hard_min_share, 3),
            }
        ]
    )

    return {
        "shifts": shifts_df,
        "headcount": sim.headcount.drop(columns=["movable"]),
        "tph": tph_df,
        "ob_plan": ob_plan_df,
        "ib_plan": ib_plan_df,
        "returns_plan": returns_plan_df,
        "cpts": cpts_df,
        "eod_guidance": eod_guidance_df,
    }


def _build_cpts(scn: Scenario, hours: list[float], ob_cap: dict[float, float]) -> pd.DataFrame:
    """Cumulative units due out by each carrier cut-off, derived from baseline capacity.

    The scenario multiplier scales one cut-off's incremental bucket, then the series is
    re-cumulated; scaling a cumulative value would scale every later cut-off too.
    """
    # Each hour's capacity counts toward the cut-off window it falls in, pro rata.
    cum_cap = [
        sum(
            ob_cap[h] * max(0.0, min(h + 1.0, cpt) - max(h, cfg.HORIZON_START))
            for h in hours
        )
        for cpt in cfg.CPT_HOURS
    ]
    cum_demand = [c * CPT_DEMAND_SHARE for c in cum_cap]

    buckets = [cum_demand[0]] + [
        cum_demand[i] - cum_demand[i - 1] for i in range(1, len(cum_demand))
    ]
    if 0 <= scn.cpt_index < len(buckets):
        buckets[scn.cpt_index] *= scn.cpt_demand_mult

    running = 0.0
    rows = []
    for i, cpt in enumerate(cfg.CPT_HOURS):
        running += buckets[i]
        rows.append(
            {
                "site": cfg.SITE,
                "cpt_index": i + 1,
                "cpt_hour": cpt,
                "cumulative_units_due_out": round(running, DERIVED_DP),
            }
        )
    return pd.DataFrame(rows)


# --- Writing ---------------------------------------------------------------

def write(frames: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write the input frames as CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


# --- Input frames -> model parameters --------------------------------------


@dataclass
class Params:
    """Everything the model needs."""

    # --- time ---------------------------------------------------------------
    hours: list[float]
    #: Hours fully covered by a work block.
    workable: list[float]
    quarters: list[int]
    #: `W_seg`: the share of each hour covered by each work block.
    cover: dict[tuple[float, int], float]
    #: `Wcover`: the share of an edge hour not covered by any block (zero here).
    edge: dict[float, float]
    #: Priority tier per hour: 0 = current shift, 1 = later.
    tier_of_hour: dict[float, int]

    # --- people -------------------------------------------------------------
    #: `n_skill`: (quarter, origin, skill set) -> people available after absenteeism.
    pool: dict[tuple[int, str, tuple[str, ...]], int]

    #: `cap_plan`: the planned assignment. Everyone works on their origin.
    planned_heads: dict[tuple[int, str], int]

    #: Everyone whose skill set contains the department; read only by the output classifier.
    movable_pool: dict[tuple[int, str], int]

    # --- rates and demand ---------------------------------------------------
    tph: dict[tuple[str, float], float]
    arrivals: dict[tuple[str, float], float]
    seed_backlog: dict[str, float]

    #: Planned capacity carried by the inputs; the `actual` side of every optimal/actual pair.
    planned_cap: dict[tuple[str, float], float]

    # --- ceilings and bands -------------------------------------------------
    ob_machine: dict[float, float]
    #: False where the input had no machine reading.
    ob_machine_observed: dict[float, bool]
    ib_storage: float
    bl_min_hours: dict[float, float]
    bl_max_hours: dict[float, float]

    #: Forward window of workable hours for the backlog band.
    bl_window: dict[float, list[float]]
    #: Forward window for the returns days-of-cover ceiling.
    rt_window: dict[float, list[float]]

    # --- deadlines ----------------------------------------------------------
    cpts: list[int]
    cpt_hour: dict[int, float]
    cpt_demand: dict[int, float]

    #: `alpha`: the share of hour h falling in cut-off c's window.
    alpha: dict[tuple[int, float], float]
    cpt_tier: dict[int, int]

    # --- end of day ---------------------------------------------------------
    eod_hour: float
    eod_capacity_per_hour: float
    eod_soft_max: float
    eod_hard_min: float
    seed_ob_total: float

    # --- reference ----------------------------------------------------------
    #: Planned backlog trajectory; not a model input, the `actual` side of the backlog charts.
    planned_backlog: dict[tuple[str, float], float] = field(default_factory=dict)
    planned_ob_total: dict[float, float] = field(default_factory=dict)


def build(frames: dict[str, pd.DataFrame]) -> Params:
    """Turn the eight input frames into model parameters."""
    shifts = frames["shifts"]
    ob = frames["ob_plan"].sort_values("hour")
    ib = frames["ib_plan"].sort_values("hour")
    rt = frames["returns_plan"].sort_values("hour")

    hours = [float(h) for h in ob["hour"]]

    # --- quarters and coverage ---------------------------------------------
    work = shifts[shifts["interval_kind"] == "work"].sort_values("start_hour")
    quarters = [int(q) for q in work["quarter"]]
    cover: dict[tuple[float, int], float] = {}
    for seg in work.itertuples():
        for h in hours:
            overlap = max(0.0, min(h + 1.0, seg.end_hour) - max(h, seg.start_hour))
            if overlap > 0:
                cover[(h, int(seg.quarter))] = overlap

    covered = {h: sum(cover.get((h, q), 0.0) for q in quarters) for h in hours}
    workable = [h for h in hours if covered[h] >= 1.0 - 1e-9]

    # Uncovered share of an edge hour; zero here as every block boundary is a whole hour.
    edge = {h: max(0.0, 1.0 - covered[h]) for h in hours}

    # --- priority tiers ----------------------------------------------------
    # The first shift is tier 0, everything after is tier 1.
    first_shift = work.iloc[0]["shift"]
    first_end = float(work[work["shift"] == first_shift]["end_hour"].max())
    tier_of_hour = {h: (0 if h < first_end else 1) for h in hours}

    # --- pool --------------------------------------------------------------
    hc = frames["headcount"]
    pool: dict[tuple[int, str, tuple[str, ...]], int] = {}
    for r in hc.itertuples():
        skills = tuple(sorted(str(r.skills).split("+")))
        key = (int(r.quarter), str(r.origin), skills)
        pool[key] = pool.get(key, 0) + int(r.available)

    planned_heads: dict[tuple[int, str], int] = {}
    for (q, origin, _), people in pool.items():
        planned_heads[(q, origin)] = planned_heads.get((q, origin), 0) + people
    for q in quarters:
        for d in cfg.DEPTS:
            planned_heads.setdefault((q, d), 0)

    # Everyone whose skill set contains the department: those on it plus those who could move in.
    movable_pool: dict[tuple[int, str], int] = {
        (q, d): 0 for q in quarters for d in cfg.DEPTS
    }
    for (q, _origin, skills), people in pool.items():
        for d in skills:
            movable_pool[(q, d)] += people

    # --- rates, demand, ceilings -------------------------------------------
    tph: dict[tuple[str, float], float] = {}
    for r in frames["tph"].itertuples():
        tph[(str(r.dept), float(r.hour))] = float(r.tph)

    arrivals: dict[tuple[str, float], float] = {}
    planned_cap: dict[tuple[str, float], float] = {}
    planned_backlog: dict[tuple[str, float], float] = {}
    for r in ob.itertuples():
        arrivals[(cfg.OB, float(r.hour))] = float(r.ob_arrivals)
        planned_cap[(cfg.OB, float(r.hour))] = float(r.ob_capacity_plan)
        planned_backlog[(cfg.OB, float(r.hour))] = float(r.ob_pickable_backlog_plan)
    for r in ib.itertuples():
        arrivals[(cfg.IB, float(r.hour))] = float(r.ib_arrivals)
        planned_cap[(cfg.IB, float(r.hour))] = float(r.ib_capacity_plan)
        planned_backlog[(cfg.IB, float(r.hour))] = float(r.ib_backlog_plan)
    for r in rt.itertuples():
        arrivals[(cfg.RT, float(r.hour))] = float(r.rt_arrivals)
        planned_cap[(cfg.RT, float(r.hour))] = float(r.rt_capacity_plan)
        planned_backlog[(cfg.RT, float(r.hour))] = float(r.rt_backlog_plan)

    seed_backlog = {d: planned_backlog[(d, hours[0])] for d in cfg.PRODUCING}

    # A missing machine reading becomes the sentinel plus observed=False, not a plausible number.
    ob_machine: dict[float, float] = {}
    ob_machine_observed: dict[float, bool] = {}
    for r in ob.itertuples():
        raw = float(r.ob_max_machine)
        missing = math.isnan(raw)
        ob_machine[float(r.hour)] = cfg.NO_MACHINE_LIMIT if missing else raw
        ob_machine_observed[float(r.hour)] = not missing

    ib_storage = float(ib["ib_max_storage"].iloc[0])
    bl_min_hours = {float(r.hour): float(r.bl_min_hours) for r in ob.itertuples()}
    bl_max_hours = {float(r.hour): float(r.bl_max_hours) for r in ob.itertuples()}

    # Forward windows skip breaks rather than averaging them in as zero capacity.
    bl_window = {h: [k for k in workable if k >= h][: cfg.BL_WINDOW_HOURS] for h in hours}
    rt_window = {h: [k for k in workable if k >= h][: cfg.RETURNS_WINDOW_HOURS] for h in hours}

    # --- deadlines ---------------------------------------------------------
    cp = frames["cpts"].sort_values("cpt_hour")
    cpts = [int(r.cpt_index) for r in cp.itertuples()]
    cpt_hour = {int(r.cpt_index): float(r.cpt_hour) for r in cp.itertuples()}
    cpt_demand = {
        int(r.cpt_index): float(r.cumulative_units_due_out) for r in cp.itertuples()
    }

    alpha: dict[tuple[int, float], float] = {}
    for i, c in enumerate(cpts):
        lo = cfg.HORIZON_START if i == 0 else cpt_hour[cpts[i - 1]]
        hi = cpt_hour[c]
        for h in hours:
            alpha[(c, h)] = max(0.0, min(h + 1.0, hi) - max(h, lo))

    cpt_tier = {c: (0 if cpt_hour[c] <= first_end else 1) for c in cpts}

    # --- end of day --------------------------------------------------------
    g = frames["eod_guidance"].iloc[0]
    eod_hour = float(g.eod_hour)
    planned_ob_total = {float(r.hour): float(r.ob_units_total_plan) for r in ob.itertuples()}

    p = Params(
        hours=hours,
        workable=workable,
        quarters=quarters,
        cover=cover,
        edge=edge,
        tier_of_hour=tier_of_hour,
        pool=pool,
        planned_heads=planned_heads,
        movable_pool=movable_pool,
        tph=tph,
        arrivals=arrivals,
        seed_backlog=seed_backlog,
        planned_cap=planned_cap,
        ob_machine=ob_machine,
        ob_machine_observed=ob_machine_observed,
        ib_storage=ib_storage,
        bl_min_hours=bl_min_hours,
        bl_max_hours=bl_max_hours,
        bl_window=bl_window,
        rt_window=rt_window,
        cpts=cpts,
        cpt_hour=cpt_hour,
        cpt_demand=cpt_demand,
        alpha=alpha,
        cpt_tier=cpt_tier,
        eod_hour=eod_hour,
        eod_capacity_per_hour=float(g.eod_capacity_per_hour),
        eod_soft_max=float(g.eod_soft_max_hours),
        eod_hard_min=float(g.eod_hard_min_hours),
        seed_ob_total=planned_ob_total[hours[0]],
        planned_backlog=planned_backlog,
        planned_ob_total=planned_ob_total,
    )
    _check(p)
    return p


def planned_capacity_from_heads(p: Params, dept: str, h: float) -> float:
    """Planned capacity rebuilt from planned headcount; must equal the inputs' `planned_cap`."""
    heads = sum(p.planned_heads.get((q, dept), 0) * p.cover.get((h, q), 0.0) for q in p.quarters)
    return p.tph.get((dept, h), 0.0) * heads + p.edge.get(h, 0.0) * p.planned_cap.get(
        (dept, h), 0.0
    )


def _check(p: Params) -> None:
    """Fail loudly on anything that would otherwise surface as a solver error."""
    for (dept, h), v in list(p.tph.items()) + list(p.arrivals.items()):
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"non-finite input for {dept} at hour {h}: {v}")

    # Replay identity, checked here so a mismatch can still name the hour.
    for dept in cfg.PRODUCING:
        for h in p.hours:
            got = planned_capacity_from_heads(p, dept, h)
            want = p.planned_cap[(dept, h)]
            if abs(got - want) > 0.5:
                raise ValueError(
                    f"planned capacity for {dept} at hour {h} does not match the headcount: "
                    f"heads imply {got:.1f}, input says {want:.1f}. The replay identity "
                    f"will not hold and every before/after delta will be contaminated."
                )

    if p.eod_hour not in p.hours:
        raise ValueError(f"end-of-day hour {p.eod_hour} is outside the horizon")

    reach = max(
        max(p.planned_cap.values()),
        max(p.planned_backlog.values()),
        max(p.planned_ob_total.values()),
        p.ib_storage,
    )
    if reach * 2.0 > cfg.BIG_M:
        raise ValueError(
            f"BIG_M ({cfg.BIG_M:.0f}) is too small for data reaching {reach:.0f}; it is "
            f"the lower bound on model.V_raw and would bind on real values"
        )