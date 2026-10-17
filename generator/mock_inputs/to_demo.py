"""Bridge: the generated frames -> the demo's 8 input frames (stages in distributions.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import fitted as F

#: Generator department token -> demo department code.
DEPT_MAP = {"ob": "OB", "ib": "IB", "vret": "RT", "da": "RT", "icqa": "SP", "support": "SP"}


def to_demo_frames(md, horizon_hours: int = 24) -> dict[str, pd.DataFrame]:
    """Convert a `generate.MockDay` into the demo's 8 input frames over `horizon_hours`."""
    site = md.site
    ob = md.frames["ob_bl"].copy()
    ib = md.frames["mhc_ib"].copy()
    tph_src = md.frames["hourly_tph"].copy()
    shifts_src = md.frames["breaks_and_shifts"].copy()
    cpt_src = md.frames["ob_risk"].copy()

    # --- horizon: starts at the first work block after "now" ---
    work = shifts_src[shifts_src.interval_type == "shift"].sort_values(
        "interval_start_local"
    )
    future = work[work.interval_start_local > md.now]
    if future.empty:
        raise ValueError("no shift starts after 'now' - widen the generated window")
    start = future.iloc[0].interval_start_local
    end = start + pd.Timedelta(hours=horizon_hours)
    base_date = start.normalize()

    def _h(ts: pd.Series) -> pd.Series:
        """Timestamp -> the demo's decimal model hour, offset from the base date."""
        return (ts - base_date).dt.total_seconds() / 3600.0

    ob = ob[(ob.full_date >= start) & (ob.full_date < end)].sort_values("full_date")
    ib = ib[(ib.full_date >= start) & (ib.full_date < end)].sort_values("full_date")
    ob["hour"] = _h(ob.full_date).round().astype(float)
    ib["hour"] = _h(ib.full_date).round().astype(float)
    hours = list(ob.hour)

    # --- work blocks: each `shift` interval is a numbered block ---
    blocks = shifts_src[
        (shifts_src.interval_start_local >= start)
        & (shifts_src.interval_start_local < end)
    ].sort_values("interval_start_local").copy()
    blocks["start_hour"] = _h(blocks.interval_start_local).round(3)
    blocks["end_hour"] = _h(blocks.interval_end_local).round(3)
    blocks = blocks[blocks.end_hour <= horizon_hours + _h(pd.Series([start])).iloc[0] + 0.001]

    quarter, rows = 0, []
    for seg in blocks.itertuples():
        is_work = seg.interval_type == "shift"
        if is_work:
            quarter += 1
        rows.append(
            {
                "site": site,
                "shift": seg.shift_name,
                "quarter": quarter if is_work else None,
                "interval_kind": "work" if is_work else "closed",
                "start_hour": seg.start_hour,
                "end_hour": seg.end_hour,
            }
        )
    shifts = pd.DataFrame(rows)
    work_blocks = shifts[shifts.interval_kind == "work"]

    # --- tph ---
    proc_to_dept = {"OUTBOUND": "OB", "INBOUND": "IB", "VENDOR_RETURNS": "RT"}
    tph_src = tph_src[
        (tph_src.full_date_local >= start) & (tph_src.full_date_local < end)
    ].copy()
    tph_src["hour"] = _h(tph_src.full_date_local).round().astype(float)
    tph = (
        tph_src[tph_src.main_process.isin(proc_to_dept)]
        .assign(site=site, dept=lambda d: d.main_process.map(proc_to_dept))
        [["site", "hour", "dept", "tph"]]
    )

    # --- headcount: wide -> long ---
    headcount = _headcount_long(md, work_blocks, base_date, site)

    # --- planned capacity: recomputed, because block-level headcount shifts it and the
    # parameter builder asserts capacity == TPH x heads ---
    cap = _recompute_capacity(tph, headcount, shifts, hours)

    ob_plan = pd.DataFrame(
        {
            "site": site,
            "hour": hours,
            "ob_capacity_plan": [round(cap[("OB", h)], 3) for h in hours],
            "ob_arrivals": ob.nwa_flm.round(1).values,
            "ob_pickable_backlog_plan": ob.pickable_units_flm.round(3).values,
            "ob_units_total_plan": ob.units_total.round(3).values,
            # Off the recomputed capacity; a carried ceiling can sit below it (infeasible).
            "ob_max_machine": [
                round(cap[("OB", h)] * F.OB_MACHINE_CEILING_MULTIPLE, 3) for h in hours
            ],
            "bl_min_hours": ob.min_bl.values,
            "bl_max_hours": ob.max_bl.values,
        }
    )
    ib_plan = pd.DataFrame(
        {
            "site": site,
            "hour": hours,
            "ib_capacity_plan": [round(cap[("IB", h)], 3) for h in hours],
            "ib_arrivals": ib.arrivals_spark.round(1).values,
            "ib_backlog_plan": ib.backlog_units_spark.round(3).values,
            # Set from the carried window's backlog, same hazard as the outbound ceiling.
            "ib_max_storage": round(
                float(max(ib.backlog_units_spark.max(), 1.0))
                * F.IB_STORAGE_CEILING_MULTIPLE,
                3,
            ),
        }
    )
    # Transship folded into vendor returns (RT).
    returns_plan = pd.DataFrame(
        {
            "site": site,
            "hour": hours,
            "rt_capacity_plan": [round(cap[("RT", h)], 3) for h in hours],
            "rt_arrivals": (ob.charge_vret + ob.charge_da).round(1).values,
            "rt_backlog_plan": (ob.backlog_vret + ob.backlog_da).round(3).values,
        }
    )

    # --- cut-offs ---
    cpt = cpt_src[
        (cpt_src.cpt_local_time >= start) & (cpt_src.cpt_local_time < end)
    ].sort_values("cpt_local_time").copy()
    cpt["cpt_hour"] = _h(cpt.cpt_local_time).round().astype(float)
    cpts = pd.DataFrame(
        {
            "site": site,
            "cpt_index": range(1, len(cpt) + 1),
            "cpt_hour": cpt.cpt_hour.values,
            "cumulative_units_due_out": cpt.f_cumulative.round(1).values,
        }
    )

    # --- end of day ---
    eod_hour = next((h for h in hours if h % 24 == 23), hours[-1])
    den = md.frames["eod_bl_denominator"]
    eod_date = (base_date + pd.Timedelta(hours=float(eod_hour))).normalize()
    match = den[den.full_date.dt.normalize() == eod_date]
    daily = float(match.avg_ob_7_day_capacity_ipt.iloc[0]) if len(match) else float(
        den.avg_ob_7_day_capacity_ipt.iloc[-1]
    )
    per_hour = daily / 24.0
    guidance = md.frames["eod_bl_guidance"]
    dow = (eod_date.dayofweek + 1) % 7
    g = guidance[guidance.day_of_week == dow].iloc[0]
    # Thresholds are in days; the demo works in hours of cover.
    eod_guidance = pd.DataFrame(
        [
            {
                "site": site,
                "eod_hour": int(eod_hour),
                "eod_capacity_per_hour": round(per_hour, 3),
                "eod_soft_max_hours": round(float(g.soft_max_bl) * 24.0, 3),
                "eod_hard_min_hours": round(float(g.hard_min_bl) * 24.0, 3),
            }
        ]
    )

    return {
        "shifts": shifts,
        "headcount": headcount,
        "tph": tph.reset_index(drop=True),
        "ob_plan": ob_plan,
        "ib_plan": ib_plan,
        "returns_plan": returns_plan,
        "cpts": cpts,
        "eod_guidance": eod_guidance,
    }


def _headcount_long(md, work_blocks: pd.DataFrame, base_date, site: str) -> pd.DataFrame:
    """Collapse the wide matrix to one long row per (block, origin, skill group), read at each block's first whole hour."""
    from .generate import _parse_flm_column

    hc = md.frames["headcount_and_possible_movements"]
    rows = []
    for blk in work_blocks.itertuples():
        rep = int(np.ceil(blk.start_hour - 1e-9))
        ts = base_date + pd.Timedelta(hours=float(rep))
        row = hc[hc.date_time_local_hour == ts]
        if row.empty:
            continue
        row = row.iloc[0]
        absent_of = {
            d: float(row.get(f"perc_absenteeism_{d}", 0.0) or 0.0)
            for d in ("ob", "ib", "da")
        }
        buckets: dict[tuple[str, tuple[str, ...]], int] = {}
        for col, val in row.items():
            if not isinstance(col, str) or not col.startswith("flm_") or "_headcount" in col:
                continue
            origin, dests = _parse_flm_column(col)
            if dests is None or int(val or 0) <= 0:
                continue
            o = DEPT_MAP.get(origin)
            if o is None or o == "SP":
                continue  # Support has no native people
            skills = tuple(sorted({o} | {DEPT_MAP[d] for d in dests if d in DEPT_MAP}))
            key = (o, skills)
            buckets[key] = buckets.get(key, 0) + int(val)
        for (origin, skills), people in sorted(buckets.items()):
            src = "da" if origin == "RT" else origin.lower()
            absent = absent_of[src]
            rows.append(
                {
                    "site": site,
                    "quarter": int(blk.quarter),
                    "origin": origin,
                    "skills": "+".join(skills),
                    "people": people,
                    "absenteeism": round(absent, 4),
                    "available": int(round(people * (1.0 - absent))),
                }
            )
    return pd.DataFrame(rows)


def _recompute_capacity(
    tph: pd.DataFrame, headcount: pd.DataFrame, shifts: pd.DataFrame, hours: list[float]
) -> dict[tuple[str, float], float]:
    """Planned capacity as TPH x coverage-weighted available heads (the model's own expression)."""
    work = shifts[shifts.interval_kind == "work"]
    cover = {}
    for seg in work.itertuples():
        for h in hours:
            overlap = max(0.0, min(h + 1.0, seg.end_hour) - max(h, seg.start_hour))
            if overlap > 0:
                cover[(h, int(seg.quarter))] = overlap

    heads = (
        headcount.groupby(["quarter", "origin"])["available"].sum().to_dict()
    )
    tph_at = {(r.dept, r.hour): r.tph for r in tph.itertuples()}
    out: dict[tuple[str, float], float] = {}
    for dept in ("OB", "IB", "RT"):
        for h in hours:
            n = sum(
                heads.get((q, dept), 0) * c
                for (hh, q), c in cover.items()
                if hh == h
            )
            out[(dept, h)] = tph_at.get((dept, h), 0.0) * n
    return out
