"""Generate the 15 solver input frames for one site (chain and rationale: distributions.md)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import fitted as F


@dataclass
class MockDay:
    """One site's generated window."""

    site: str
    #: Local timestamps, hourly, covering the whole window.
    hours: pd.DatetimeIndex
    #: Local "now", when the solver is invoked.
    now: pd.Timestamp
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Diagnostics of what was drawn, used by `fit_diagnostics.py`.
    audit: dict[str, float] = field(default_factory=dict)


# --- Stochastic primitives ---


def _lognormal_unit_mean(rng: np.random.Generator, sd: float, size: int) -> np.ndarray:
    """Lognormal multipliers with mean 1 (centred on -sd^2/2; distributions.md section 0.2)."""
    return rng.lognormal(mean=-0.5 * sd * sd, sigma=sd, size=size)


def _ar1_multiplier(
    rng: np.random.Generator, n: int, phi: float, resid_sd: float
) -> np.ndarray:
    """AR(1) multiplier with unit mean and the given stationary sd (section 0.1).

    Innovation sd is scaled by sqrt(1 - phi^2); built in log space so it stays positive.
    """
    log_sd = np.log1p(resid_sd)
    innov = log_sd * np.sqrt(max(1.0 - phi * phi, 1e-9))
    x = np.empty(n)
    x[0] = rng.normal(0.0, log_sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, innov)
    return np.exp(x - 0.5 * log_sd * log_sd)


# --- Geometry ---


def _window(start_date: str, days_before: int, days_after: int) -> pd.DatetimeIndex:
    base = pd.Timestamp(start_date).normalize()
    lo = base - pd.Timedelta(days=days_before)
    hi = base + pd.Timedelta(days=days_after + 1)
    return pd.date_range(lo, hi, freq="h", inclusive="left")


def _segments(hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Tile the fitted daily pattern across every date in the window."""
    rows = []
    for day in pd.date_range(hours[0].normalize(), hours[-1].normalize(), freq="D"):
        for kind, lo, hi, shift_name in F.DAY_PATTERN:
            rows.append(
                {
                    "day": day.date().isoformat(),
                    "shift_name": shift_name or "CLOSED",
                    "start": day + pd.Timedelta(hours=lo),
                    "end": day + pd.Timedelta(hours=hi),
                    "interval_type": kind,
                }
            )
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def _coverage_of(hours: pd.DatetimeIndex, segs: pd.DataFrame, kind: str) -> np.ndarray:
    """Share of each hour covered by intervals of the given `interval_type`."""
    sel = segs[segs.interval_type == kind] if kind != "break" else segs[
        ~segs.interval_type.isin(["shift", "closed"])
    ]
    out = np.zeros(len(hours))
    for i, h in enumerate(hours):
        h_end = h + pd.Timedelta(hours=1)
        cov = 0.0
        for s in sel.itertuples():
            lo = max(h, s.start)
            hi = min(h_end, s.end)
            if hi > lo:
                cov += (hi - lo).total_seconds() / 3600.0
        out[i] = min(cov, 1.0)
    return out


# --- The chain ---


def generate(
    start_date: str = "2026-06-15",
    days_before: int = 1,
    days_after: int = 5,
    now_hour: int = 5,
    seed: int = 20261011,
) -> MockDay:
    """Generate the input frames for one site. `now_hour=5` falls in the closure, so the first shift starts at 06:00."""
    rng = np.random.default_rng(seed)
    hours = _window(start_date, days_before, days_after)
    segs = _segments(hours)
    now = pd.Timestamp(start_date).normalize() + pd.Timedelta(hours=now_hour)
    n = len(hours)
    hod = np.array([h.hour for h in hours])
    day_key = np.array([h.normalize().value for h in hours])
    md = MockDay(site=F.SITE, hours=hours, now=now)

    # --- day-level multiplier: one draw per date, shared by all series ---
    uniq_days = np.unique(day_key)
    day_mult = dict(
        zip(uniq_days, _lognormal_unit_mean(rng, F.OB_CAPACITY_DAILY_CV, len(uniq_days)))
    )
    L = np.array([day_mult[d] for d in day_key])

    # A break is not a closure: only `closed` hours produce nothing; shipping continues
    # through breaks (zeroing them overshoots the negative-arrival share).
    closed = _coverage_of(hours, segs, "closed")
    on_break = _coverage_of(hours, segs, "break")
    producing = closed < 0.999

    # --- 1. outbound capacity ---
    # Zero closed hours, then renormalise the share per day so the daily level is preserved.
    ob_share = np.array(F.OB_CAPACITY_HOUR_SHARE)[hod] * producing
    for d in uniq_days:
        m = day_key == d
        tot = ob_share[m].sum()
        if tot > 0:
            ob_share[m] = ob_share[m] / tot
    ob_cap = (
        F.OB_CAPACITY_DAILY_MEAN
        * ob_share
        * L
        * _ar1_multiplier(rng, n, F.OB_CAPACITY_AR1_PHI, F.OB_CAPACITY_RESID_SD)
    )

    # --- 2. throughput per hour ---
    tph = {}
    for proc, level in F.TPH.items():
        swing = 1.0 + F.TPH_INTRASHIFT_SWING * np.sin(np.pi * (hod % 8) / 8.0)
        tph[proc] = level * swing * (1.0 + rng.normal(0, 0.01, n))

    # --- 3. headcount = capacity / TPH ---
    with np.errstate(divide="ignore", invalid="ignore"):
        ob_heads = np.where(ob_cap > 0, np.round(ob_cap / tph["OUTBOUND"]), 0.0)
    heads = {"ob": ob_heads}
    for dept, share in F.DEPT_HEADCOUNT_SHARE_OF_OB.items():
        key = dept.lower()
        if key == "ib":
            in_window = (hod >= F.IB_WINDOW[0]) & (hod < F.IB_WINDOW[1]) & producing
            heads["ib"] = np.where(in_window, np.round(ob_heads * share), 0.0)
        else:
            heads[key] = np.where(producing, np.round(ob_heads * share), 0.0)

    # --- 4. inbound capacity ---
    ib_cap = heads["ib"] * tph["INBOUND"]
    ib_cap *= _ar1_multiplier(rng, n, F.IB_CAPACITY_AR1_PHI, F.IB_CAPACITY_RESID_SD)

    # --- 5. pickable backlog, anchored on its fitted profile (section 0.3) ---
    pickable = (
        np.array(F.OB_PICKABLE_HOUR_MEAN, dtype=float)[hod]
        * L
        * _ar1_multiplier(rng, n, F.OB_PICKABLE_AR1_PHI, F.OB_PICKABLE_RESID_SD)
    )

    # --- 6. shipped, then arrivals derived from the flow identity ---
    shipped = (
        np.array(F.OB_SHIPPED_HOUR_MEAN, dtype=float)[hod]
        * L
        * _ar1_multiplier(rng, n, F.OB_CAPACITY_AR1_PHI, 0.18)
    )
    # Capped by the pool, not by capacity: fitted utilisation exceeds 1 in many hours.
    shipped = np.minimum(shipped, np.maximum(pickable, 0.0))

    # A[t] = B[t+1] - B[t] + C[t]. Stays signed: negative hours are intentional.
    ob_arrivals = np.empty(n)
    ob_arrivals[:-1] = pickable[1:] - pickable[:-1] + shipped[:-1]
    # No B[t+1] for the last hour: fall back to the fitted profile.
    ob_arrivals[-1] = (
        F.OB_ARRIVALS_DAILY_MEAN * F.OB_ARRIVALS_HOUR_SHARE[hod[-1]] * L[-1]
    )
    seed_bl = float(pickable[0])

    # --- 7. backlog composition ---
    gambler = (
        pickable
        * (F.GAMBLER_SHARE_OF_TOTAL / F.PICKABLE_SHARE_OF_TOTAL)
        * _ar1_multiplier(rng, n, F.GAMBLER_AR1_PHI, min(F.GAMBLER_RESID_SD, 0.9))
    )
    tcap = (
        pickable
        * (F.TCAP_SHARE_OF_TOTAL / F.PICKABLE_SHARE_OF_TOTAL)
        * _ar1_multiplier(rng, n, F.TCAP_AR1_PHI, F.TCAP_RESID_SD)
    )
    # Built up from its parts so `units_total >= pickable_units_flm` holds.
    wip_share = 1.0 - (
        F.PICKABLE_SHARE_OF_TOTAL + F.GAMBLER_SHARE_OF_TOTAL + F.TCAP_SHARE_OF_TOTAL
    )
    units_total = (pickable + gambler + tcap) / max(1.0 - wip_share, 0.5)

    # --- 8. inbound backlog ---
    ib_arrivals = ib_cap * F.IB_ARRIVALS_SHARE_OF_CAPACITY
    # Arrivals continue through closed hours, which is why the storage ceiling binds.
    daily_ib = ib_arrivals.sum() / max(len(uniq_days), 1)
    ib_arrivals = np.where(
        ib_cap > 0, ib_arrivals * 0.75, daily_ib / 24.0 * 0.55
    )
    ib_backlog = np.empty(n)
    b = F.IB_BACKLOG_SEED
    for i in range(n):
        ib_backlog[i] = b
        b = max(b + ib_arrivals[i] - min(ib_cap[i], b + ib_arrivals[i]), 0.0)
    ib_storage = float(ib_backlog.max() * F.IB_STORAGE_CEILING_MULTIPLE)

    # --- 9. side flows ---
    tso_cap = ob_cap * F.TSO_CAPACITY_SHARE_OF_OB
    vret_cap = ob_cap * F.VRET_CAPACITY_SHARE_OF_OB
    tso_bl, vret_bl = np.empty(n), np.empty(n)
    bt, bv = F.TSO_BACKLOG_SEED, F.VRET_BACKLOG_SEED
    tso_arr = tso_cap * F.TSO_ARRIVALS_SHARE_OF_CAPACITY
    vret_arr = vret_cap * F.VRET_ARRIVALS_SHARE_OF_CAPACITY
    for i in range(n):
        tso_bl[i], vret_bl[i] = bt, bv
        bt = max(bt + tso_arr[i] - min(tso_cap[i], bt + tso_arr[i]), 0.0)
        bv = max(bv + vret_arr[i] - min(vret_cap[i], bv + vret_arr[i]), 0.0)

    md.audit = {
        "hours_closed": float(closed.sum()),
        "hours_on_break": float(on_break.sum()),
        "ob_capacity_daily_mean": float(ob_cap.sum() / len(uniq_days)),
        "ob_arrivals_daily_mean": float(ob_arrivals.sum() / len(uniq_days)),
        "ob_arrivals_negative_share": float((ob_arrivals < 0).mean()),
        "ob_pickable_mean": float(pickable.mean()),
        "ob_shipped_daily_mean": float(shipped.sum() / len(uniq_days)),
        "ob_shipped_over_capacity_p90": float(
            np.nanpercentile(
                np.where(ob_cap > 0, shipped / np.where(ob_cap > 0, ob_cap, np.nan), np.nan),
                90,
            )
        ),
        "ob_heads_peak": float(ob_heads.max()),
        "ib_storage_ceiling": ib_storage,
        "pickable_share_of_total": float((pickable / units_total).mean()),
        "seed_backlog": float(seed_bl),
    }

    # --- emit ---
    md.frames = _build_frames(
        md, rng, segs, hours, ob_cap, ob_arrivals, pickable, gambler, units_total,
        ib_arrivals, ib_backlog, tso_bl, tso_arr, vret_bl, vret_arr, heads, tph,
    )
    return md


def _build_frames(
    md, rng, segs, hours, ob_cap, ob_arrivals, pickable, gambler, units_total,
    ib_arrivals, ib_backlog, tso_bl, tso_arr, vret_bl, vret_arr, heads, tph,
) -> dict[str, pd.DataFrame]:
    """The frames `to_demo` and `fit_diagnostics` read, with only the columns they use."""
    out: dict[str, pd.DataFrame] = {}

    out["breaks_and_shifts"] = pd.DataFrame(
        {
            "shift_name": segs.shift_name,
            "interval_start_local": segs.start,
            "interval_end_local": segs.end,
            "interval_type": segs.interval_type,
        }
    )

    # Consumes `rng`, so it must come after the draws above and before any other draw.
    out["headcount_and_possible_movements"] = _headcount_frame(rng, hours, heads)
    out["ob_risk"] = _cpt_frame(hours, ob_cap, md.now)

    out["mhc_ib"] = pd.DataFrame(
        {
            "full_date": hours,
            "arrivals_spark": ib_arrivals.round(1),
            "backlog_units_spark": ib_backlog.round(1),
        }
    )

    out["ob_bl"] = pd.DataFrame(
        {
            "full_date": hours,
            "capacity_flm": ob_cap.round(1),
            "nwa_flm": ob_arrivals.round(1),
            "pickable_units_flm": pickable.round(1),
            "min_bl": F.BL_MIN_HOURS,
            "max_bl": F.BL_MAX_HOURS,
            "backlog_da": tso_bl.round(1),
            "charge_da": tso_arr.round(1),
            "backlog_vret": vret_bl.round(1),
            "charge_vret": vret_arr.round(1),
            "eligible_units_gambler": gambler.round(1),
            "units_total": units_total.round(1),
        }
    )

    daily_cap = (
        pd.Series(ob_cap, index=hours).groupby(lambda t: t.normalize()).sum()
    )
    # 7-day trailing average.
    avg7 = daily_cap.rolling(7, min_periods=1).mean()
    out["eod_bl_denominator"] = pd.DataFrame(
        {"full_date": avg7.index, "avg_ob_7_day_capacity_ipt": avg7.values.round(1)}
    )

    # Day of week with SUNDAY = 0, matching the `(dayofweek + 1) % 7` lookup in `to_demo`.
    out["eod_bl_guidance"] = pd.DataFrame(
        {
            "day_of_week": list(range(7)),
            "soft_max_bl": [round(F.EOD_SOFT_MAX_DAYS * m, 3)
                            for m in (1.15, 0.95, 0.95, 0.95, 1.0, 1.05, 1.10)],
            "hard_min_bl": [round(F.EOD_HARD_MIN_DAYS * m, 3)
                            for m in (1.10, 0.95, 0.95, 0.95, 1.0, 1.0, 1.05)],
        }
    )

    out["hourly_tph"] = pd.concat(
        [
            pd.DataFrame(
                {
                    "full_date_local": hours,
                    "main_process": proc,
                    "tph": np.round(series, 2),
                }
            )
            for proc, series in tph.items()
        ],
        ignore_index=True,
    )
    return out


def _headcount_frame(
    rng: np.random.Generator, hours: pd.DatetimeIndex, heads: dict
) -> pd.DataFrame:
    """The cross-training matrix, one column per `flm_<origin>_<dest...>` bucket for the origins `to_demo` reads.

    The `_headcount` totals are needed for the rounding sweep below and dropped before returning.
    """
    from .columns import HEADCOUNT_COLUMNS

    n = len(hours)
    day_of = np.array([h.normalize().value for h in hours])
    absent = {}
    for dept, mean in F.ABSENTEEISM_MEAN.items():
        per_day = {
            d: float(np.clip(rng.normal(mean, F.ABSENTEEISM_SD), 0.0, 0.25))
            for d in np.unique(day_of)
        }
        absent[dept] = np.array([per_day[d] for d in day_of])

    frame = {"date_time_local_hour": hours}
    # origin -> bucket columns, for the rounding sweep below.
    buckets: dict[str, list[str]] = {}
    native_col: dict[str, str] = {}
    totals_col: dict[str, str] = {}
    for col in HEADCOUNT_COLUMNS:
        if col == "date_time_local_hour":
            continue
        if col.startswith("perc_absenteeism_"):
            frame[col] = absent[col.removeprefix("perc_absenteeism_")].round(4)
            continue
        origin, dests = _parse_flm_column(col)
        base = heads.get(origin, np.zeros(n))
        if dests is None:
            frame[col] = base.astype(int)
            totals_col[origin] = col
            continue
        buckets.setdefault(origin, []).append(col)
        if not dests:
            native_col[origin] = col
        share = F.CROSS_TRAINING.get(origin.upper(), {}).get(
            tuple(d.upper() for d in dests), 0.0
        )
        frame[col] = np.round(base * share).astype(int)

    # Sweep rounding residue into the native-only bucket so buckets sum exactly to the
    # headcount column (the solver's Largest-Remainder loop requires it).
    for origin, cols in buckets.items():
        if origin not in totals_col or origin not in native_col:
            continue
        total = frame[totals_col[origin]]
        allocated = sum(frame[c] for c in cols)
        frame[native_col[origin]] = frame[native_col[origin]] + (total - allocated)
        frame[native_col[origin]] = np.maximum(frame[native_col[origin]], 0)
    return pd.DataFrame(frame).drop(columns=list(totals_col.values()))


def _parse_flm_column(col: str) -> tuple[str, tuple[str, ...] | None]:
    """`flm_ob_support_ib_da` -> ("ob", ("ib", "da")); `support` is a fixed word, not a destination."""
    body = col.removeprefix("flm_")
    if body.endswith("_headcount"):
        return body.removesuffix("_headcount"), None
    parts = body.split("_")
    origin = parts[0]
    rest = tuple(p for p in parts[1:] if p != "support")
    return origin, rest


def _cumsum_from(values: np.ndarray, hours: pd.DatetimeIndex, now: pd.Timestamp) -> np.ndarray:
    """Cumulative sum that restarts at `now` and is zero before it."""
    mask = np.asarray(hours >= now, dtype=float)
    return np.cumsum(values * mask) * mask


def _cpt_frame(
    hours: pd.DatetimeIndex, ob_cap: np.ndarray, now: pd.Timestamp
) -> pd.DataFrame:
    """Confirmed units due per carrier cut-off, from cumulative outbound capacity."""
    cum = _cumsum_from(ob_cap, hours, now)
    rows = []
    for i, h in enumerate(hours):
        if h < now:
            continue
        if h.hour not in F.CPT_HOURS and float(h.hour) not in F.CPT_HOURS:
            continue
        f_cum = cum[i] * F.CPT_DEMAND_SHARE
        rows.append({"cpt_local_time": h, "f_cumulative": round(f_cum, 1)})
    return pd.DataFrame(rows)
