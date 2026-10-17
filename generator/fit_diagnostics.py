#!/usr/bin/env python3
"""Publish the dimensionless fit diagnostics (persistence, indexed profiles, ratios) for the generated-data checks.

    python fit_diagnostics.py --out ../output

Writes fit_persistence.csv, fit_profiles.csv, fit_calibration.csv and fit_arrivals.csv.
The generated side is measured by running the generator, not transcribed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mock_inputs import fitted as F
from mock_inputs.generate import generate

SEEDS = 30
HOURS = 24

#: Fitted lag-1 autocorrelation of the de-seasonalised residual; levels persist, flows do not.
FITTED_PHI = [
    ("outbound capacity",       "level", F.OB_CAPACITY_AR1_PHI),
    ("pickable backlog",        "level", F.OB_PICKABLE_AR1_PHI),
    ("aged units",              "level", F.GAMBLER_AR1_PHI),
    ("arrivals (derived)",      "flow",  F.OB_ARRIVALS_AR1_PHI),
    ("ship plan",               "flow",  0.047),
]

#: Generated (frame, column) for each row above; `None` means computed in `collect`.
GENERATED_OF = {
    "outbound capacity":  ("ob_bl", "capacity_flm"),
    "pickable backlog":   ("ob_bl", "pickable_units_flm"),
    "aged units":         ("ob_bl", "eligible_units_gambler"),
    "arrivals (derived)": None,
    "ship plan":          ("ob_bl", "nwa_flm"),
}


def _lag1(series: np.ndarray, hod: np.ndarray) -> float:
    """Lag-1 autocorrelation after removing the hour profile (log residual if positive, else additive)."""
    s = np.asarray(series, dtype=float)
    profile = np.array([s[hod == h].mean() for h in range(HOURS)])
    if np.all(s > 0) and np.all(profile > 0):
        resid = np.log(s) - np.log(profile[hod])
    else:
        resid = s - profile[hod]
    resid = resid - resid.mean()
    denom = float((resid * resid).sum())
    return float((resid[:-1] * resid[1:]).sum() / denom) if denom else float("nan")


def collect(seeds: int = SEEDS):
    """Run the generator and collect profiles, persistence, arrivals and audits."""
    audits: list[dict[str, float]] = []
    phis: dict[str, list[float]] = {name: [] for name, _, _ in FITTED_PHI}
    cap_prof, bl_prof, arrivals = [], [], []

    for s in range(seeds):
        md = generate(seed=20261011 + s)
        ob = md.frames["ob_bl"].reset_index(drop=True)
        hod = ob["full_date"].dt.hour.to_numpy()
        audits.append(md.audit)

        for name, _, _ in FITTED_PHI:
            spec = GENERATED_OF[name]
            if spec is not None:
                phis[name].append(_lag1(ob[spec[1]].to_numpy(), hod))

        # Derived arrivals: A[t] = B[t+1] - B[t] + C[t]
        b = ob["pickable_units_flm"].to_numpy()
        c = ob["nwa_flm"].to_numpy()
        a = b[1:] - b[:-1] + c[:-1]
        phis["arrivals (derived)"].append(_lag1(a, hod[:-1]))
        arrivals.append(a / np.abs(a).mean())

        cap_prof.append([ob.loc[hod == h, "capacity_flm"].mean() for h in range(HOURS)])
        bl_prof.append([ob.loc[hod == h, "pickable_units_flm"].mean() for h in range(HOURS)])

    profiles = pd.DataFrame({
        "hour_of_day": np.arange(HOURS),
        "capacity_fitted": np.asarray(F.OB_CAPACITY_HOUR_SHARE),
        "capacity_generated": np.mean(cap_prof, axis=0),
        "backlog_fitted": np.asarray(F.OB_PICKABLE_HOUR_MEAN, dtype=float),
        "backlog_generated": np.mean(bl_prof, axis=0),
    })
    # Indexed to each series' own mean.
    for col in ("capacity_fitted", "capacity_generated", "backlog_fitted",
                "backlog_generated"):
        profiles[col] = profiles[col] / profiles[col].mean()

    return profiles, phis, np.concatenate(arrivals), pd.DataFrame(audits)


def calibration(audits: pd.DataFrame) -> pd.DataFrame:
    """Generated against fitted, as ratios."""
    rows = [
        ("outbound capacity, daily", "ob_capacity_daily_mean", F.OB_CAPACITY_DAILY_MEAN),
        ("outbound shipped, daily",  "ob_shipped_daily_mean",  F.OB_SHIPPED_DAILY_MEAN),
        ("arrivals, daily",          "ob_arrivals_daily_mean", F.OB_ARRIVALS_DAILY_MEAN),
        ("pickable share of total",  "pickable_share_of_total", F.PICKABLE_SHARE_OF_TOTAL),
        ("shipped / capacity, p90",  "ob_shipped_over_capacity_p90", 1.402),
        ("pickable backlog, mean",   "ob_pickable_mean",
         float(np.mean(F.OB_PICKABLE_HOUR_MEAN))),
        ("negative arrival share",   "ob_arrivals_negative_share", 0.053),
    ]
    out = []
    for label, key, want in rows:
        got = float(audits[key].mean())
        out.append({"quantity": label, "ratio": got / want,
                    "seed_sd_of_ratio": float(audits[key].std() / want)})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("../output"), type=Path)
    ap.add_argument("--seeds", default=SEEDS, type=int)
    args = ap.parse_args()
    data = args.out / "data"
    data.mkdir(parents=True, exist_ok=True)

    profiles, phis, arrivals, audits = collect(args.seeds)
    cal = calibration(audits)

    pers = pd.DataFrame([
        {"series": name, "kind": kind, "phi_fitted": phi,
         "phi_generated": float(np.mean(phis[name])) if phis[name] else float("nan")}
        for name, kind, phi in FITTED_PHI
    ])

    pers.round(4).to_csv(data / "fit_persistence.csv", index=False)
    profiles.round(6).to_csv(data / "fit_profiles.csv", index=False)
    cal.round(4).to_csv(data / "fit_calibration.csv", index=False)
    pd.DataFrame({"arrivals_index": arrivals}).round(6).to_csv(
        data / "fit_arrivals.csv", index=False)

    print(f"wrote 4 frames to {data}  ({args.seeds} seeds)\n")
    print(pers.round(3).to_string(index=False))
    print()
    print(cal.round(3).to_string(index=False))
    share = float(audits["ob_arrivals_negative_share"].mean())
    print(f"\nnegative arrival share, generated: {share:.3f}   fitted: 0.053")
    print("  (from the generator's audit. The arrivals frame spans closures too, so its"
          " own\n   negative share is a wider quantity and is not the one compared here.)")


if __name__ == "__main__":
    main()
