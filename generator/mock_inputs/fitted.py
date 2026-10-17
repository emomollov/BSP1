"""Generator parameters, tagged FITTED / STRUCTURAL / ASSUMED. Rationale: generator/distributions.md."""

from __future__ import annotations

# --- Site ---

#: ASSUMED. Synthetic site code.
SITE = "BSP1"


# --- Shift structure (distributions.md section 1) ---

#: (interval_type, start hour, end hour, shift name), tiled across every date. Local hours.
#: The solver matches `shift` and `closed` by name and treats any other type as a break,
#: so breaks use distinct per-quarter labels to exercise that path.
DAY_PATTERN = [
    ("shift", 0.0, 4.0, "NIGHT"),
    ("break_q4", 4.0, 5.0, "NIGHT"),
    ("closed", 5.0, 6.0, None),
    ("shift", 6.0, 9.0, "MORNING"),
    ("break_q1", 9.0, 10.0, "MORNING"),
    ("shift", 10.0, 13.0, "MORNING"),
    ("break_q2", 13.0, 14.0, "MORNING"),
    ("shift", 14.0, 20.0, "AFTERNOON"),
    ("break_q3", 20.0, 21.0, "AFTERNOON"),
    ("shift", 21.0, 22.0, "AFTERNOON"),
    ("shift", 22.0, 24.0, "NIGHT"),
]

#: FITTED.
IB_WINDOW = (6.0, 20.0)


# --- Outbound capacity (section 2) ---

OB_CAPACITY_DAILY_MEAN = 254_360.0
OB_CAPACITY_DAILY_CV = 0.148

#: FITTED. Share of daily capacity per local hour; sums to 1.
OB_CAPACITY_HOUR_SHARE = [
    0.0508, 0.0523, 0.0395, 0.0375, 0.0349, 0.0381, 0.0381, 0.0343,
    0.0339, 0.0342, 0.0362, 0.0343, 0.0376, 0.0439, 0.0494, 0.0449,
    0.0407, 0.0406, 0.0383, 0.0394, 0.0449, 0.0519, 0.0523, 0.0521,
]

#: FITTED. AR(1) on the multiplicative residual.
OB_CAPACITY_AR1_PHI = 0.904
OB_CAPACITY_RESID_SD = 0.191


# --- Outbound pickable arrivals (section 3) ---
# FITTED but derived via A[t] = B[t+1] - B[t] + C[t] (B = pickable backlog, C = shipped).

OB_ARRIVALS_DAILY_MEAN = 229_329.0

#: FITTED. Share of daily arrivals per local hour. Hour 5 is negative on purpose (closure
#: drains the pool); do not clip.
OB_ARRIVALS_HOUR_SHARE = [
    0.0530, 0.0543, 0.0422, 0.0338, 0.0205, -0.0094, 0.0188, 0.0328,
    0.0507, 0.0257, 0.0429, 0.0524, 0.0604, 0.0107, 0.0668, 0.0735,
    0.0626, 0.0520, 0.0530, 0.0655, 0.0449, 0.0280, 0.0630, 0.0590,
]

#: FITTED.
OB_ARRIVALS_AR1_PHI = 0.139


# --- Backlog composition (section 5) ---
# FITTED shares of `total_units`. The pickable series is drawn and the rest derived from it,
# so `units_total >= pickable_units` holds by construction.

PICKABLE_SHARE_OF_TOTAL = 0.544
GAMBLER_SHARE_OF_TOTAL = 0.119
TCAP_SHARE_OF_TOTAL = 0.157

#: FITTED.
GAMBLER_AR1_PHI = 0.977
GAMBLER_RESID_SD = 1.224

#: FITTED.
TCAP_AR1_PHI = 0.965
TCAP_RESID_SD = 0.449


#: FITTED. Hour-of-day mean of `total_pickable_units`, in units (section 4). The backlog is
#: drawn from this and arrivals derived from the flow identity.
OB_PICKABLE_HOUR_MEAN = [
    66_058, 67_049, 64_867, 63_970, 62_137, 63_588, 61_131, 57_531,
    54_329, 55_586, 54_502, 53_406, 54_165, 59_154, 57_394, 58_118,
    60_606, 61_761, 64_013, 64_611, 66_907, 69_123, 64_883, 64_453,
]

#: FITTED on the de-seasonalised residual.
OB_PICKABLE_AR1_PHI = 0.965
OB_PICKABLE_RESID_SD = 0.229

#: FITTED. Hour-of-day mean of `actual_units` (shipped), the C[t] term of the flow identity.
OB_SHIPPED_HOUR_MEAN = [
    10_681, 14_137, 10_190, 9_267, 3_069, 393, 7_743, 10_422,
    9_902, 6_747, 10_534, 10_787, 8_303, 4_110, 13_987, 13_704,
    12_629, 9_192, 11_063, 12_116, 7_680, 10_409, 14_312, 13_029,
]
OB_SHIPPED_DAILY_MEAN = 234_407.0


# --- Backlog band (section 6) ---
# FITTED. Hours of cover, not days (the end-of-day thresholds below are in days).

BL_MIN_HOURS = 3.5
BL_MAX_HOURS = 6.0


# --- End of day (section 7) ---
# FITTED mean; the soft max and hard min are configuration set around it, not drawn.

EOD_SOFT_MAX_DAYS = 0.750
EOD_HARD_MIN_DAYS = 0.430


# --- Throughput per hour (section 8) ---
# ASSUMED levels, FITTED shape.

TPH = {
    "OUTBOUND": 118.0,
    "INBOUND": 82.0,
    "VENDOR_RETURNS": 48.0,
    "DA": 64.0,
}

#: ASSUMED. +/- swing across a shift block, peaking mid-block.
TPH_INTRASHIFT_SWING = 0.08


# --- Headcount and cross-training (section 9) ---

#: ASSUMED. Share of each department's headcount per cross-training bucket, keyed by
#: destination set; `()` is native-only. Each origin MUST sum to 1.0 (see `_check`).
CROSS_TRAINING = {
    "OB": {(): 0.62, ("IB",): 0.20, ("IB", "DA"): 0.08, ("VRET",): 0.06, ("ICQA",): 0.04},
    "IB": {(): 0.59, ("OB",): 0.26, ("OB", "DA"): 0.10, ("VRET",): 0.05},
    "DA": {(): 0.58, ("OB",): 0.30, ("OB", "IB"): 0.12},
    "VRET": {(): 0.56, ("OB",): 0.34, ("OB", "IB"): 0.10},
}

#: ASSUMED. Departments other than outbound and inbound, as a share of outbound headcount.
DEPT_HEADCOUNT_SHARE_OF_OB = {"IB": 0.45, "DA": 0.14, "VRET": 0.10}

#: ASSUMED. Drawn per department per day.
ABSENTEEISM_MEAN = {"ob": 0.060, "ib": 0.055, "da": 0.050}
ABSENTEEISM_SD = 0.015


# --- Inbound (section 10) ---
# Capacity FITTED; arrivals, backlog and storage ceiling are STRUCTURAL off the capacity.

IB_CAPACITY_AR1_PHI = 0.904  # borrowed from outbound
IB_CAPACITY_RESID_SD = 0.191

#: STRUCTURAL. Just under 1 so the backlog does not trend.
IB_ARRIVALS_SHARE_OF_CAPACITY = 0.96

#: ASSUMED. Seed units; storage ceiling as a multiple of the peak backlog.
IB_BACKLOG_SEED = 120_000.0
IB_STORAGE_CEILING_MULTIPLE = 1.45

OB_MACHINE_CEILING_MULTIPLE = 1.30


# --- Carrier cut-offs (section 11) ---
# ASSUMED times, STRUCTURAL demand (a share of cumulative capacity to each cut-off).

#: ASSUMED. Local hours.
CPT_HOURS = [2.0, 7.0, 11.0, 14.0, 17.0, 19.0, 21.0, 23.0]

#: STRUCTURAL. Confirmed units due, as a share of cumulative capacity.
CPT_DEMAND_SHARE = 0.82


# --- Transship out and vendor returns (section 12), all ASSUMED ---

#: Share of outbound capacity.
TSO_CAPACITY_SHARE_OF_OB = 0.085
VRET_CAPACITY_SHARE_OF_OB = 0.055

TSO_BACKLOG_SEED = 26_000.0
VRET_BACKLOG_SEED = 31_000.0

#: Arrivals as a share of the department's own capacity.
TSO_ARRIVALS_SHARE_OF_CAPACITY = 0.95
VRET_ARRIVALS_SHARE_OF_CAPACITY = 0.95


# --- Invariants ---


def _check() -> None:
    """Fail at import on a parameter set the solver would reject (notably cross-training sums)."""
    for origin, buckets in CROSS_TRAINING.items():
        total = sum(buckets.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"CROSS_TRAINING[{origin!r}] sums to {total:.4f}, not 1.0. Every person "
                f"must be in exactly one bucket, including the native-only `()` bucket, "
                f"or the solver's headcount override raises IndexError at "
                f"the solver's Largest-Remainder apportionment."
            )
    if not (0.0 < sum(OB_CAPACITY_HOUR_SHARE) <= 1.001):
        raise ValueError("OB_CAPACITY_HOUR_SHARE must sum to about 1.0")
    if len(OB_PICKABLE_HOUR_MEAN) != 24 or len(OB_SHIPPED_HOUR_MEAN) != 24:
        raise ValueError("hour-of-day profiles must have 24 entries")


_check()
