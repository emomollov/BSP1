"""Site geometry, departments, constraint thresholds and display vocabulary."""

from __future__ import annotations

# --- departments ---

OB = "OB"  # Outbound - pick, pack, ship
IB = "IB"  # Inbound  - receive, stow
RT = "RT"  # Returns  - process returned units
SP = "SP"  # Support  - the sink for spare people

DEPTS = [OB, IB, RT, SP]

#: Departments with capacity. Support produces nothing, so parking people there is a tie-break, not a cost.
PRODUCING = [OB, IB, RT]

#: Display names; every chart label follows.
VOCABULARY = {OB: "Outbound", IB: "Inbound", RT: "Returns", SP: "Support"}

SITE = "BSP1"


# --- site geometry ---
# Model time is decimal hours from midnight, so indices run past 24 into the next day.
# Each shift's break splits it into two work blocks, each its own decision point.

SHIFTS = [
    # (shift name, start, break start, break end, end)
    ("A", 6.0, 10.0, 11.0, 14.0),
    ("B", 14.0, 18.0, 19.0, 22.0),
    ("C", 22.0, 26.0, 27.0, 30.0),
]

#: Horizon is anchored on a shift start.
HORIZON_START = 6.0
HORIZON_HOURS = 24
HORIZON_END = HORIZON_START + HORIZON_HOURS  # 30.0

#: Carrier cut-offs inside the horizon, in model hours.
CPT_HOURS = [11.0, 15.0, 19.0, 23.0, 27.0, 29.0]

#: End-of-day hour.
EOD_HOUR = 23


# --- constraint thresholds ---

THRESHOLDS = {
    # Inbound storage; 1.0 = at the ceiling.
    "storage_limit": 1.00,

    # Bottom-guardrail line, below the ceiling.
    "storage_buffer": 0.90,

    # Hours of cover either side of the band centre where the `ob_bl_centre` guardrail is zero.
    "ob_bl_centre_deadband_hours": 0.5,

    # Cumulative demand must stay under this share of cumulative capacity, leaving headroom.
    "cpt_risk": 0.95,

    # Returns backlog ceiling, in days of forward cover.
    "returns_bl_days": 3.0,

    # Relative cost of a movement.
    "move_cost": 10.0,
}

#: Hours in the forward window turning a backlog in units into hours of cover.
BL_WINDOW_HOURS = 3

#: Forward window for the returns days-of-cover ceiling.
RETURNS_WINDOW_HOURS = 24


# --- sentinels ---
#: Published as-is, so a missing machine reading means "no ceiling known".
NO_MACHINE_LIMIT = 100_000.0


# --- solver ---
#: MIP gaps, far below the 1e-4 default: the ladder freezes each tier's value as a
#: constraint, so it must be reproducible across re-solves or the freeze reads infeasible.
SOLVER_GAP_REL = 1e-9
SOLVER_GAP_ABS = 1e-6

#: Ladder freeze tolerance. Above the solver gap (or re-solves fall outside the freeze),
#: yet small enough that a lower tier cannot buy gains by degrading a higher one.
FREEZE_EPS_ABS = 1e-4
FREEZE_EPS_REL = 1e-7

#: Looser relative tolerance for `guardrail` tiers only.
FREEZE_EPS_REL_GUARDRAIL = 1e-2

#: Finite stand-in for infinity, in units: lower bound on `model.V_raw` (backlog + signed
#: arrivals can go negative) and a sanity bound checked in `data.build`.
BIG_M = 5_000_000.0

#: The gurobipy pip licence caps model size and fails at `optimize()`;
#: `model._check_licence_size` turns that into a readable message.
LICENCE_MAX_VARS = 2000
LICENCE_MAX_CONSTRS = 2000


def display(dept: str) -> str:
    """Department code -> display name."""
    return VOCABULARY.get(dept, dept)


def hour_label(h: float) -> str:
    """Model hour -> a clock label. Hour 27 reads as 03:00, not 27:00."""
    return f"{int(h) % 24:02d}:00"
