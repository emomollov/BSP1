"""Named scenarios: each overrides a few knobs of the baseline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Scenario:
    """Knobs on the synthetic data generator.

    Defaults put the site inside every band; a scenario overrides one or two fields.
    """

    name: str
    headline: str


    # --- demand ---
    #: Multiplier on outbound arrivals, applied over `ob_surge_hours`.
    ob_arrival_mult: float = 1.0
    ob_surge_hours: tuple[float, ...] = ()

    #: Multiplier on inbound arrivals, applied over `ib_surge_hours`.
    ib_arrival_mult: float = 1.0
    ib_surge_hours: tuple[float, ...] = ()

    # --- deadlines ---
    #: Multiplier on the demand due at one cut-off, by index into config.CPT_HOURS.
    cpt_demand_mult: float = 1.0
    cpt_index: int = 0

    # --- ceilings ---
    #: Outbound machine ceiling as a share of planned capacity.
    ob_machine_share: float = 1.35

    #: Hours the ceiling in `ob_machine_share` applies to.
    ob_machine_peak_hours: tuple[float, ...] = ()

    #: Ceiling share outside the peak window (read only when peak hours are set). Never below
    #: 1.0 (no-move solve goes infeasible); too wide and `machine_capped` collapses into
    #: `ob_backlog_build`.
    ob_machine_offpeak_share: float = 1.35

    #: Inbound storage ceiling as a share of the peak planned inbound backlog.
    ib_storage_share: float = 1.60

    # --- people ---
    #: Share of the cross-trained pool available to move.
    xt_pool_share: float = 1.0

    # --- targets ---
    #: End-of-day soft maximum as a share of the baseline end-of-day backlog; below 1.0 the plan misses it.
    eod_soft_max_share: float = 1.45
    eod_hard_min_share: float = 0.35

    # --- data honesty ---
    #: Hours with no machine-ceiling reading (published as the sentinel plus an `*_observed` flag).
    unobserved_machine_hours: tuple[float, ...] = field(default_factory=tuple)


BASELINE = Scenario(
    name="baseline",
    headline="Nothing is wrong, and the plan is still not optimal",
)


SCENARIOS: dict[str, Scenario] = {
    "baseline": BASELINE,
    "ib_surge": replace(
        BASELINE,
        name="ib_surge",
        headline="Inbound arrivals double",
        ib_arrival_mult=2.0,
        ib_surge_hours=tuple(float(h) for h in range(8, 14)),
    ),
    "ob_backlog_build": replace(
        BASELINE,
        name="ob_backlog_build",
        headline="Outbound arrivals surge",
        # Sized to breach yet be relievable; a larger surge becomes a second `machine_capped`.
        ob_arrival_mult=1.25,
        ob_surge_hours=tuple(float(h) for h in range(7, 13)),
    ),
    "cpt_at_risk": replace(
        BASELINE,
        name="cpt_at_risk",
        headline="One carrier cut-off is at risk",
        # Applies to the incremental demand in that cut-off's bucket (see data._build_cpts).
        # Larger and it becomes a second `machine_capped`.
        cpt_demand_mult=1.45,
        cpt_index=1,
    ),
    "machine_capped": replace(
        BASELINE,
        name="machine_capped",
        headline="The outbound machine ceiling is the wall",
        # "Unfixable" means the breach cannot reach zero, not that none of it is recoverable.
        ob_arrival_mult=1.25,
        ob_surge_hours=tuple(float(h) for h in range(7, 13)),
        # At plan (not below: no-move solve goes infeasible) over a six-hour peak. A peak
        # window rather than the whole day, so the model has lighter hours to shift work into
        # and "the machine is the wall" is a finding, not arithmetic.
        ob_machine_share=1.00,
        ob_machine_peak_hours=tuple(float(h) for h in range(20, 26)),
        # Chosen, not derived; see `Scenario.ob_machine_offpeak_share` for its bounds.
        ob_machine_offpeak_share=1.02,
    ),
    "eod_target_miss": replace(
        BASELINE,
        name="eod_target_miss",
        headline="The end-of-day target is missed",
        eod_soft_max_share=0.80,
    ),
    "ib_then_eod": replace(
        BASELINE,
        name="ib_then_eod",
        headline="Inbound floods the morning, and the day still has to land",
        # The only scenario where `storage_now` fires. The surge stays in hours 6-11 so the
        # backlog peaks before the first shift ends at 14:00; later, it lands in `storage_later`.
        ib_arrival_mult=2.2,
        ib_surge_hours=tuple(float(h) for h in range(6, 11)),
        # 0.78, not lower: at 0.72 the two phases blur into churn (measured).
        eod_soft_max_share=0.78,
    ),
    "pool_exhausted": replace(
        BASELINE,
        name="pool_exhausted",
        headline="Almost nobody is cross-trained",
        ob_arrival_mult=1.25,
        ob_surge_hours=tuple(float(h) for h in range(7, 13)),
        # A tenth, not a half: the pool (about 26 per block) far exceeds the 3 the fix needs,
        # so a mild cut would silently become a second `ob_backlog_build`.
        xt_pool_share=0.10,
    ),
}


#: Display and publication order.
ORDER = [
    "baseline",
    "ib_surge",
    "ob_backlog_build",
    "cpt_at_risk",
    "eod_target_miss",
    "ib_then_eod",
    "machine_capped",
    "pool_exhausted",
]


def get(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise SystemExit(f"unknown scenario {name!r}; choose from {', '.join(ORDER)} or 'all'")


def resolve(names: list[str]) -> list[Scenario]:
    """Resolve CLI scenario names, expanding 'all' into ORDER."""
    if not names or names == ["all"]:
        return [SCENARIOS[n] for n in ORDER]
    return [get(n) for n in names]
