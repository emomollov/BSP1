"""Curated seeds with their recorded (measured, not chosen) outcomes; `run_fitted.py` replays them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Day:
    """Exact recorded outcome of a seed; any change means model behaviour moved."""

    moves: int
    tiers_breached: int
    tiers_resolved: int
    limit: str


#: The curated demo set, keyed by seed.
DEMO_SEEDS: dict[int, Day] = {
    90025: Day(
        moves=95,
        tiers_breached=4,
        tiers_resolved=1,
        limit='cross_trained_people',
    ),
    90044: Day(
        moves=97,
        tiers_breached=4,
        tiers_resolved=2,
        limit='resolved',
    ),
    90047: Day(
        moves=94,
        tiers_breached=5,
        tiers_resolved=2,
        limit='cross_trained_people',
    ),
    90062: Day(
        moves=97,
        tiers_breached=5,
        tiers_resolved=2,
        limit='resolved',
    ),
    90073: Day(
        moves=68,
        tiers_breached=6,
        tiers_resolved=0,
        limit='cross_trained_people',
    ),
    90121: Day(
        moves=97,
        tiers_breached=5,
        tiers_resolved=1,
        limit='resolved',
    ),
    90123: Day(
        moves=94,
        tiers_breached=4,
        tiers_resolved=0,
        limit='cross_trained_people',
    ),
    90127: Day(
        moves=91,
        tiers_breached=5,
        tiers_resolved=1,
        limit='machine',
    ),
    90130: Day(
        moves=97,
        tiers_breached=4,
        tiers_resolved=0,
        limit='resolved',
    ),
    90133: Day(
        moves=324,
        tiers_breached=6,
        tiers_resolved=1,
        limit='resolved',
    ),
    90151: Day(
        moves=109,
        tiers_breached=4,
        tiers_resolved=1,
        limit='cross_trained_people',
    ),
    90173: Day(
        moves=36,
        tiers_breached=3,
        tiers_resolved=1,
        limit='resolved',
    ),
    90188: Day(
        moves=98,
        tiers_breached=5,
        tiers_resolved=0,
        limit='not_constrained',
    ),
}

#: Every seed in this range has been run and solves.
VERIFIED_RANGE = (90000, 90200)


def curated() -> list[int]:
    """Seeds in the curated set, in order."""
    return sorted(DEMO_SEEDS)


def expected(seed: int) -> Day:
    try:
        return DEMO_SEEDS[seed]
    except KeyError:
        raise KeyError(
            f"seed {seed} is not in the curated set. Verified seeds are "
            f"range{VERIFIED_RANGE}; the curated subset is {curated()}."
        ) from None


def is_verified(seed: int) -> bool:
    lo, hi = VERIFIED_RANGE
    return lo <= seed < hi
