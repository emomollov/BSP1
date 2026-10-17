"""The optimisation model, solver backend and lexicographic ladder."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Protocol

from . import config as cfg
from .data import Params


# --- Solver backend ---


class Status(Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    OTHER = "other"


class TierInfeasible(RuntimeError):
    """A tier of the ladder did not solve."""

    def __init__(self, tier: str, status: Status):
        super().__init__(f"objective tier {tier!r} did not solve: {status.value}")
        self.tier = tier
        self.status = status


class Backend(Protocol):
    """The solver surface the model may use."""

    def int_var(self, name: str, lb: float = 0.0, ub: float | None = None) -> Any: ...
    def num_var(self, name: str, lb: float = 0.0, ub: float | None = None) -> Any: ...
    def xsum(self, terms: Iterable[Any]) -> Any: ...
    def add(self, constraint: Any, name: str) -> None: ...
    def add_min(self, target: Any, args: list[Any], name: str) -> None: ...
    def add_max_zero(self, target: Any, arg: Any, name: str) -> None: ...
    def set_objective(self, expr: Any) -> None: ...
    def solve(self) -> Status: ...
    def value(self, expr: Any) -> float: ...
    def report_size(self) -> tuple[int, int]: ...


# --- Gurobi ---

_ENV: Any = None


def _env() -> Any:
    global _ENV
    if _ENV is None:
        import gurobipy as gp

        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


class GurobiBackend:
    """Gurobi backend."""

    name = "gurobi"

    def __init__(self, label: str = "capacity_allocator", msg: bool = False):
        import gurobipy as gp

        self._gp = gp
        self._m = gp.Model(label, env=_env())
        self._m.Params.OutputFlag = 1 if msg else 0
        self._m.Params.MIPGap = cfg.SOLVER_GAP_REL
        self._m.Params.MIPGapAbs = cfg.SOLVER_GAP_ABS

    def int_var(self, name, lb=0.0, ub=None):
        return self._m.addVar(
            vtype=self._gp.GRB.INTEGER,
            lb=lb,
            ub=self._gp.GRB.INFINITY if ub is None else ub,
            name=name,
        )

    def num_var(self, name, lb=0.0, ub=None):
        return self._m.addVar(
            lb=lb, ub=self._gp.GRB.INFINITY if ub is None else ub, name=name
        )


    def xsum(self, terms):
        return self._gp.quicksum(terms)

    def add(self, constraint, name):
        self._m.addConstr(constraint, name=name)

    def add_min(self, target, args, name):
        """target == min(args), exactly."""
        self._m.addGenConstrMin(target, args, name=name)

    def add_max_zero(self, target, arg, name):
        """target == max(arg, 0), exactly, where `arg` may be negative."""
        self._m.addGenConstrMax(target, [arg], constant=0.0, name=name)

    def set_objective(self, expr):
        self._m.setObjective(expr, self._gp.GRB.MINIMIZE)

    def solve(self):
        self._m.optimize()
        if self._m.Status == self._gp.GRB.OPTIMAL:
            return Status.OPTIMAL
        if self._m.Status == self._gp.GRB.INFEASIBLE:
            return Status.INFEASIBLE
        return Status.OTHER

    def value(self, expr):
        return float(expr.getValue() if hasattr(expr, "getValue") else expr.X)

    def report_size(self):
        self._m.update()
        return self._m.NumVars, self._m.NumConstrs


def make_backend(label: str = "capacity_allocator", msg: bool = False) -> Backend:
    try:
        return GurobiBackend(label, msg)
    except ImportError as exc:
        raise SystemExit(
            "gurobipy is not installed. `pip install gurobipy` gives a size-limited "
            "licence that needs no licence file and is enough for this model."
        ) from exc


def _check_licence_size(backend: Backend) -> None:
    n_vars, n_cons = backend.report_size()
    if n_vars <= cfg.LICENCE_MAX_VARS and n_cons <= cfg.LICENCE_MAX_CONSTRS:
        return
    raise SystemExit(
        f"model is {n_vars} variables / {n_cons} constraints, over the "
        f"{cfg.LICENCE_MAX_VARS} / {cfg.LICENCE_MAX_CONSTRS} limit of the gurobipy pip "
        "licence. Either shorten the horizon, drop a department, or install a full "
        "Gurobi licence."
    )


# --- The lexicographic ladder ---


@dataclass
class Tier:
    """One rung of the ladder."""

    name: str
    expr: Any
    label: str
    kind: str = "slack"


@dataclass
class LadderResult:
    attained: dict[str, float] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)


def solve_lexicographic(backend: Backend, tiers: list[Tier]) -> LadderResult:
    """Solve the tiers in strict priority order, highest priority first."""
    result = LadderResult()
    _check_licence_size(backend)
    for tier in tiers:
        backend.set_objective(tier.expr)
        status = backend.solve()
        if status is not Status.OPTIMAL:
            raise TierInfeasible(tier.name, status)
        v = backend.value(tier.expr)
        result.attained[tier.name] = v
        result.order.append(tier.name)
        result.labels[tier.name] = tier.label
        result.kinds[tier.name] = tier.kind
        rel = (
            cfg.FREEZE_EPS_REL_GUARDRAIL
            if tier.kind == "guardrail"
            else cfg.FREEZE_EPS_REL
        )
        slack = cfg.FREEZE_EPS_ABS + rel * abs(v)
        backend.add(tier.expr <= v + slack, f"freeze_{tier.name}")
    return result


# --- Model definitions ---


@dataclass
class Model:
    """The built model: variable handles plus the objective ladder."""

    backend: Backend
    params: Params

    #: Pin every movement to zero (the BEFORE solve). A field so every stage in `STAGES`
    #: takes the same argument.
    freeze_moves: bool = False

    # people
    P: dict[tuple[int, str], Any] = field(default_factory=dict)
    y: dict[tuple[int, str, tuple[str, ...], str], Any] = field(default_factory=dict)
    H: dict[tuple[int, str, tuple[str, ...]], Any] = field(default_factory=dict)
    X: dict[tuple[int, str, str], Any] = field(default_factory=dict)

    # flow
    Hcap: dict[tuple[str, float], Any] = field(default_factory=dict)
    V: dict[tuple[str, float], Any] = field(default_factory=dict)
    V_raw: dict[tuple[str, float], Any] = field(default_factory=dict)
    U: dict[tuple[str, float], Any] = field(default_factory=dict)
    Y: dict[tuple[str, float], Any] = field(default_factory=dict)
    U_total: dict[float, Any] = field(default_factory=dict)

    # windows
    S_BL: dict[float, Any] = field(default_factory=dict)
    C_RT: dict[float, Any] = field(default_factory=dict)
    H_cum: dict[int, Any] = field(default_factory=dict)

    # slacks
    o_storage: dict[float, Any] = field(default_factory=dict)
    v_minus: dict[float, Any] = field(default_factory=dict)
    v_plus: dict[float, Any] = field(default_factory=dict)
    g_cpt: dict[int, Any] = field(default_factory=dict)
    w_rt: dict[float, Any] = field(default_factory=dict)
    e_eod_max: Any = None
    e_eod_min: Any = None

    # bottom guardrails - slacks against targets tighter than the operational ones
    #: Absolute deviation of outbound backlog from the centre of its band, in units.
    d_centre: dict[float, Any] = field(default_factory=dict)
    #: Inbound backlog above the buffer line that sits under the storage ceiling.
    o_storage_buffer: dict[float, Any] = field(default_factory=dict)

    tiers: list[Tier] = field(default_factory=list)


def build(backend: Backend, p: Params, freeze_moves: bool = False) -> Model:
    """Build the model; `freeze_moves` gives the BEFORE (no-move) solve."""
    m = Model(backend=backend, params=p, freeze_moves=freeze_moves)
    for stage in STAGES:
        stage(m)
    m.tiers = _build_ladder(m, backend.xsum)
    return m


# --- People ---


def _add_people(m: Model) -> None:
    """Labour conservation through skill profiles.

    Every profile is tracked separately, so nobody is placed in a department they are not
    trained for. `m.freeze_moves` pins all movements to zero.
    """
    b, p, xsum = m.backend, m.params, m.backend.xsum
    freeze_moves = m.freeze_moves

    for q in p.quarters:
        for d in cfg.DEPTS:
            m.P[(q, d)] = b.int_var(f"P_q{q}_{d}")

    # (1) Everybody in a skill group goes somewhere in their skill set.
    for (q, origin, skills), people in p.pool.items():
        if people <= 0:
            continue
        for j in skills:
            m.y[(q, origin, skills, j)] = b.int_var(
                f"y_q{q}_{origin}_{'_'.join(skills)}_to_{j}"
            )
        b.add(
            xsum(m.y[(q, origin, skills, j)] for j in skills) == people,
            f"split_q{q}_{origin}_{'_'.join(skills)}",
        )

    # (2) Headcount by destination and skill profile.
    #
    # `dict.fromkeys`, not a set: hash order is randomised per process, and variable order
    # changes which of several equally optimal solutions the solver returns.
    profiles = list(dict.fromkeys((q, skills) for (q, _o, skills) in p.pool))
    for q, skills in profiles:
        for j in skills:
            terms = [
                m.y[(q, o, s, j)]
                for (qq, o, s, jj) in m.y
                if qq == q and s == skills and jj == j
            ]
            if terms:
                m.H[(q, j, skills)] = b.int_var(f"H_q{q}_{j}_{'_'.join(skills)}")
                b.add(
                    m.H[(q, j, skills)] == xsum(terms),
                    f"defH_q{q}_{j}_{'_'.join(skills)}",
                )

    # (3) Department headcount is the sum over profiles present in it.
    for q in p.quarters:
        for d in cfg.DEPTS:
            terms = [v for (qq, j, _s), v in m.H.items() if qq == q and j == d]
            b.add(m.P[(q, d)] == (xsum(terms) if terms else 0), f"P_q{q}_{d}")

    # (4) Reported movements are DERIVED from the profile splits, never free variables.
    for q in p.quarters:
        for i in cfg.DEPTS:
            for j in cfg.DEPTS:
                if i == j:
                    continue
                m.X[(q, i, j)] = b.int_var(f"X_q{q}_{i}_to_{j}")
                terms = [
                    m.y[(qq, o, s, jj)]
                    for (qq, o, s, jj) in m.y
                    if qq == q and o == i and jj == j
                ]
                b.add(
                    m.X[(q, i, j)] == (xsum(terms) if terms else 0),
                    f"linkX_q{q}_{i}_to_{j}",
                )
                if freeze_moves:
                    b.add(m.X[(q, i, j)] == 0, f"nomove_q{q}_{i}_to_{j}")


# --- Capacity ---


def _add_capacity(m: Model) -> None:
    """Units capacity per department per hour.

        capacity = TPH x (heads weighted by how much of the hour they cover)
                 + planned capacity for the uncovered fraction

    The edge term is zero here because every work-block boundary is a whole hour.
    """
    b, p, xsum = m.backend, m.params, m.backend.xsum

    for d in cfg.PRODUCING:
        for h in p.hours:
            m.Hcap[(d, h)] = b.num_var(f"Hcap_{d}_{int(h)}")
            heads = xsum(
                m.P[(q, d)] * p.cover[(h, q)] for q in p.quarters if (h, q) in p.cover
            )
            edge = p.edge.get(h, 0.0) * p.planned_cap.get((d, h), 0.0)
            b.add(
                m.Hcap[(d, h)] == p.tph.get((d, h), 0.0) * heads + edge,
                f"cap_{d}_{int(h)}",
            )

    # The machine ceiling.
    for h in p.hours:
        b.add(m.Hcap[(cfg.OB, h)] <= p.ob_machine[h], f"machine_OB_{int(h)}")


# --- Flow ---


def _add_flow(m: Model) -> None:
    """Backlog recursion per department.

        available   = backlog + arrivals
        throughput  = min(capacity, available)
        backlog[+1] = available - throughput
    """
    b, p = m.backend, m.params
    hours = p.hours

    for d in cfg.PRODUCING:
        for h in hours:
            m.U[(d, h)] = b.num_var(f"U_{d}_{int(h)}")
            # The only variable allowed below zero: arrivals are signed (holds,
            # cancellations), so `backlog + arrivals` can be negative.
            m.V_raw[(d, h)] = b.num_var(f"Vraw_{d}_{int(h)}", lb=-cfg.BIG_M)
            m.V[(d, h)] = b.num_var(f"V_{d}_{int(h)}")
            m.Y[(d, h)] = b.num_var(f"Y_{d}_{int(h)}")

        b.add(m.U[(d, hours[0])] == p.seed_backlog[d], f"seed_{d}")
        for h in hours:
            b.add(
                m.V_raw[(d, h)] == m.U[(d, h)] + p.arrivals.get((d, h), 0.0),
                f"avail_raw_{d}_{int(h)}",
            )
            b.add_max_zero(m.V[(d, h)], m.V_raw[(d, h)], f"avail_{d}_{int(h)}")
            b.add_min(
                m.Y[(d, h)], [m.Hcap[(d, h)], m.V[(d, h)]], f"wc_{d}_{int(h)}"
            )
        for prev, nxt in zip(hours[:-1], hours[1:]):
            b.add(
                m.U[(d, nxt)] == m.V[(d, prev)] - m.Y[(d, prev)],
                f"flow_{d}_{int(prev)}_{int(nxt)}",
            )

    # Total outbound units, the end-of-day numerator.
    for h in hours:
        m.U_total[h] = b.num_var(f"U_total_{int(h)}")
    b.add(m.U_total[hours[0]] == p.seed_ob_total, "seed_total")
    for prev, nxt in zip(hours[:-1], hours[1:]):
        b.add(
            m.U_total[nxt]
            == m.U_total[prev] + p.arrivals.get((cfg.OB, prev), 0.0) - m.Y[(cfg.OB, prev)],
            f"flow_total_{int(prev)}_{int(nxt)}",
        )


# --- Inbound storage ---


def _add_storage(m: Model) -> None:
    """Inbound backlog against the storage ceiling. One soft level, penalised per hour."""
    b, p = m.backend, m.params
    cap = p.ib_storage

    for h in p.hours:
        m.o_storage[h] = b.num_var(f"o_storage_{int(h)}")
        b.add(
            m.U[(cfg.IB, h)] <= cfg.THRESHOLDS["storage_limit"] * cap + m.o_storage[h],
            f"storage_{int(h)}",
        )

        # --- bottom guardrail: a buffer line under the ceiling --------------
        m.o_storage_buffer[h] = b.num_var(f"o_storage_buffer_{int(h)}")
        b.add(
            m.U[(cfg.IB, h)]
            <= cfg.THRESHOLDS["storage_buffer"] * cap + m.o_storage_buffer[h],
            f"storage_buffer_{int(h)}",
        )


# --- Outbound backlog band ---


def _add_ob_band(m: Model) -> None:
    """Outbound backlog inside its band, measured in hours of forward cover."""
    b, p, xsum = m.backend, m.params, m.backend.xsum

    for h in p.hours:
        window = p.bl_window[h]
        m.v_minus[h] = b.num_var(f"v_minus_{int(h)}")
        m.v_plus[h] = b.num_var(f"v_plus_{int(h)}")
        if len(window) < cfg.BL_WINDOW_HOURS:
            continue
        m.S_BL[h] = b.num_var(f"S_BL_{int(h)}")
        b.add(
            m.S_BL[h] == xsum(m.Hcap[(cfg.OB, k)] for k in window),
            f"SBL_{int(h)}",
        )
        rate = m.S_BL[h] * (1.0 / len(window))
        b.add(
            m.U[(cfg.OB, h)] <= p.bl_max_hours[h] * rate + m.v_minus[h],
            f"blmax_{int(h)}",
        )
        b.add(
            m.U[(cfg.OB, h)] >= p.bl_min_hours[h] * rate - m.v_plus[h],
            f"blmin_{int(h)}",
        )

        # --- bottom guardrail: distance outside a zone around the band centre ---
        plan_window = [p.planned_cap.get((cfg.OB, k), 0.0) for k in window]
        planned_rate = sum(plan_window) / len(window)
        centre_hours = 0.5 * (p.bl_min_hours[h] + p.bl_max_hours[h])
        dead_hours = cfg.THRESHOLDS["ob_bl_centre_deadband_hours"]
        zone_hi = (centre_hours + dead_hours) * planned_rate
        zone_lo = (centre_hours - dead_hours) * planned_rate
        m.d_centre[h] = b.num_var(f"d_centre_{int(h)}")
        b.add(m.d_centre[h] >= m.U[(cfg.OB, h)] - zone_hi, f"centre_hi_{int(h)}")
        b.add(m.d_centre[h] >= zone_lo - m.U[(cfg.OB, h)], f"centre_lo_{int(h)}")


# --- End of day ---


def _add_eod(m: Model) -> None:
    """A single day-level bound on the total outbound backlog."""
    b, p = m.backend, m.params
    den = p.eod_capacity_per_hour

    m.e_eod_max = b.num_var("e_eod_max")
    m.e_eod_min = b.num_var("e_eod_min")
    b.add(
        m.U_total[p.eod_hour] <= p.eod_soft_max * den + m.e_eod_max,
        "eod_soft_max",
    )
    b.add(
        m.U_total[p.eod_hour] >= p.eod_hard_min * den - m.e_eod_min,
        "eod_hard_min",
    )


# --- Carrier cut-off risk ---


def _add_cpt_risk(m: Model) -> None:
    """Cumulative demand against cumulative capacity, per carrier cut-off."""
    b, p, xsum = m.backend, m.params, m.backend.xsum
    tol = cfg.THRESHOLDS["cpt_risk"]

    interval = {}
    for c in p.cpts:
        interval[c] = b.num_var(f"H_interval_{c}")
        b.add(
            interval[c]
            == xsum(p.alpha[(c, h)] * m.Hcap[(cfg.OB, h)] for h in p.hours if p.alpha[(c, h)] > 0),
            f"H_interval_{c}",
        )
    for i, c in enumerate(p.cpts):
        m.H_cum[c] = b.num_var(f"H_cum_{c}")
        b.add(
            m.H_cum[c] == xsum(interval[k] for k in p.cpts[: i + 1]),
            f"H_cum_{c}",
        )
        m.g_cpt[c] = b.num_var(f"g_cpt_{c}")
        b.add(
            p.cpt_demand[c] <= tol * m.H_cum[c] + m.g_cpt[c],
            f"cpt_risk_{c}",
        )


# --- Returns backlog ---


def _add_returns_band(m: Model) -> None:
    """Returns backlog under a days-of-cover ceiling.

    The window is clipped to the horizon and scaled by the hours present, so clipping
    does not tighten the ceiling.
    """
    b, p, xsum = m.backend, m.params, m.backend.xsum
    days = cfg.THRESHOLDS["returns_bl_days"]

    for h in p.hours:
        window = p.rt_window[h]
        m.w_rt[h] = b.num_var(f"w_rt_{int(h)}")
        if not window:
            continue
        m.C_RT[h] = b.num_var(f"C_RT_{int(h)}")
        b.add(
            m.C_RT[h] == xsum(m.Hcap[(cfg.RT, k)] for k in window),
            f"C_RT_{int(h)}",
        )
        daily = m.C_RT[h] * (24.0 / len(window))
        b.add(m.U[(cfg.RT, h)] <= days * daily + m.w_rt[h], f"rt_bl_{int(h)}")


# --- The build order ---


STAGES = (
    _add_people,
    _add_capacity,
    _add_flow,
    _add_storage,
    _add_ob_band,
    _add_eod,
    _add_cpt_risk,
    _add_returns_band,
)


# --- The ladder ---


def _build_ladder(m: Model, xsum) -> list[Tier]:
    """The objective tiers, highest priority first.

    Within a family the current shift outranks everything later. A guardrail is a loose
    version of a family, settling which breached family gets attention first.
    """
    p = m.params
    now = [h for h in p.hours if p.tier_of_hour[h] == 0]
    later = [h for h in p.hours if p.tier_of_hour[h] == 1]
    cpt_now = [c for c in p.cpts if p.cpt_tier[c] == 0]
    cpt_later = [c for c in p.cpts if p.cpt_tier[c] == 1]

    def over(d: dict, keys) -> Any:
        return xsum(d[k] for k in keys if k in d)

    moves = xsum(
        cfg.THRESHOLDS["move_cost"] * v for v in m.X.values()
    )
    support = xsum(-m.P[(q, cfg.SP)] for q in p.quarters)

    return [
        Tier("storage_now", over(m.o_storage, now), "Inbound storage (this shift)"),
        Tier("storage_later", over(m.o_storage, later), "Inbound storage (later)"),
        Tier("cpt_now", over(m.g_cpt, cpt_now), "Carrier cut-off risk (this shift)"),
        Tier("cpt_later", over(m.g_cpt, cpt_later), "Carrier cut-off risk (later)"),
        Tier("ob_bl_max_now", over(m.v_minus, now), "Outbound backlog too high (this shift)"),
        Tier("ob_bl_max_later", over(m.v_minus, later), "Outbound backlog too high (later)"),
        Tier("eod_max", m.e_eod_max, "End-of-day backlog above target"),
        Tier("ob_bl_min_now", over(m.v_plus, now), "Outbound backlog too low (this shift)"),
        Tier("ob_bl_min_later", over(m.v_plus, later), "Outbound backlog too low (later)"),
        Tier("eod_min", m.e_eod_min, "End-of-day backlog below target"),
        Tier("returns_bl", over(m.w_rt, p.hours), "Returns backlog ceiling"),
        # --- bottom guardrails ---------------------------------------------
        Tier("storage_buffer", over(m.o_storage_buffer, p.hours),
             "Inbound storage: buffer under the ceiling", kind="guardrail"),
        Tier("ob_bl_centre", over(m.d_centre, p.hours),
             "Outbound backlog: toward band centre", kind="guardrail"),
        Tier("moves", moves, "Minimise movements", kind="preference"),
        Tier("support", support, "Park spare people in Support", kind="preference"),
    ]
