# POC architecture

The demo is one CLI script with no network calls and no credentials. It generates
synthetic inputs, solves a MIP, and writes three CSVs that the presentation notebook charts.

---

## 1. Overview

```
               SYNTHETIC DATA                        THE MODEL                       ARTIFACTS
 ┌──────────────────┐  ┌───────────────────┐  ┌───────────────────────────────┐  ┌────────────────────┐
 │ scenarios.py     │  │ data.py           │  │ data.py                       │  │ run.py             │
 │                  │  │                   │  │   CSVs -> Params              │  │   -> dataframes on │
 │ baseline         │─>│ shifts.csv        │─>│                               │─>│      run.Result    │
 │ ib_surge         │  │ headcount.csv     │  │                               │  └─────────┬──────────┘
 │ ob_backlog_build │  │ tph.csv           │  │ model.py                      │            │
 │ cpt_at_risk      │  │ ob_plan.csv       │  │   vars, constraints,          │            v
 │ eod_target_miss  │  │ ib_plan.csv       │  │   15 objective tiers          │  ┌────────────────────┐
 │ ib_then_eod      │  │ returns_plan.csv  │  │                               │  │ run.write_data     │
 │ machine_capped   │  │ cpts.csv          │  │   Gurobi behind Backend       │  │   moves.csv        │
 │ pool_exhausted   │  │ eod_guidance.csv  │  │   lexicographic ladder        │  │   review.csv       │
 └────────┬─────────┘  └───────────────────┘  │   (all in model.py)           │  │   detail.csv       │
          │                                   │                               │  │                    │
          │ config.py                         │   TWO SOLVES PER SCENARIO:    │  └─────────┬──────────┘
          │  thresholds, VOCABULARY, DEPTS    │     X == 0  -> BEFORE         │            │
          └──────────────────────────────────>│     free    -> AFTER          │            v
                                              └───────────────────────────────┘  ┌────────────────────┐
                                                              ^                  │ charts.py          │
                                                     run_demo.py (CLI)           │ (frame, Axes)      │
                                                                                 └──────────┬─────────┘
                                                                                            │
                                                                output/presentation.ipynb <─┘
                                                                    live, one per cell
```

**The three CSVs are the seam between generating and presenting.** `run.write_data` reshapes
the `Result` frames at emit time - `Result` itself keeps the shape the model expects - and
everything presentable is then chosen at presentation time rather than at render time. `charts.py` takes a published dataframe and an Axes the caller owns.

### 1.1 The fitted-day pipeline

`generator/` is a second entry point into the same model. It draws a day from distributions
fitted to real operational output, solves it with `model.py` unchanged, and appends
the result to the SAME three CSVs under `scenario = "fitted_<seed>"` via `run.fitted_result` and
`run.append_data`. 

That is why `write_data` replaces only the engineered scenarios and leaves everything else alone -
the two commands write the same files and must not depend on being run in a particular order. The
pinned days are in `model/fitted_cases.toml`

---

## 2. Modules

| Module | Responsibility | 
|---|---|
| `config.py` | Thresholds, department list, `VOCABULARY`, instance geometry, solver tolerances | 
| `scenarios.py` | Named parameter overrides; one dataclass per scenario | 
| `data.py` | Generate the 8 input CSVs for a given scenario, and read them into a `Params` dataclass | 
| `model.py` | Backend adapter, the lexicographic ladder, variables, constraints, objective tiers | 
| `charts.py` | `(dataframe, Axes)` → a drawn chart | 
| `run.py` | orchestration; `Params` + solution → dataframes; the only module that knows the artifact layout, and the one that reshapes for emit | 
| `output/presentation.ipynb` | read the 3 CSVs, one chart per cell, live | 

`data.build` is the whole input boundary - every `inputs[...]` read happens inside it - and
`model.build` is the whole model. Keeping that seam is what lets the production-to-demo mapping be
checked rather than asserted.

`capacity_allocator/__init__.py` keeps the old module names (`synth`, `params`, `solver`, `outputs`)
as aliases of `data`, `data`, `model`, `run`, so the notebooks and scripts written against the
earlier layout still import.

---

## 3. Two solves per scenario

Every chart is a before/after pair, and every published quantity follows the optimal/actual naming
convention

```
{domain}_{quantity}_optimal[_{unit}]   what the solver would have had
{domain}_{quantity}_actual[_{unit}]    what the plan already did
```

**BEFORE** pins every movement to zero (`model.build(..., freeze_moves=True)`) and solves. With no
movements allowed, headcount collapses to the planned assignment and the recursions evaluate to the
planned trajectory.

**AFTER** removes the pin and solves the full ladder.

This is not cosmetic. If BEFORE does not reproduce the planned backlog **exactly**, every delta in
every chart is contaminated by reconstruction drift rather than being attributable to the
recommendation. 
---

## 4. The lexicographic ladder - the one real implementation risk

### 4.1 Hand-rolling `setObjectiveN`

Gurobi solves a multi-objective model lexicographically for free (`:2791-2922`)

```python
m.setObjectiveN(obj_expr, index=idx, priority=n_objectives - idx, name=obj_name)
```

`model.py` implements solve-freeze-continue instead, because it returns something the native path
does not: a per-tier record of what the model gave up and where.

```python
# config.py
FREEZE_EPS_ABS = 1e-4
FREEZE_EPS_REL = 1e-7
FREEZE_EPS_REL_GUARDRAIL = 1e-2     # guardrail tiers only - see below

def solve_lexicographic(backend, tiers):
    """tiers: list[Tier(name, expr, label, kind)], highest priority first."""
    result = LadderResult()
    _check_licence_size(backend)
    for tier in tiers:
        backend.set_objective(tier.expr)
        status = backend.solve()
        if status is not Status.OPTIMAL:
            raise TierInfeasible(tier.name, status)   # fail loudly, never skip a tier
        v = backend.value(tier.expr)
        result.attained[tier.name] = v
        # Freeze this tier so no later tier can trade against it.
        rel = FREEZE_EPS_REL_GUARDRAIL if tier.kind == "guardrail" else FREEZE_EPS_REL
        backend.add(tier.expr <= v + FREEZE_EPS_ABS + rel * abs(v), f"freeze_{tier.name}")
    return result
```

A `Tier` carries a `kind` - `slack`, `guardrail` or `preference` - which decides both the freeze
tolerance and how the value is reported. 

`attained` is the objective ladder, published as `review.csv` - one row per (scenario, tier) - and
charted (chart 7). That chart is the clearest available picture of how the model prioritises, and it
is a thing the production Gurobi path does **not** expose.

### 4.2 Runtime

15 tiers x 2 solves (before/after) x 8 scenarios = 240 MIP solves.

**Measured:** 922 variables and 769 constraints after presolve; 0.2-0.4 s per scenario under
Gurobi, about 3 s for the whole `run_demo.py` run including Python start-up.

---

## 5. Verification

### 5.1 Invariants

Properties the model holds, each checkable against the published output.

| Property | What holds 
|---|---|
| **Replay identity** | BEFORE reproduces the planned backlog series to 1e-6 units | 
| **People conservation** | Per quarter, moves out == moves in; `P` never exceeds the `n_skill` pool |
| **Monotonicity** | Adding a head to Outbound never decreases Outbound capacity | 
| **Guardrails never degrade** | no tier at priority 1-11 attains a worse value with the two bottom guardrails in the ladder than without them | 
| **`machine_capped` honesty** | Breaches in BOTH before and after; `limited_by == "machine"`; the wall is a strict subset of the workable hours | 
| **Highest breached tier never worsens** | The highest-priority *breached* tier is at least as good in AFTER as in BEFORE | 

---

## 6. Build order

1. `config.py` → `data.py`. Get synthetic CSVs into `Params`, with the
   **replay identity holding before any solving**. If the no-move trajectory does not
   reproduce the plan, nothing downstream is trustworthy.
2. The lexicographic ladder in `model.py`, exercised on a toy two-tier model
   first to shake out the freeze tolerance.
3. The model itself, baseline scenario only.
4. Add constraint families one at a time, each with its scenario.
5. The dataframe layer in `run.py`, then `charts.py`.

---
