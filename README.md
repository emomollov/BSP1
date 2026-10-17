# Capacity Allocator

A standalone, scaled-down reduction of a forward-looking capacity solver.

Generates synthetic data, solves a reduced mixed-integer program, and presents the results
live from a notebook. No network, no credentials, no real data - every number is synthetic.

## What is here

```
model/       the model - engineered scenarios, a reduced MIP
generator/   draws unengineered days from fitted distributions
output/      presentation.ipynb, notebooks/, and everything they produce: data/, inputs/
docs/        architecture overview
```

## The demo

```bash
cd model && pip install -r requirements.txt && python run_demo.py
cd ../generator && python run_fitted.py --cases --out ../output
cd ../generator && python run_fitted.py --verified 25 --out ../output
cd output && jupyter lab presentation.ipynb                          # present it
```

**Generation and presentation are separate.** `run_demo.py` writes three CSVs;
`presentation.ipynb` reads them and draws one chart per cell, live, in the order of the results.

### What it writes

Three CSVs in `output/data/`, one file per question:

| File | One row per | The question it answers |
|---|---|---|
| `moves.csv` | non-zero movement | **What did the model decide?** The recommendation itself: how many people, from where, to where, at which block boundary. A scenario that recommends nothing contributes no rows. |
| `review.csv` | (scenario, objective tier) | **Did it work?** The priority ladder with each tier's breach before and after, plus the scenario's verdict - `moves_total`, `limited_by`, `resolved` - repeated on every row, so nothing needs a join. |
| `detail.csv` | (scenario, hour) | **What happened, hour by hour?** The full trajectory as optimal/actual pairs, the carrier cut-offs joined onto the hours they fall in, and the day-level targets. Wide on purpose: the notebook filters it. |

### Unengineered days in the same frames

Every engineered scenario except `baseline` injects a defect. `generator/run_fitted.py` draws
days from distributions fitted to real output instead, where several constraints go live at
once - and selected days land in those same three CSVs as `scenario = fitted_<seed>`, so
the notebook charts them with identical calls.

```bash
cd generator
python run_fitted.py --cases --out ../output   # the pinned days
python run_fitted.py --seeds 90065,90073 --out ../output       # any days
python run_fitted.py --scan 200 --select --out ../output
```

`--cases` reads `model/fitted_cases.toml`, the manifest of days discussed, with a seed,
a pattern name and a description per case.

Appending is per-scenario idempotent: re-running a seed replaces its rows, so the published
CSVs stay byte-identical. A distribution run writes the fitted-day summary only - it would
otherwise push hundreds of per-hour rows for days nobody plots into `detail.csv`.

## Engineered scenarios

| Scenario | Breach | Moves | Outcome |
|---|---|---|---|
| `baseline` | none | 25 | no defect to fix - every movement buys risk headroom |
| `ib_surge` | storage, 14,415 | 53 | storage breach cleared |
| `ob_backlog_build` | backlog band, 22,662 | 32 | backlog band breach cleared |
| `cpt_at_risk` | cut-off risk, 8,929 | 35 | deadline risk 109% -> 100% |
| `eod_target_miss` | EOD target, 6,298 | 53 | day-level target met |
| `ib_then_eod` | storage **now** 3,724, storage later 6,587, EOD 6,926 | 83 | both cleared - 29 people out of OB in the morning, 19 + 14 back in the afternoon |
| `machine_capped` | backlog band, 22,662 | 4 | **44% recovered** - `limited_by = machine`, and it names the 6 hours |
| `pool_exhausted` | backlog band, 22,662 | 12 | **partly fixed** - `limited_by = cross_trained_people` |

### A day nobody engineered

After the engineered scenarios, one unengineered run. `generator/` generates the solver's
inputs from distributions fitted to real retrospective output, and `run_fitted.py`
puts N such days through the same model.

| limit | demo, 25 days | real, 135 sites | delta |
|---|---|---|---|
| `no_more_capacity_needed` | 39.7% | 56.5% | -16.8 |
| `cross_trained_people` | 39.3% | 31.0% | +8.3 |
| `outside_shift` | 20.8% | 11.3% | +9.5 |
| `no_work_available` | 0.2% | 1.2% | -1.0 |

## What it models

A fulfilment site where people are trained to work in more than one department. Demand,
arrivals and deadlines move hour to hour. At the start of each work block somebody decides
how many people go where.

```
4 departments   Outbound, Inbound, Returns, Support
24-hour horizon 3 shifts x 2 work blocks = 6 decision points; breaks are closures
4 constraints   inbound storage ceiling, outbound backlog band,
                end-of-day backlog target, carrier cut-off risk
15 tiers        solved in strict priority order; 11 operational, 2 bottom
                guardrails, 2 preferences
922 variables / 769 constraints, as Gurobi reports them after presolve
```

## Layout

```
model/run_demo.py                  CLI - generate, solve, publish
model/capacity_allocator/
  config.py                  thresholds, geometry, VOCABULARY
  scenarios.py               named parameter overrides
  data.py                    synthetic data generator + CSVs -> model parameters (the whole input boundary)
  model.py                   variables, constraints, objective tiers, Gurobi adapter, lexicographic ladder
  charts.py                  (dataframe, Axes) -> a drawn chart
  run.py                     orchestration, solved model -> dataframes, and the 3 published CSVs
output/presentation.ipynb          charts from the published CSVs
output/notebooks/                  model build and solve walkthroughs
```
