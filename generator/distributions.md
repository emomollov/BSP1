# Distributions for simulating the solver inputs

How to simulate each solver input, for one site. Every parameter is tagged:

| Tag | Meaning |
|---|---|
| **FITTED** | estimated from real data held in this repo, source and date given |
| **STRUCTURAL** | implied by an identity, not estimated - `capacity = TPH x heads` means that once capacity is fitted and TPH is chosen, headcount follows |
| **ASSUMED** | a plausible value with the reasoning stated |

Fitting sources: two extracts of retrospective output, held outside this folder. Both are
wide enough that the per-parameter counts quoted below are meaningful.

The shape comes from a single **anchor site**, chosen for >95% complete rows on the needed
columns, a full fortnight of dates, and a mid-range capacity rather than a network outlier.
It is not named here and the generator never needs it: fitting was done once and
the parameters below are the whole of its residue.

Implementation: `mock_inputs/fitted.py` holds the parameters, `mock_inputs/generate.py`
the chain.

---

## How a run actually works

One command drives the whole chain.

```
seeds.py              which day(s) to run
   │                  curated 13  |  --verified N  |  --seeds  |  --cases
   ▼
fitted.py             the parameters: hour profiles, phi, residual sd, daily means
   │                  columns.py supplies the headcount column names it keeps
   ▼
generate.py           draws ONE window -> the frames to_demo reads
   │
   ▼
to_demo.py            generated frames -> the demo's 8 input frames
   │                  6 departments -> 4, wide headcount -> long,
   │                  hourly -> work blocks, 24h slice
   ▼
model/capacity_allocator
   │                  
   │                  params.build -> model.build -> solve_lexicographic
   │                  
   ▼
outputs               classify the binding limit, and ALSO run production's own
   │                  classify_capacity_limit on the same hours for comparison
   ▼
CSVs in <--out>/data
```

A fitted day is published as just another value of the `scenario` column
(`scenario=fitted_<seed>`), which is why it plots through the same chart functions as the
engineered scenarios with no special-casing anywhere downstream.

### What each mode writes

| Command | Writes |
|---|---|
| `--verified N --out D` | `fitted_days.csv` (one row per day). Unbiased |
| `--cases --out D` | the above, plus `fitted_cases.csv`, plus appends `scenario=fitted_<seed>` rows to `moves.csv` / `review.csv` / `detail.csv` |
| `--seeds 90017,90042 --out D` | as `--cases`, for named seeds. Re-running a seed replaces its rows, so it is idempotent |
| `--scan N --select --out D` | `fitted_cooccurrence.csv`, `fitted_families_per_day.csv` - which constraint families co-occur. A development step, not a presentation step |
| no `--out` | solves and prints, writes nothing |

---

## Where this document sits

```
real output                        two extracts, held outside this folder
   │   fitting - done once, recorded in fitted.py with a tag per parameter
   ▼
fitted parameters                  hour profiles, phi, residual sd, daily means
   │   generate.py - draws a window for one site
   ▼
generated frames                 the shapes to_demo reads
   │   to_demo.py
   ▼
the solver                         moves / review / detail .csv
```

This document is the middle arrow: why each parameter has the value it has, and why the
chain is assembled in the order it is.

---

## The recipe every series below follows

Sections 1-12 all apply the same decomposition, so it is written out once here rather than
restated twelve times. A series is split into three parts because each needs different
treatment:

**1. The hour-of-day profile - the deterministic skeleton.** Group the real series by local
hour and take the mean of each. Capacity is normalised into 24 shares summing to 1; backlog
is kept in levels, because it is anchored rather than scaled.

**2. The residual - what is left once the shape is removed.** Divide the observed series by
its own hour profile. What remains is a unitless multiplier centred near 1: "this hour ran
hot or cold relative to its usual self."

**3. Persistence and spread of that residual.** Take logs, then

```
phi      = lag-1 autocorrelation of the log residual   = sum(r[t] * r[t+1]) / sum(r^2)
resid_sd = sd of the log residual
family   = lognormal if log-skew ~ 0 while raw skew is not; otherwise normal
```

Logs earn their place twice: a Gaussian AR(1) walked on a level eventually goes negative,
and negative capacity is not a thing - and log-skew against raw skew is what decides the
family rather than an eyeball.

On top of the three parts, two global terms:

- **a daily level and CV** per series (capacity: 254,360/day at CV 0.148);
- **one day multiplier per date**, shared by every series on that date, because a site has
  good days and bad days as a whole rather than per series.

So the general form each section instantiates is:

```
series[t] = daily_mean * profile[hour(t)] * day_level[date(t)] * AR1(phi, resid_sd)
```

with flows using an iid draw in place of the `AR1` term - which is finding §0.1.

---

## 0. Four findings that change how you simulate this

Each of these was wrong in my first draft. 

### 0.1 Everything is autocorrelated. Independent hourly draws are wrong.

Lag-1 autocorrelation on the de-seasonalised multiplicative residual:

| Series | phi | residual sd | family |
|---|---|---|---|
| `total_hrly_capacity_sum_new` | **0.904** | 0.191 | lognormal (log-skew 0.48) |
| `total_pickable_units` | **0.965** | 0.229 | left-skewed; a state variable, do not draw it directly |
| `total_units` | **0.990** | 0.296 | - |
| `eligible_units_gambler` | **0.977** | 1.224 | lognormal (log-skew -0.16, raw skew 1.06) |
| `total_tcap_units` | **0.965** | 0.449 | lognormal (log-skew 0.23) |
| derived arrivals | **0.139** | 0.567 | effectively white noise |
| `max_shippable_items` | **0.047** | 0.571 | effectively white noise |

So: **levels are AR(1), flows are iid.** Draw capacity and backlog as AR(1) around their
hour-of-day profile; draw arrivals and the ship plan iid around theirs.

```python
# stationary sd = resid_sd, so scale the innovation
innovation_sd = resid_sd * sqrt(1 - phi**2)
```

Build it in **log space** and exponentiate. A Gaussian AR(1) applied to a level eventually
goes negative and produces negative capacity.

### 0.2 Centre lognormal multipliers on `-sigma^2/2`, or you inflate the level

`exp(Normal(0, sigma))` has mean `exp(sigma^2/2)`, not 1. Applying it to a fitted level
inflates that level by 1.8% at sigma = 0.19 and by **112%** at the gambler series' sigma =
1.22. Use `rng.lognormal(mean=-0.5*sigma**2, sigma=sigma)`.

### 0.3 Anchor the backlog; derive arrivals. Not the other way round.

The obvious construction - draw arrivals and capacity, run the recursion, let the backlog
fall out - **does not work**. The backlog's level then depends on the accumulated gap
between two independently fitted series (capacity 254,360/day, arrivals 229,329/day)
instead of on anything fitted, and nothing holds it near the band its thresholds are
defined against.

The real site holds 53-69k all day because the operation *holds backlog deliberately*: the
band floor alone is 3.5 hours of cover, about 37,000 units. **That level is an input to the
solver, not something the plan implies.**

So draw the backlog around its fitted hour-of-day profile, draw what was shipped around
its own, and derive arrivals from the flow identity:

```
A[t] = B[t+1] - B[t] + C[t]
```
---

## 1. Shift structure 

**FITTED** from the anchor site's hour-of-day activity, where two independent signals agree:

- `optimal_ib_capacity` is exactly 0 at hour 5, full at 6-8 and 10-12, and partial at 9
  (50% of peak), 13 (23%) and 20 (44%);
- `actual_units` collapses to 3% of peak at hour 5 and 21% at hour 4, and dips to 29% at 13.

Reading them together gives a daily pattern, tiled across dates:

| Hours | `interval_type` | Shift |
|---|---|---|
| 00:00-04:00 | `shift` | NIGHT |
| 04:00-05:00 | `break_q4` | NIGHT |
| 05:00-06:00 | `closed` | - |
| 06:00-09:00 | `shift` | MORNING |
| 09:00-10:00 | `break_q1` | MORNING |
| 10:00-13:00 | `shift` | MORNING |
| 13:00-14:00 | `break_q2` | MORNING |
| 14:00-20:00 | `shift` | AFTERNOON |
| 20:00-21:00 | `break_q3` | AFTERNOON |
| 21:00-22:00 | `shift` | AFTERNOON |
| 22:00-24:00 | `shift` | NIGHT |

Inbound runs 06:00-20:00 - **FITTED**: 12.9 hours per day at the anchor site.

---

## 2. Outbound capacity

**FITTED**:

```
daily total   254,360 units    sd 37,544    CV 0.148
hourly mean   ~10,600 units
residual      sd 0.191, AR(1) phi 0.904
```

Share of the day's capacity by local hour - note the **U shape**, heaviest either side of
midnight and lightest 07:00-12:00, which is the opposite of the intuitive guess:

```
h    0      1      2      3      4      5      6      7      8      9     10     11
   .0508  .0523  .0395  .0375  .0349  .0381  .0381  .0343  .0339  .0342  .0362  .0343
h   12     13     14     15     16     17     18     19     20     21     22     23
   .0376  .0439  .0494  .0449  .0407  .0406  .0383  .0394  .0449  .0519  .0523  .0521
```

Simulate:

```
day_level  ~ LogNormal(unit mean, sd = 0.148)        # one draw per DATE, shared by all series
capacity[t] = 254_360 * share[hour(t)] * day_level * AR1(phi=0.904, sd=0.191)
```

**Zero it on `closed` hours only, then renormalise the remaining shares per day.** Without
the renormalisation the day loses the ~11% of share sitting on non-working hours and the
generated daily total lands 11% low.

---

## 3. Shipped and arrivals 

`actual_shipped_units` is the `C[t]` of the flow identity. **FITTED** hour-of-day means:

```
h    0      1      2      3      4      5      6      7      8      9     10     11
  10681  14137  10190   9267   3069    393   7743  10422   9902   6747  10534  10787
h   12     13     14     15     16     17     18     19     20     21     22     23
   8303   4110  13987  13704  12629   9192  11063  12116   7680  10409  14312  13029
```

Daily total **234,407** - below the 254,360 of capacity, so availability binds more often
than capacity does.

**Cap shipped by what the pool holds, NOT by reported capacity.** Fitted
`actual_shipped_units / avg_3h_capacity` has median 0.980 but **p90 1.402**: the site
out-produces its own reported capacity in a large minority of hours. 

Arrivals then follow from `A[t] = B[t+1] - B[t] + C[t]`. **FITTED** check values:
mean 9,555/hr, daily 229,329, CV 0.604, and **negative in 5.3% of hours**.

The negatives are the point, not noise: holds going on, cancellations and the pick-to-ship
timing gap all remove units from the pickable pool without shipping them.

---

## 4. Pickable backlog 

**FITTED** hour-of-day mean of `total_pickable_units`, in units:

```
h    0      1      2      3      4      5      6      7      8      9     10     11
  66058  67049  64867  63970  62137  63588  61131  57531  54329  55586  54502  53406
h   12     13     14     15     16     17     18     19     20     21     22     23
  54165  59154  57394  58118  60606  61761  64013  64611  66907  69123  64883  64453
```

AR(1) phi 0.965, residual sd 0.229. Seed at 23:00: mean 64,453, sd 7,495.

```
backlog[t] = profile[hour(t)] * day_level * AR1(phi=0.965, sd=0.229)
```

---

## 5. Backlog composition  

**FITTED** shares of `total_units` at the anchor site:

| Component | mean | sd | p10 | p90 |
|---|---|---|---|---|
| pickable / total | 0.544 | 0.092 | 0.402 | 0.658 |
| unavailable / total | 0.119 | 0.120 | 0.008 | 0.309 |
| transportation_restricted / total | 0.157 | 0.051 | 0.094 | 0.214 |

They sum to 0.82; the remainder is work in progress.

**Build the total UP from its parts**, so `units_total >= pickable_units` holds by
construction. 

'Unavailable' units are the most dispersed series in the data (CV 1.165) and among the most
persistent (phi 0.977). Lognormal is the right family - log-skew -0.16 against a raw skew
of 1.06.

---

## 6. Backlog band 

**FITTED**. The anchor site runs `min_bl` 3.5-4.0 and `max_bl` 6.0 flat. Across the extract, the
common pairs by row count:

```
(4.5, 7.0) 14,574    (3.5, 6.0) 5,674    (2.0, 4.0) 4,487
(1.0, 2.5)  3,360    (1.5, 3.0) 3,360    (2.0, 3.5) 3,143
```

These are **HOURS** of cover. The end-of-day thresholds in the same model are in **DAYS**.

---

## 7. End of day 

`avg_ob_7_day_capacity` is **STRUCTURAL**: a 7-day trailing mean of the daily capacity.
The solver treats it as fixed - *"NOT affected by intra-day capacity moves, so it
is a fixed input"* - which is what keeps the constraint linear.

`eod_bl` **FITTED**: all sites mean 0.509 days, sd 0.371, p5 0.198, p95 1.278. At the anchor site mean
0.578.

---

## 8. Throughput per hour 

**ASSUMED**:

| Process | units/person/hour |
|---|---|
| `OUTBOUND` | 118 |
| `INBOUND` | 82 |
| `VENDOR_RETURNS` | 48 |
| `DA` | 64 |

Sanity check: 254,360 units/day at 118/person/hour is ~2,155 person-hours, or ~90 people on
shift around the clock. Right order of magnitude for a site this size.

Change TPH and the whole headcount side moves **inversely**. Within-block ramp +/- 8%.

---

## 9. Headcount and cross-training 

Headcount is **STRUCTURAL**: `heads = round(capacity / TPH)`. 

Other departments as a share of outbound - **ASSUMED**: IB 0.45, DA 0.14, VRET 0.10.

### The cross-training matrix, and the precondition nothing checks

**ASSUMED** proportions:

| Origin | native only | movable buckets |
|---|---|---|
| Outbound (OB) | 0.62 | IB 0.20, IB+RE 0.08, RE 0.10 |
| Inbound (IB) | 0.59 | OB 0.26, OB+RE 0.10, RE 0.05 |
| Returns (RE) | 0.56 | OB 0.34, OB+IB 0.10 |

**Each origin's buckets MUST sum to 1.0, including the native-only bucket.**

Absenteeism - **ASSUMED**, drawn once per department per **day** because the real figure is
`ABSENCE_HOURS / SCHEDULED_HOURS` over a shift: OB 6.0%, IB 5.5%, RE 4.5%, sd 1.5pp.

---

## 10. Inbound 

**FITTED**: mean 13,821, sd 9,224, peak 37,049

Everything else inbound is **STRUCTURAL or ASSUMED**:

| Quantity | Value | Tag |
|---|---|---|
| arrivals / capacity | 0.96, so the backlog holds rather than trends | STRUCTURAL |
| backlog seed | 120,000 units | ASSUMED |
| storage ceiling | 1.45 x the peak backlog the recursion reaches | ASSUMED |
| machine ceiling | 1.30 x planned capacity | ASSUMED |

Spread some inbound arrivals across closed hours - trailers arrive when the stow line is
down, which is the whole reason a storage ceiling binds.

---

## 11. Carrier cut-offs 

**ASSUMED** times: 02:00, 07:00, 11:00, 14:00, 17:00, 19:00, 21:00, 23:00

**STRUCTURAL** demand: `f_cumulative = 0.82 x cumulative outbound capacity to that cut-off`, with
`g_cumulative = 1.06 x f_cumulative` since the forecast includes orders not yet placed.

---

## 12. Vendor returns 

**ASSUMED** throughout

Capacity as a share of outbound 0.055. Backlog seeds 31,000. Arrivals at 0.95 of its own capacity so no trends.

Vendor returns must emit **both demand shapes**, because the solver reads them differently:

- **future units** - one row per deadline (`exsd + 24h`), with a running cumulative;
- **overdue units** - a SINGLE row per site, `is_overdue = 1`, with a **NULL** deadline and
  a **NULL** cumulative. The solver paces this scalar with a linear ramp over 24 hours
  because it has no deadline left to model.

---

## Calibration: generated against fitted

30 seeds, 7-day windows.

### Levels

| Quantity | Ratio | seed sd of the ratio |
|---|---|---|
| OB capacity, daily | **0.99** | 0.09 |
| OB shipped, daily | **0.99** | 0.08 |
| OB arrivals, daily (derived) | **1.01** | 0.08 |
| pickable / total | **1.03** | 0.07 |
| shipped / capacity, p90 | **0.97** | 0.09 |
| pickable backlog, mean | **1.00** | 0.13 |
| negative arrival share | 1.21 | 0.33 |

The backlog row is referenced against the mean of the fitted hour-of-day profile (61,223), so the denominator is a
value held in this repo and the ratio can be re-derived rather than taken on trust. It
lands at 1.00 on that basis.

A single seed is noise: daily capacity has a fitted CV of 0.148, so one 7-day window
carries a standard error of ~5.6% on its mean. One seed came out 10% high, which is 1.9 se
and nothing.

The negative-arrival share at 1.21x is the weakest match, 6.4% against a fitted 5.3%. The
**seed-to-seed sd of that ratio is 0.33** against a miss of 0.21, so the gap is inside one
sd of its own sampling noise - tuning it closed would be fitting to a statistic whose error
bar is larger than the error. The mechanism is reproduced, which is what the model needs.

---

## Scaling to more sites

Everything above is one site. What has to change:

1. **Site-level scale.** Pick a daily capacity per site; the anchor's 254,360 is mid-range. The
   network spans at least an order of magnitude.
2. **Band pairs.** Draw from the six fitted `(min_bl, max_bl)` pairs in §6 rather than
   using the anchor's.
3. **Shift pattern.** The §1 pattern is the anchor's. Others differ, and the one thing to avoid
   is giving every site the same roster - `outside_shift` is 11.3% of real site-hours, and a
   single pattern would make that figure a property of the generator rather than of the
   network.
   