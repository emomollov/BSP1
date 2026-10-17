"""Charts for the presentation notebook: `(dataframe, Axes)` in, drawn chart out."""

from __future__ import annotations

from pathlib import Path

# No `matplotlib.use(...)` here: the backend is the caller's choice.
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from . import config as cfg
from . import run

# --- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

ACTUAL = "#112E81"  # categorical slot 1 - the plan
OPTIMAL = "#C654C3"  # categorical slot 2 - the recommendation
RESOLVED = "#006300"  # status - a breach that was cleared
BAND = "#f0efec"  # neutral wash

#: `LIMIT_LINE` is where a limit is, `BREACH` marks a value crossing one. Same colour, two names.
LIMIT_LINE = INK
BREACH = INK

#: Per-shift fills keyed to `config.SHIFTS`, light to dark. Blocks have no outline, so only
#: the darkest steps hold their shape (contrast vs surface 2.1, 3.8, 7.2).
SHIFT_COLOURS = {"A": "#ED96D7", "B": "#C654C3", "C": "#8E1EA2"}


LINE_W = 2.0
MARKER = 5.0

#: Width of the `actual` underlay line.
UNDERLAY_W = 5.0

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "figure.autolayout": False,
    }
)


def _frame(ax, ygrid: bool = True) -> None:
    """Recessive chrome: hairline y-grid under the data, no top or right spine."""
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)


def _frame_x(ax) -> None:
    """`_frame` for horizontal bars: x grid, no category-axis spine."""
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)


def _hour_ticks(ax, hours: list[float], every: int = 3) -> None:
    show = [h for i, h in enumerate(hours) if i % every == 0]
    ax.set_xticks(show)
    ax.set_xticklabels([cfg.hour_label(h) for h in show])


def _pair_series(ax, hrs, actual, optimal) -> None:
    """Before/after line pair; coincident series read as a navy band with a pink core."""
    ax.plot(hrs, actual, color=ACTUAL, lw=UNDERLAY_W, marker="o", ms=MARKER + 2.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0, label="Planned", zorder=2,
            solid_capstyle="round")
    ax.plot(hrs, optimal, color=OPTIMAL, lw=LINE_W, marker="o", ms=MARKER,
            markeredgecolor=SURFACE, markeredgewidth=1.0, label="Recommended", zorder=3)


def _pair_barh(ax, y, actual, optimal, hgt, labels, edged: bool = True) -> None:
    """Before/after horizontal bar pair, `actual` above `optimal`.

    `edged` adds a surface-coloured edge; omit it where bars are so short the edge dominates.
    """
    style = {"edgecolor": SURFACE, "linewidth": 2} if edged else {}
    ax.barh([v + hgt / 2 for v in y], actual, hgt, color=ACTUAL, label=labels[0], **style)
    ax.barh([v - hgt / 2 for v in y], optimal, hgt, color=OPTIMAL, label=labels[1], **style)


def _contrast(fg: str, bg: str) -> float:
    """WCAG contrast ratio between two colours, 1.0 (identical) to 21.0 (black on white)."""
    def luminance(colour: str) -> float:
        channels = (
            c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            for c in to_rgb(colour)
        )
        r, g, b = channels
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    lo, hi = sorted((luminance(fg), luminance(bg)))
    return (hi + 0.05) / (lo + 0.05)


def _label_on(bg: str) -> str:
    """`INK` or `SURFACE`, whichever contrasts more with `bg`."""
    return max((INK, SURFACE), key=lambda fg: _contrast(fg, bg))


def _title(label: str | None, suffix: str, override: str | None = None) -> str:
    """Scenario headline plus chart subject; `override` replaces both."""
    if override is not None:
        return override
    return f"{label} - {suffix}" if label else suffix[:1].upper() + suffix[1:]


def _one(frame, what: str):
    """Guard: a per-scenario chart must get one scenario, else it plots several days as one."""
    if "scenario" in frame.columns and frame["scenario"].nunique() > 1:
        raise ValueError(
            f"{what} draws one scenario at a time; was given "
            f"{sorted(frame['scenario'].unique())}. Filter it first, e.g. "
            f"detail.query(\"scenario == 'ib_surge'\")."
        )
    return frame


# --- figure sizes ----------------------------------------------------------

SIZE_TIMELINE = (9.0, 2.0)
SIZE_SERIES = (9.0, 3.4)  # ob_backlog, ib_storage
SIZE_CPT = (9.0, 3.2)
SIZE_EOD = (5.4, 3.2)
SIZE_FAMILIES = SIZE_SERIES
SIZE_LADDER_W = 8.2

#: Default panels for `headcount`. Support stays in: it is where spare people park.
HEADCOUNT_DEPTS = list(cfg.DEPTS)


def families_per_day(frame, ax, title: str | None = None) -> None:
    """Legacy compatibility chart for the fitted families-per-day distribution."""
    data = frame.copy()
    if data.empty:
        ax.text(0.5, 0.5, "no fitted families-per-day data", ha="center", va="center")
        ax.set_axis_off()
        return
    if "families_breached" not in data.columns or "days" not in data.columns:
        raise ValueError(
            "families_per_day expects a dataframe with 'families_breached' and 'days' columns"
        )
    x = data["families_breached"].astype(int)
    y = data["days"].astype(float)
    bars = ax.bar(x, y, color=OPTIMAL, width=0.8)
    ax.set_xlabel("families breached")
    ax.set_ylabel("days")
    ax.set_xticks(x)
    ax.set_ylim(0, max(1.0, float(y.max()) * 1.2))
    if title is not None:
        ax.set_title(title)
    _frame(ax)
    return bars


def size_headcount(depts: list[str] | None = None) -> tuple[float, float]:
    return 2.9 * len(depts or HEADCOUNT_DEPTS), 3.1


def size_ladder(review) -> tuple[float, float]:
    """Figure size that grows with the tiers and bands in `review`."""
    tiers = len(review)
    bands = len(_ladder_bands(review))
    return SIZE_LADDER_W, 0.34 * tiers + 0.95 * bands + 0.7


# ---------------------------------------------------------------------------
# 1. Shift timeline
# ---------------------------------------------------------------------------


def _work_blocks() -> list[tuple[int, float, float, str]]:
    """`(block number, start, end, shift name)` for every work block in `config.SHIFTS`."""
    blocks = []
    q = 0
    for shift, start, brk_start, brk_end, end in cfg.SHIFTS:
        for lo, hi in ((start, brk_start), (brk_end, end)):
            q += 1
            blocks.append((q, lo, hi, shift))
    return blocks


def shift_timeline(ax, label: str | None = None) -> None:
    """When decisions can be made. Reads only `config`; `label` is accepted and ignored."""
    blocks = _work_blocks()
    hours = [float(h) for h in range(int(cfg.HORIZON_START), int(cfg.HORIZON_END))]
    covered = {
        h: any(lo <= h and h + 1.0 <= hi for _q, lo, hi, _s in blocks) for h in hours
    }

    for q, lo, hi, shift in blocks:
        # MUTED grey for an unmapped shift, so it cannot pass for another shift.
        fill = SHIFT_COLOURS.get(shift, MUTED)
        ax.barh(0, hi - lo, left=lo, height=0.45, color=fill, linewidth=0, zorder=2)
        ax.text(
            (lo + hi) / 2, 0, f"block {q}", ha="center", va="center",
            color=_label_on(fill), fontsize=8, fontweight="semibold", zorder=3,
        )

    # Left-aligned at shift start: centred, it would land on the break gap. INK_2, as the
    # palest fill is unreadable as text.
    for shift, start, _brk_start, _brk_end, _end in cfg.SHIFTS:
        ax.text(
            start + 0.15, 0.30, f"shift {shift}", ha="left", va="bottom",
            color=INK_2, fontsize=8, fontweight="semibold",
        )

    for h in hours:
        if not covered[h]:
            ax.text(h + 0.5, -0.42, "break", ha="center", va="top", color=MUTED, fontsize=7)

    ax.set_ylim(-0.6, 0.62)
    ax.set_yticks([])
    _hour_ticks(ax, hours, every=2)
    ax.set_xlim(min(hours), max(hours) + 1)
    _frame(ax, ygrid=False)
    ax.spines["left"].set_visible(False)
    ax.set_title(
        f"Work blocks at {cfg.SITE} - people can only be reallocated at a block boundary"
    )


# ---------------------------------------------------------------------------
# 2. Headcount
# ---------------------------------------------------------------------------


def headcount(detail, axes, label: str | None = None, title: str | None = None,
              depts: list[str] | None = None) -> None:
    """Headcount per hour, planned vs recommended, one Axes per department.

    Bars, not a step line: a break drawn as a line is a cliff to zero that reads as a glitch.
    Pass the same `depts` to `size_headcount()`.
    """
    _one(detail, "headcount")
    depts = list(depts) if depts is not None else list(HEADCOUNT_DEPTS)
    if not depts:
        raise ValueError("headcount needs at least one department, got depts=[].")
    # Fail on the argument, not on a KeyError for a missing column.
    unknown = [d for d in depts if d not in HEADCOUNT_DEPTS]
    if unknown:
        raise ValueError(
            f"headcount got department(s) {unknown} that are not in the config; "
            f"choose from {HEADCOUNT_DEPTS}."
        )
    try:
        axes = list(axes)
    except TypeError:  # a single bare Axes
        axes = [axes]
    if len(axes) != len(depts):
        raise ValueError(
            f"headcount draws one panel per department: expected {len(depts)} "
            f"axes for {depts}, got {len(axes)}. Size the figure with "
            f"plt.subplots(1, len(depts), figsize=charts.size_headcount(depts)) - "
            f"or drop the `depts` argument for all of {HEADCOUNT_DEPTS}."
        )

    hrs = list(detail["hour"])
    series = {
        d: (list(detail[f"{d.lower()}_people_actual"]),
            list(detail[f"{d.lower()}_people_optimal"]))
        for d in depts
    }
    # Shared ceiling with headroom for delta labels, independent of sharey.
    top = max([v for a, o in series.values() for v in a + o] + [1.0]) * 1.22

    w = 0.42
    for ax, d in zip(axes, depts):
        a, o = series[d]
        ax.bar([h - w / 2 for h in hrs], a, w, color=ACTUAL, label="Planned", zorder=2)
        ax.bar([h + w / 2 for h in hrs], o, w, color=OPTIMAL, label="Recommended", zorder=3)

        # Label each contiguous run that changed, not each hour.
        for lo, hi in _runs_of_change(a, o):
            peak = max(range(lo, hi + 1), key=lambda i: abs(o[i] - a[i]))
            ax.text(
                (hrs[lo] + hrs[hi]) / 2, max(max(a[lo:hi + 1]), max(o[lo:hi + 1])),
                f"{o[peak] - a[peak]:+.0f}", ha="center", va="bottom",
                color=OPTIMAL, fontsize=8, fontweight="semibold",
            )

        # Panels are narrow, so fewer ticks.
        _hour_ticks(ax, hrs, every=6)
        # Padding so the end bars are not clipped by the spine.
        ax.set_xlim(min(hrs) - 0.7, max(hrs) + 0.7)
        ax.set_ylim(0, top)
        ax.set_title(cfg.display(d))
        _frame(ax)
    axes[0].set_ylabel("people")

    # Legend below the panels, clear of the x labels.
    fig = axes[0].get_figure()
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=names, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(_title(label, "recommended allocation, hour by hour", title), y=1.02,
                 fontsize=11, fontweight="semibold", color=INK)


def _runs_of_change(actual: list[float], optimal: list[float]) -> list[tuple[int, int]]:
    """Index ranges where the recommendation differs from the plan by more than half a person."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, (av, ov) in enumerate(zip(actual, optimal)):
        if abs(ov - av) > 0.5:
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(actual) - 1))
    return runs


# ---------------------------------------------------------------------------
# 3. Outbound backlog
# ---------------------------------------------------------------------------


def ob_backlog(detail, ax, label: str | None = None, title: str | None = None) -> None:
    """Outbound backlog against its target band, in hours of cover."""
    h = _one(detail, "ob_backlog")

    hrs = list(h["hour"])
    ax.fill_between(
        hrs, h["ob_bl_min_hours"], h["ob_bl_max_hours"],
        color=BAND, zorder=0, label="Target band",
    )
    # Band centre: a reference line, kept out of the legend.
    ax.plot(hrs, h["ob_bl_centre_hours"], color=AXIS, lw=0.9, ls=(0, (4, 3)), zorder=1)
    _pair_series(ax, hrs, h["ob_bl_actual_hours"], h["ob_bl_optimal_hours"])

    over = h[h["ob_bl_actual_hours"] > h["ob_bl_max_hours"]]
    if not over.empty:
        # Ink ring: it must differ from the marker by size, not hue.
        ax.scatter(over["hour"], over["ob_bl_actual_hours"], s=130, facecolors="none",
                   edgecolors=BREACH, linewidths=1.3, zorder=4)
        # Axes-relative so it stays clear of the title.
        ax.text(
            0.015, 0.95, f"{len(over)} hours above the band", transform=ax.transAxes,
            color=BREACH, fontsize=8.5, fontweight="semibold", va="top",
        )

    _hour_ticks(ax, hrs)
    ax.set_ylabel("hours of cover")
    ax.set_xlim(min(hrs), max(hrs))
    # Pad beyond the band so it still reads as a band.
    lo = min(float(h["ob_bl_min_hours"].min()), float(h["ob_bl_optimal_hours"].min(skipna=True)))
    hi = max(float(h["ob_bl_max_hours"].max()), float(h["ob_bl_actual_hours"].max(skipna=True)))
    ax.set_ylim(lo - 0.45, hi + 0.55)
    _frame(ax)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.14))
    ax.set_title(_title(label, "outbound backlog vs its band", title))


# ---------------------------------------------------------------------------
# 4. Inbound storage
# ---------------------------------------------------------------------------


def ib_storage(detail, ax, label: str | None = None, title: str | None = None) -> None:
    """Inbound backlog as a share of the storage ceiling."""
    h = _one(detail, "ib_storage")
    hrs = list(h["hour"])

    ax.axhline(1.0, color=LIMIT_LINE, lw=1.4, ls="--", zorder=1)
    # Left end: backlog climbs, so the right end is crowded by the planned series.
    ax.text(min(hrs), 1.015, "storage ceiling", ha="left", va="bottom",
            color=LIMIT_LINE, fontsize=8, fontweight="semibold")

    _pair_series(ax, hrs, h["ib_storage_actual_pct"], h["ib_storage_optimal_pct"])

    _hour_ticks(ax, hrs)
    ax.set_ylabel("share of storage ceiling")
    ax.set_xlim(min(hrs), max(hrs))
    ax.set_ylim(0, max(1.15, float(h["ib_storage_actual_pct"].max()) * 1.08))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    _frame(ax)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.14))
    ax.set_title(_title(label, "inbound backlog vs storage ceiling", title))


# ---------------------------------------------------------------------------
# 5. Carrier cut-off risk
# ---------------------------------------------------------------------------


def cpt_risk(detail, ax, label: str | None = None, title: str | None = None) -> None:
    """Deadline risk per carrier cut-off (the rows where `cpt_hour` is set), as bars."""
    c = _one(detail, "cpt_risk")
    c = c[c["cpt_hour"].notna()].reset_index(drop=True)
    x = list(range(len(c)))
    w = 0.38

    # Named in the legend: a floating label has no safe spot on this chart.
    ax.axhline(1.0, color=LIMIT_LINE, lw=1.4, ls="--", zorder=3, label="Risk tolerance")

    a = list(c["cpt_risk_actual_pct"])
    o = list(c["cpt_risk_optimal_pct"])
    ax.bar([i - w / 2 for i in x], a, w, color=ACTUAL, edgecolor=SURFACE, linewidth=2,
           label="Planned")
    ax.bar([i + w / 2 for i in x], o, w, color=OPTIMAL, edgecolor=SURFACE, linewidth=2,
           label="Recommended")

    # Label only the cut-offs at risk.
    for i, (av, ov) in enumerate(zip(a, o)):
        if av > 1.0:
            ax.text(i - w / 2, av, f"{av * 100:.0f}%", ha="center", va="bottom",
                    color=BREACH, fontsize=8, fontweight="semibold")
            ax.text(i + w / 2, ov, f"{ov * 100:.0f}%", ha="center", va="bottom",
                    color=OPTIMAL, fontsize=8, fontweight="semibold")

    ax.set_xticks(x)
    ax.set_xticklabels([cfg.hour_label(v) for v in c["cpt_hour"]])
    ax.set_xlabel("carrier cut-off")
    ax.set_ylabel("demand vs capacity to the deadline")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    _frame(ax)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.20))
    ax.set_title(_title(label, "carrier cut-off risk", title))


# ---------------------------------------------------------------------------
# 6. End of day
# ---------------------------------------------------------------------------


def eod_backlog(detail, ax, label: str | None = None, title: str | None = None) -> None:
    """End-of-day backlog against its band; the band and denominator are per-scenario `detail` columns."""
    h = _one(detail, "eod_backlog")
    row = h[h["hour"] == cfg.EOD_HOUR].iloc[0]
    den = float(row["eod_capacity_per_hour"])
    soft_max = float(row["eod_soft_max_hours"])
    hard_min = float(row["eod_hard_min_hours"])
    a = float(row["ob_total_actual_units"]) / den
    o = float(row["ob_total_optimal_units"]) / den

    ax.axhspan(hard_min, soft_max, color=BAND, zorder=0)
    ax.text(1.46, (hard_min + soft_max) / 2, "target\nband", ha="left",
            va="center", color=INK_2, fontsize=8)

    for i, (v, color, name) in enumerate(((a, ACTUAL, "Planned"), (o, OPTIMAL, "Recommended"))):
        ax.bar(i, v, 0.46, color=color, edgecolor=SURFACE, linewidth=2, label=name)
        over = v > soft_max
        ax.text(i, v, f"{v:.2f}h", ha="center", va="bottom", fontsize=9,
                fontweight="semibold", color=BREACH if over else color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Planned", "Recommended"])
    ax.set_ylabel(f"backlog at {cfg.hour_label(cfg.EOD_HOUR)}, hours of cover")
    ax.set_xlim(-0.6, 1.9)
    _frame(ax)
    ax.set_title(_title(label, "end-of-day backlog target", title or "End-of-day backlog target"))


# ---------------------------------------------------------------------------
# 7. The objective ladder
# ---------------------------------------------------------------------------


#: Ladder bands in priority order. Each gets its own panel and x-scale: the units differ,
#: and on one axis the largest would flatten the rest.
LADDER_BANDS = (
    ("slack", "Operational constraints", "units of constraint breached"),
    ("guardrail", "Bottom guardrails", "units from the target, over the horizon"),
    ("preference", "Preferences", "cost (movements, Support tie-break)"),
)


def _ladder_bands(review):
    """The bands actually present in this frame, in ladder order."""
    return [b for b in LADDER_BANDS if (review["kind"] == b[0]).any()]


def ladder_ratios(review) -> list[int]:
    """Height ratio per band panel - one unit per tier, so every row is the same height."""
    return [int((review["kind"] == kind).sum()) for kind, _, _ in _ladder_bands(review)]


def objective_ladder(review, axes, label: str | None = None, title: str | None = None) -> None:
    """What the model gave up, per tier, one Axes per band.

    Every tier gets a row whether or not it fired. Size the figure with `size_ladder` and
    `ladder_ratios`.
    """
    lad = _one(review, "objective_ladder")
    bands = _ladder_bands(lad)
    try:
        axes = list(axes)
    except TypeError:
        axes = [axes]
    if len(axes) != len(bands):
        raise ValueError(
            f"objective_ladder draws one panel per ladder band: expected {len(bands)} "
            f"axes for {[b[1] for b in bands]}, got {len(axes)}. Build the figure with "
            f"plt.subplots(len(charts.ladder_ratios(review)), 1, "
            f"figsize=charts.size_ladder(review), "
            f"height_ratios=charts.ladder_ratios(review))."
        )

    hgt = 0.36
    for ax, (kind, band_name, unit) in zip(axes, bands):
        rows = lad[lad["kind"] == kind].reset_index(drop=True)
        y = list(range(len(rows)))[::-1]
        a = list(rows["slack_actual"])
        o = list(rows["slack_optimal"])

        _pair_barh(ax, y, a, o, hgt,
                   ("Before (the plan)", "After (the recommendation)"))

        lo = min([0.0] + a + o)
        hi = max([0.0] + a + o)
        span = max(hi - lo, 1.0)
        for v, av, ov, outcome in zip(y, a, o, rows["outcome"]):
            if max(abs(av), abs(ov)) <= run.EPS:
                # Never fired: print "0" so the row does not read as missing data.
                ax.text(span * 0.012, v, "0", va="center", fontsize=8, color=MUTED)
                continue
            ax.text(max(av, 0) + span * 0.012, v + hgt / 2, f"{av:,.0f}", va="center",
                    fontsize=8, color=ACTUAL, fontweight="semibold")
            tag = "cleared" if outcome == "resolved" else outcome
            color = RESOLVED if outcome == "resolved" else (
                INK_2 if kind != "slack" else BREACH
            )
            ax.text(max(ov, 0) + span * 0.012, v - hgt / 2, f"{ov:,.0f}  {tag}",
                    va="center", fontsize=8, color=color, fontweight="semibold")

        ax.set_yticks(y)
        ax.set_yticklabels([f"{p}. {lb}" for p, lb in zip(rows["priority"], rows["label"])])
        ax.tick_params(axis="y", length=0, labelcolor=INK_2)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlim(lo - span * 0.02, hi + span * 0.42)
        ax.set_xlabel(unit, fontsize=8)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        _frame_x(ax)
        # Band name inside the plot, saving vertical space.
        ax.text(0.0, 1.02, band_name, transform=ax.transAxes, fontsize=9,
                fontweight="semibold", color=INK, va="bottom")

    # Upper right: high-priority tiers there usually cleared, leaving room; the bottom has long bars.
    axes[0].legend(loc="upper right", ncol=1, fontsize=8)
    fig = axes[0].get_figure()
    fig.suptitle(
        _title(label, "the priority ladder", title)
        + "\nsolved top to bottom; a lower tier can never trade against a higher one",
        fontsize=11, fontweight="semibold", color=INK, ha="left", x=0.0,
    )


# ---------------------------------------------------------------------------
# 10. How many constraints a real day puts under pressure
# ---------------------------------------------------------------------------


def fit_diagnostics(persistence, profiles, arrivals, calibration, axes,
                    title: str | None = None) -> None:
    """Four panels comparing generated inputs to the fitted data (fitted = ACTUAL colour, generated = OPTIMAL).

    `axes` is four Axes in panel order A-D.
    """
    ax_phi, ax_prof, ax_arr, ax_cal = axes

    # --- A. persistence: the finding that decided the generator's shape -------------
    # Sorted by kind so the gap between levels and flows is visible.
    p = persistence.sort_values(["kind", "phi_fitted"], ascending=[True, True])
    y = np.arange(len(p))
    hgt = 0.38
    _pair_barh(ax_phi, y, p["phi_fitted"], p["phi_generated"], hgt,
               ("fitted", "generated"), edged=False)
    ax_phi.set_yticks(y)
    ax_phi.set_yticklabels(p["series"], fontsize=8)
    ax_phi.set_xlim(0, 1.0)
    ax_phi.set_xlabel("lag-1 autocorrelation")
    # Rule between regimes: flows sort first, so flows are below it. No in-axes labels; they
    # would land on level bars.
    flows = int((p["kind"] == "flow").sum())
    ax_phi.axhline(flows - 0.5, color=AXIS, lw=0.8)
    ax_phi.legend(loc="lower right", fontsize=8)
    _frame(ax_phi, ygrid=False)
    ax_phi.grid(axis="x", color=GRID, linewidth=0.8)
    ax_phi.set_title("Levels persist, flows do not", fontsize=10)

    # --- B. hour-of-day profile of the anchored series ------------------------------
    h = profiles["hour_of_day"]
    ax_prof.plot(h, profiles["backlog_fitted"], color=ACTUAL, lw=LINE_W, label="fitted")
    ax_prof.plot(h, profiles["backlog_generated"], color=OPTIMAL, lw=LINE_W,
                 ls="--", label="generated")
    ax_prof.set_xticks(list(range(0, 24, 4)))
    ax_prof.set_xlabel("hour of the day")
    ax_prof.set_ylabel("index, mean = 1")
    ax_prof.legend(loc="lower right", fontsize=8)
    _frame(ax_prof)
    ax_prof.set_title("Backlog keeps its shape over the day", fontsize=10)

    # --- C. the left tail that a naive generator clips away -------------------------
    # Clipped to a readable range; the tails run to +/-8 and would squash the body.
    a = arrivals["arrivals_index"].to_numpy()
    lo, hi = -3.0, 4.0
    ax_arr.hist(np.clip(a, lo, hi), bins=56, range=(lo, hi),
                color=ACTUAL, edgecolor=SURFACE, linewidth=0.4)
    ax_arr.axvspan(lo, 0.0, color=BAND, zorder=0)
    ax_arr.axvline(0.0, color=LIMIT_LINE, lw=1.2, ls="--")
    ax_arr.set_xlim(lo, hi)
    top = ax_arr.get_ylim()[1]
    ax_arr.annotate("holds, cancellations,\nthe pick-to-ship gap",
                    xy=(-0.1, top * 0.72), xytext=(lo + 0.15, top * 0.92),
                    fontsize=8, color=LIMIT_LINE, va="top",
                    arrowprops=dict(arrowstyle="->", color=LIMIT_LINE, lw=0.9))
    ax_arr.set_xlabel("hourly arrivals, index")
    ax_arr.set_ylabel("hours")
    _frame(ax_arr)
    ax_arr.set_title("Arrivals go negative, and that is real", fontsize=10)

    # --- D. generated against fitted, as ratios -------------------------------------
    c = calibration.iloc[::-1].reset_index(drop=True)
    yc = np.arange(len(c))
    ax_cal.axvline(1.0, color=LIMIT_LINE, lw=1.2, ls="--", zorder=1)
    ax_cal.hlines(yc, 1.0, c["ratio"], color=AXIS, lw=1.0, zorder=2)
    ax_cal.scatter(c["ratio"], yc, s=38, color=OPTIMAL, zorder=3)
    # Label on the side of the marker away from the 1.0 rule.
    for yi, r in zip(yc, c["ratio"]):
        side = 1 if r >= 1.0 else -1
        ax_cal.annotate(f"{r:.2f}", (r, yi), xytext=(9 * side, 0),
                        textcoords="offset points", fontsize=8, color=INK_2,
                        ha="left" if side > 0 else "right", va="center")
    ax_cal.set_yticks(yc)
    ax_cal.set_yticklabels(c["quantity"], fontsize=8)
    ax_cal.set_xlabel("generated / fitted")
    ax_cal.set_xlim(0.8, 1.35)
    _frame(ax_cal, ygrid=False)
    ax_cal.grid(axis="x", color=GRID, linewidth=0.8)
    ax_cal.set_title("Within a tenth, except where it is not", fontsize=10)

    if title:
        axes[0].figure.suptitle(title, fontsize=11, fontweight="semibold")


SIZE_FIT = (10.0, 6.6)
