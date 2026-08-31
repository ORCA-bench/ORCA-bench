"""1×3 horizontal-bar retrieval chart (Figure 8).

Renders three per-model diagnostics on how well each agent retrieves each
telemetry modality, in one PDF:

1. **Mean telemetry commands / trial** — from
   ``output_dir/telemetry_classifications/`` via
   :func:`format_telemetry_calls.aggregate_telemetry_classifications`.
2. **Failure rate (%)** — one stacked Empty (grey) + Error (red) bar per model,
   the per-trial shares of *all* classified telemetry calls, as a per-model
   mean ± SEM (combined whisker) with a bold total-% label.
3. **Any-match (%) by telemetry type** — grouped bars (metrics / logs / traces),
   the ``metrics_any_match`` / ``logs_any_match`` / ``traces_any_match`` judge
   rollups from :func:`utils.load_data`, shown as per-model mean ± SEM.

Panel (a) is a neutral magnitude metric (Blues shades) and panel (b) a stacked
Empty/Error bar per model; panel (c) is stratified by telemetry type. The two
right panels are drawn wider than (a). Model identity comes from the shared
y-axis tick labels on the leftmost panel.

Examples::

    cd examples/otel-demo
    uv run python plot_retrieval.py -od out-0619-3 -jd jobs-sub
"""

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import sem

from format_telemetry_calls import (
    LABEL_COLORS,
    TELEM_TYPES,
    _short_group_label,
    aggregate_telemetry_classifications,
    build_trial_to_group,
)
from harbor_utils.plotting_utils import (
    ICLR_WIDTH,
    LEGEND_FONT_SIZE,
    OUTWARD,
    TICK_FONT_SIZE,
)
from utils import (
    MODEL_ALIASES,
    get_base_parser,
    load_data,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Local rename applied just for this figure (keeps the y-axis tick narrow
# enough to fit the leftmost panel). Upstream ``MODEL_ALIASES`` keep the
# longer "DeepSeek-V4-Pro" form for tables and other plots.
_LOCAL_RENAMES: dict[str, str] = {"DeepSeek-V4-Pro": "DS-V4-Pro"}


def _relabel(name: str) -> str:
    return _LOCAL_RENAMES.get(name, name)


# Top-to-bottom model order shown in the figure. Diverges from
# ``format_telemetry_calls.MODEL_ORDER`` to match the reference screenshot:
# open-weight models (DeepSeek, GLM) on top, then closed-source frontier
# models (Sonnet, Opus, GPT) underneath. Adjust here, not upstream.
MODEL_ORDER_TOP_DOWN: tuple[str, ...] = (
    "DS-V4-Pro",
    "GLM-5",
    "Sonnet 4.6",
    "Opus 4.7",
    "GPT-5.5",
)

# Colors for the grouped-by-type bars in panels (b) and (c). Iterated in
# ``TELEM_TYPES`` order (metrics, logs, traces) so the legend is stable.
TELEM_TYPE_COLORS: dict[str, str] = {
    "metrics": "#1f77b4",  # blue
    "logs": "#ff7f0e",  # orange
    "traces": "#9467bd",  # purple
}

# Judge rollup column feeding each telemetry-type bar in panel (c).
TYPE_MATCH_COL: dict[str, str] = {
    "metrics": "metrics_any_match",
    "logs": "logs_any_match",
    "traces": "traces_any_match",
}


def _mean_sem(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard error of the mean (0.0 for empty / single-element)."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    se = float(sem(arr)) if arr.size > 1 else 0.0
    return mean, se


def _draw_value_labels(
    ax: plt.Axes,
    y: np.ndarray,
    vals: np.ndarray,
    fmt: str,
    color: str = "#444444",
    bold: bool = False,
    pos: np.ndarray | None = None,
) -> None:
    """Annotate each bar with the formatted ``vals``.

    ``pos`` (optional) is the x-coordinate where each label is rendered —
    default is ``vals`` itself, but pass ``vals + sem`` when bars carry
    error whiskers so the text sits just outside the cap.
    """
    weight = "bold" if bold else "normal"
    positions = pos if pos is not None else vals
    for yi, v, p in zip(y, vals, positions):
        if v <= 0:
            continue
        ax.text(
            p,
            yi,
            " " + fmt.format(v),
            va="center",
            ha="left",
            fontsize=TICK_FONT_SIZE - 1,
            color=color,
            fontweight=weight,
        )


def _draw_grouped_by_type(
    ax: plt.Axes,
    y: np.ndarray,
    models: list[str],
    data: dict[str, dict[str, tuple[float, float]]],
    err_kw: dict,
) -> list:
    """Draw one grouped-by-telemetry-type horizontal bar cluster per model.

    ``data`` maps ``{display_model: {ttype: (mean, sem)}}``. Within each model's
    y-slot, one thin bar per telemetry type (metrics on top, traces on the
    bottom once the y-axis is inverted), colored by :data:`TELEM_TYPE_COLORS`.
    Returns the bar-container handles so the caller can build a shared legend.
    """
    types = list(TELEM_TYPES)
    n_types = len(types)
    h = 0.8 / n_types
    handles = []
    for i, ttype in enumerate(types):
        # metrics (i=0) gets the most-negative offset so it sits at the top of
        # the cluster after ``invert_yaxis`` flips the axis downward.
        offset = (i - (n_types - 1) / 2) * h
        vals = np.array([data.get(m, {}).get(ttype, (0.0, 0.0))[0] for m in models])
        errs = np.array([data.get(m, {}).get(ttype, (0.0, 0.0))[1] for m in models])
        bar = ax.barh(
            y + offset,
            vals,
            height=h * 0.9,
            color=TELEM_TYPE_COLORS[ttype],
            label=ttype,
            edgecolor="white",
            linewidth=0.5,
            xerr=errs,
            error_kw=err_kw,
        )
        handles.append(bar)
    return handles


def main() -> None:
    """Render the 1×3 retrieval bar chart and save as PDF."""
    parser = get_base_parser()
    parser.description = (
        "Per-model retrieval figure: mean telemetry commands, per-type failure "
        "rate, and per-type any-match rate."
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Panels (a) and (b): telemetry-call aggregates from the classifications.
    trial_to_group = build_trial_to_group(args.jobs_dir)
    telem = aggregate_telemetry_classifications(
        args.output_dir / "telemetry_classifications", trial_to_group
    )
    calls_by_model: dict[str, tuple[float, float]] = {
        _relabel(_short_group_label(g)): _mean_sem(per_trial)
        for g, per_trial in telem.per_trial_calls_by_group.items()
    }
    # Panel (b) failure rate: per-model Empty% and Error% over ALL telemetry
    # calls (not split by type). Per-trial fractions are meaned (with SEM);
    # trials with zero classified calls are dropped (no denominator).
    empty_pct_by_model: dict[str, tuple[float, float]] = {}
    error_pct_by_model: dict[str, tuple[float, float]] = {}
    for g, per_trial_counts in telem.per_trial_label_counts_by_group.items():
        empty_fracs: list[float] = []
        error_fracs: list[float] = []
        for tc in per_trial_counts:
            total = tc["success"] + tc["empty"] + tc["error"]
            if total == 0:
                continue
            empty_fracs.append(100 * tc["empty"] / total)
            error_fracs.append(100 * tc["error"] / total)
        m = _relabel(_short_group_label(g))
        empty_pct_by_model[m] = _mean_sem(empty_fracs)
        error_pct_by_model[m] = _mean_sem(error_fracs)

    # Panel (c): per-type any-match rate from the judge rollups.
    # TODO: once ``utils.load_data`` can filter to a single judge model at the
    # source, select only that judge's rows here so each trial counts once.
    # Until then we pool over every scored row load_data returns.
    df = load_data(args.jobs_dir, args.effort)
    df["_disp"] = df["_agent_model"].map(
        lambda m: _relabel(MODEL_ALIASES.get(m, m)) if isinstance(m, str) else m
    )
    match_by_model_type: dict[str, dict[str, tuple[float, float]]] = {}
    for disp, grp in df.groupby("_disp"):
        by_type: dict[str, tuple[float, float]] = {}
        for ttype, col in TYPE_MATCH_COL.items():
            vals = [int(v) for v in grp[col].tolist() if pd.notna(v)]
            mean, se = _mean_sem(vals)
            by_type[ttype] = (100 * mean, 100 * se)
        match_by_model_type[str(disp)] = by_type

    available = set(calls_by_model) | set(empty_pct_by_model) | set(match_by_model_type)
    models_top_down = [m for m in MODEL_ORDER_TOP_DOWN if m in available]
    n = len(models_top_down)
    if n == 0:
        logger.error("No models found across the data sources; nothing to plot.")
        return
    y = np.arange(n)

    # Blues shade map for the neutral magnitude panel (a), fading from dark
    # (top model) to light (bottom model) — matches the reference screenshot.
    blue_shades = plt.get_cmap("Blues")(np.linspace(0.85, 0.35, n))

    fig, axes = plt.subplots(
        1,
        3,
        # ICLR_HEIGHT (= 2") is a touch tall for a single-row barh; override
        # locally to 1.5" so the panels sit closer to a 3:1 aspect ratio. The
        # two grouped-by-type panels (b)/(c) are drawn 1.5× wider than the
        # single-bar telemetry-calls panel (a).
        figsize=(ICLR_WIDTH, 1.5),
        sharey=True,
        gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1.5, 1.5]},
    )

    # Common kwargs for the thin black SEM whiskers on every panel.
    err_kw = {
        "ecolor": "#444444",
        "elinewidth": 0.8,
        "capsize": 1.5,
        "capthick": 0.8,
    }

    # Panel (a): Mean telemetry commands per trial
    ax = axes[0]
    means_sems = [calls_by_model.get(m, (0.0, 0.0)) for m in models_top_down]
    vals = np.array([m for m, _ in means_sems])
    errs = np.array([s for _, s in means_sems])
    ax.barh(
        y,
        vals,
        color=blue_shades,
        height=0.65,
        edgecolor="white",
        linewidth=0.5,
        xerr=errs,
        error_kw=err_kw,
    )
    ax.set_xlabel(
        "(a) # telemetry\ncommands", fontsize=TICK_FONT_SIZE, fontweight="bold"
    )
    _draw_value_labels(ax, y, vals, "{:.1f}", pos=vals + errs)
    ax.set_yticks(y, labels=models_top_down)

    # Common styling for the legends drawn inside the right-hand panels.
    inside_leg_kw = {
        "fontsize": LEGEND_FONT_SIZE - 1,
        "frameon": True,
        "framealpha": 0.8,
        "edgecolor": "none",
        "handlelength": 1.0,
        "handletextpad": 0.4,
        "borderpad": 0.3,
        "labelspacing": 0.3,
    }

    # Panel (b): Failure rate (%) — one stacked Empty (grey) + Error (red) bar
    # per model over all telemetry calls, with a combined SEM whisker.
    ax = axes[1]
    empty_ms = [empty_pct_by_model.get(m, (0.0, 0.0)) for m in models_top_down]
    error_ms = [error_pct_by_model.get(m, (0.0, 0.0)) for m in models_top_down]
    empty_vals = np.array([m for m, _ in empty_ms])
    error_vals = np.array([m for m, _ in error_ms])
    empty_errs = np.array([s for _, s in empty_ms])
    error_errs = np.array([s for _, s in error_ms])
    ax.barh(
        y,
        empty_vals,
        color=LABEL_COLORS["empty"],
        height=0.65,
        label="empty",
        edgecolor="white",
        linewidth=0.5,
    )
    # Error segment carries the combined Empty+Error SEM (pooled in quadrature),
    # drawn on the right of the stack so it attaches to the total.
    combined_errs = np.sqrt(empty_errs**2 + error_errs**2)
    ax.barh(
        y,
        error_vals,
        left=empty_vals,
        color=LABEL_COLORS["error"],
        height=0.65,
        label="error",
        edgecolor="white",
        linewidth=0.5,
        xerr=combined_errs,
        error_kw=err_kw,
    )
    ax.set_xlabel("(b) Failure\nrate (%)", fontsize=TICK_FONT_SIZE, fontweight="bold")
    totals = empty_vals + error_vals
    _draw_value_labels(
        ax, y, totals, "{:.1f}%", color="#d62728", bold=True, pos=totals + combined_errs
    )
    # Empty/Error legend inside the panel, lower-right. Extra right headroom is
    # added below (the failure bars are wide) so it clears the value labels.
    shade_handles = [
        Patch(facecolor=LABEL_COLORS["empty"], edgecolor="white", label="empty"),
        Patch(facecolor=LABEL_COLORS["error"], edgecolor="white", label="error"),
    ]
    ax.legend(handles=shade_handles, loc="lower right", **inside_leg_kw)

    # Panel (c): Any-match rate (%) grouped by telemetry type
    ax = axes[2]
    _draw_grouped_by_type(ax, y, models_top_down, match_by_model_type, err_kw)
    ax.set_xlabel("(c) Any-match\n(%)", fontsize=TICK_FONT_SIZE, fontweight="bold")
    # metrics/logs/traces legend in the upper-right whitespace (top models have
    # short any-match bars there).
    type_patches = [
        Patch(facecolor=TELEM_TYPE_COLORS[t], edgecolor="white", label=t)
        for t in TELEM_TYPES
    ]
    ax.legend(handles=type_patches, loc="upper right", **inside_leg_kw)

    # barh draws y=0 at the bottom; flip so MODEL_ORDER_TOP_DOWN[0] sits at
    # the top of the figure. Axes share y, so one invert propagates.
    axes[0].invert_yaxis()

    for ax in axes:
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_position(("outward", OUTWARD))
        ax.spines["bottom"].set_position(("outward", OUTWARD))
        # Right headroom so value labels / whiskers aren't clipped. Panel (b)
        # gets extra room to seat its lower-right Empty/Error legend clear of the
        # bars and total-% labels.
        headroom = 1.55 if ax is axes[1] else 1.18
        _, xhi = ax.get_xlim()
        ax.set_xlim(0, max(xhi, 1e-9) * headroom)

    fig.tight_layout()
    fig_path = args.output_dir / "figures" / "figure8_retrieval.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
