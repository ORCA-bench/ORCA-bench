"""1×4 horizontal-bar efficiency chart (Figure 8).

Renders four per-model inefficiency metrics in one PDF:

1. **Mean total tokens / trial** — from
   :func:`harbor_utils.token_utils.collect_harbor_token_usage` over ``JOBS_DIR``.
2. **Mean total telemetry calls / trial** — from
   ``CALLS_OD/telemetry_classifications/`` via
   :func:`format_telemetry_calls.aggregate_telemetry_classifications`.
3. **Failure rate (%)** — stacked Empty (grey) + Error (red) shares of the
   same telemetry classifications.
4. **Redundant calls (%)** — from ``CALLS_OD/redundant_telemetry_calls/`` via
   :func:`format_redundant_calls.aggregate_redundant_classifications`.

All four metrics are lower-is-better. Each panel uses solid horizontal bars
with a ``Reds`` shade map per model (darker = top row in the screenshot's
DS-V4-Pro → GPT-5.5 ordering). No hatching — model identity comes from the
shared y-axis tick labels on the leftmost panel.

Examples::

    cd examples/otel-demo
    uv run python plot_efficiency.py
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

from format_redundant_calls import aggregate_redundant_classifications
from format_telemetry_calls import (
    LABEL_COLORS,
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
from harbor_utils.token_utils import collect_harbor_token_usage
from utils import MODEL_ALIASES, model_display, setup_logging

logger = logging.getLogger(__name__)

TOKENS_OD = Path("out-0522")
CALLS_OD = Path("out-0518-sample100")
JOBS_DIR = Path("jobs-sub")
FIG_PATH = CALLS_OD / "figures" / "figure8_efficiency.pdf"

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

# Scale factor for the tokens panel: report in millions so the axis numbers
# stay one-to-two digits across all five models.
TOKEN_SCALE = 1_000_000
TOKEN_SCALE_SUFFIX = "M"


def _tokens_per_trial(jobs_dir: Path) -> dict[str, list[float]]:
    """Return ``{display_model: [per-trial total_tokens]}``.

    The underlying ``collect_harbor_token_usage`` walks every
    ``trajectory.json`` under ``jobs_dir`` (~5 min over jobs-sub) but is itself
    joblib-cached, so this wrapper no longer needs its own cache. We return
    the raw per-trial values so callers can compute mean + SEM (or any other
    summary) without re-walking. ``collect_harbor_token_usage`` reports the
    LiteLLM-qualified model name (e.g. ``gradient_ai/anthropic-claude-opus-4.7``);
    strip the provider prefix with :func:`model_display` before looking it up
    in :data:`MODEL_ALIASES` so the keys match the telemetry/redundant
    panels (``Opus 4.7``, ``DeepSeek-V4-Pro``, ...).
    """
    records = collect_harbor_token_usage(jobs_dir, include_reasoning_effort=False)
    if not records:
        return {}
    df = pd.DataFrame(records)
    df["total_tokens"] = df["total_prompt_tokens"] + df["total_completion_tokens"]
    df["_display_model"] = df["model_name"].map(
        lambda m: MODEL_ALIASES.get(model_display(m), model_display(m))
    )
    return {
        str(model): grp["total_tokens"].astype(float).tolist()
        for model, grp in df.groupby("_display_model")
    }


def _mean_sem(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard error of the mean (0.0 for empty / single-element)."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    se = float(sem(arr)) if arr.size > 1 else 0.0
    return mean, se


def _by_display(
    counts_by_group: dict[str, object],
) -> dict[str, object]:
    """Collapse ``{group: value}`` to ``{display_name: value}`` via _short_group_label."""
    return {_short_group_label(g): v for g, v in counts_by_group.items()}


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


def main() -> None:
    """Render the 1×4 efficiency bar chart and save as PDF."""
    setup_logging(logging.INFO)

    # _tokens_per_trial keys on the raw display name (e.g. ``DeepSeek-V4-Pro``);
    # apply the figure-local rename here so updates to ``_LOCAL_RENAMES`` take
    # effect without invalidating the joblib cache inside
    # ``collect_harbor_token_usage`` (the slow trajectory walk). Returned shape:
    # ``{display_model: (mean, sem)}``.
    tokens_by_model: dict[str, tuple[float, float]] = {
        _relabel(k): _mean_sem(v) for k, v in _tokens_per_trial(JOBS_DIR).items()
    }

    trial_to_group = build_trial_to_group(JOBS_DIR)
    telem = aggregate_telemetry_classifications(
        CALLS_OD / "telemetry_classifications", trial_to_group, invalid_ids=set()
    )
    calls_by_model: dict[str, tuple[float, float]] = {
        _relabel(_short_group_label(g)): _mean_sem(per_trial)
        for g, per_trial in telem.per_trial_calls_by_group.items()
    }
    # Per-trial fractions → per-model mean + SEM for the stacked failure panel.
    # Trials with zero classified calls are dropped (no denominator).
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

    redun = aggregate_redundant_classifications(
        CALLS_OD / "redundant_telemetry_calls", trial_to_group, invalid_ids=set()
    )
    redun_pct_by_model: dict[str, tuple[float, float]] = {}
    for g, per_trial_counts in redun.per_trial_label_counts_by_group.items():
        fracs: list[float] = []
        for tc in per_trial_counts:
            total = tc["unique"] + tc["redundant"]
            if total == 0:
                continue
            fracs.append(100 * tc["redundant"] / total)
        redun_pct_by_model[_relabel(_short_group_label(g))] = _mean_sem(fracs)

    available = set(tokens_by_model) | set(calls_by_model) | set(redun_pct_by_model)
    models_top_down = [m for m in MODEL_ORDER_TOP_DOWN if m in available]
    n = len(models_top_down)
    if n == 0:
        logger.error("No models found across the three data sources; nothing to plot.")
        return
    y = np.arange(n)

    # Panels (a) tokens and (b) telemetry calls are neutral magnitude metrics,
    # so use a Blues shade map; (c) failure rate and (d) redundancy are
    # "lower is better", so use Reds. Both maps fade from dark (top model) to
    # light (bottom model) — matches the reference screenshot.
    blue_shades = plt.get_cmap("Blues")(np.linspace(0.85, 0.35, n))
    red_shades = plt.get_cmap("Reds")(np.linspace(0.85, 0.35, n))

    fig, axes = plt.subplots(
        1,
        4,
        # ICLR_HEIGHT (= 2") is a touch tall for a single-row barh; override
        # locally to 1.5" so the four panels sit closer to a 3:1 aspect ratio.
        figsize=(ICLR_WIDTH, 1.5),
        sharey=True,
        gridspec_kw={"wspace": 0.3},
    )

    # Common kwargs for the thin black SEM whiskers on every panel.
    err_kw = {
        "ecolor": "#444444",
        "elinewidth": 0.8,
        "capsize": 1.5,
        "capthick": 0.8,
    }

    # Panel 1: Mean total tokens (in millions)
    ax = axes[0]
    means_sems = [tokens_by_model.get(m, (0.0, 0.0)) for m in models_top_down]
    vals = np.array([m / TOKEN_SCALE for m, _ in means_sems])
    errs = np.array([s / TOKEN_SCALE for _, s in means_sems])
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
        f"(a) Total\ntokens ({TOKEN_SCALE_SUFFIX})",
        fontsize=TICK_FONT_SIZE,
        fontweight="bold",
    )
    _draw_value_labels(ax, y, vals, "{:.1f}", pos=vals + errs)
    ax.set_yticks(y, labels=models_top_down)

    # Panel 2: Mean total telemetry calls
    ax = axes[1]
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
        "(b) Total\ntelemetry calls", fontsize=TICK_FONT_SIZE, fontweight="bold"
    )
    _draw_value_labels(ax, y, vals, "{:.1f}", pos=vals + errs)

    # Panel 3: Failure rate (stacked Empty + Error)
    ax = axes[2]
    empty_means_sems = [empty_pct_by_model.get(m, (0.0, 0.0)) for m in models_top_down]
    error_means_sems = [error_pct_by_model.get(m, (0.0, 0.0)) for m in models_top_down]
    empty_vals = np.array([m for m, _ in empty_means_sems])
    error_vals = np.array([m for m, _ in error_means_sems])
    empty_errs = np.array([s for _, s in empty_means_sems])
    error_errs = np.array([s for _, s in error_means_sems])
    ax.barh(
        y,
        empty_vals,
        color=LABEL_COLORS["empty"],
        height=0.65,
        label="Empty",
        edgecolor="white",
        linewidth=0.5,
    )
    # Error segment carries the SEM whisker for the combined Empty+Error
    # share — drawn on the right of the stack so it visually attaches to
    # the total. Empty/Error errors are pooled in quadrature (independent
    # per-trial fractions); approximation matches what readers expect from
    # a stacked-bar SEM annotation.
    combined_errs = np.sqrt(empty_errs**2 + error_errs**2)
    ax.barh(
        y,
        error_vals,
        left=empty_vals,
        color=LABEL_COLORS["error"],
        height=0.65,
        label="Error",
        edgecolor="white",
        linewidth=0.5,
        xerr=combined_errs,
        error_kw=err_kw,
    )
    ax.set_xlabel("(c) Failure\nrate (%)", fontsize=TICK_FONT_SIZE, fontweight="bold")
    totals = empty_vals + error_vals
    _draw_value_labels(
        ax,
        y,
        totals,
        "{:.1f}%",
        color="#d62728",
        bold=True,
        pos=totals + combined_errs,
    )
    # Legend centered above panel 3, mirroring the screenshot layout.
    ax.legend(
        fontsize=LEGEND_FONT_SIZE - 1,
        loc="lower center",
        ncol=2,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 1.02),
    )

    # Panel 4: Redundant calls
    ax = axes[3]
    means_sems = [redun_pct_by_model.get(m, (0.0, 0.0)) for m in models_top_down]
    vals = np.array([m for m, _ in means_sems])
    errs = np.array([s for _, s in means_sems])
    ax.barh(
        y,
        vals,
        color=red_shades,
        height=0.65,
        edgecolor="white",
        linewidth=0.5,
        xerr=errs,
        error_kw=err_kw,
    )
    ax.set_xlabel(
        "(d) Redundant\ncalls (%)", fontsize=TICK_FONT_SIZE, fontweight="bold"
    )
    _draw_value_labels(ax, y, vals, "{:.1f}%", pos=vals + errs)

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
        # Leave 18% headroom on the right so value labels don't get clipped.
        xlo, xhi = ax.get_xlim()
        ax.set_xlim(0, max(xhi, 1e-9) * 1.18)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
