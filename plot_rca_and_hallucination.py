r"""1×3 figure: RCA Accuracy (Medium), RCA Accuracy (Hard), Hallucination Rate.

Side-by-side bar charts (one bar per agent_model) for three slices of the
incident-task data. We show the two harder difficulty tiers (Medium and
Hard); Easy is omitted here because it looks deceptively good and pulls
attention from the realistic setting. Generation emits the ``_granularity``
ladder ``easy``/``medium``/``hard`` directly (``utils.GRANULARITY_ALIASES``
is now an identity passthrough).

1. **RCA Accuracy (Medium)** — ``feature_flag_all_match`` mean restricted
   to ``_granularity == "medium"``: a broad feature-area complaint.
2. **RCA Accuracy (Hard)** — same metric restricted to
   ``_granularity == "hard"``: a flag-agnostic question (no hint which
   feature flag failed).
3. **Hallucination Rate** — ``hallucinate_any`` mean across all incident
   tasks (lower is better, ↓; the RCA panels are higher is better, ↑).

Bars are colored solidly by model using the shared palette from
``plot_human_vs_llm_scatter_grid.py`` so model identity is consistent
across the paper; the ↑/↓ in each title marks the metric's polarity.

Examples::

    cd examples/otel-demo
    uv run python plot_rca_and_hallucination.py -od out-0520-2 -e high

"""

import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

from harbor_utils.plotting_utils import (
    ICLR_WIDTH,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    OUTWARD,
    TICK_FONT_SIZE,
)
from utils import (
    Q_COLS,
    get_base_parser,
    load_data,
    model_alias,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Match the font of plot_human_vs_llm_scatter_grid.py (Fira Code, monospace).
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["Fira Code"]

# Per-model palette shared with plot_human_vs_llm_scatter_grid.py so model
# identity reads consistently across the paper. Keyed by a substring of the
# ``_agent_model`` display name (which varies, e.g. ``claude-4.6-sonnet`` vs
# ``claude-sonnet-4.6``), checked in order.
SCATTER_COLORS: dict[str, str] = {
    "opus": "#2166ac",
    "sonnet": "#92c5de",
    "gpt": "#4daf4a",
    "glm": "#e41a1c",
    "deepseek": "#984ea3",
}


def _model_color(model: str) -> str:
    """Return the scatter-grid palette color for ``model`` (grey fallback)."""
    name = model.lower()
    for key, color in SCATTER_COLORS.items():
        if key in name:
            return color
    return "#777777"


def _bar_stats(
    sub_df: pd.DataFrame, field: str, model_order: list[str]
) -> tuple[list[float], list[float], list[int]]:
    """Return (means_pct, ses_pct, n) per model in ``model_order``.

    Drops NaN values within each model slice before computing the mean
    and standard error; missing models render as bar height 0 with N=0.
    """
    means: list[float] = []
    ses: list[float] = []
    ns: list[int] = []
    for m in model_order:
        vals = (
            sub_df.loc[sub_df["_agent_model"].astype(str) == m, field]
            .dropna()
            .astype(float)
            .to_numpy()
        )
        if vals.size == 0:
            means.append(0.0)
            ses.append(0.0)
            ns.append(0)
            continue
        means.append(float(vals.mean()) * 100)
        ses.append(float(sem(vals)) * 100 if vals.size > 1 else 0.0)
        ns.append(int(vals.size))
    return means, ses, ns


def main() -> None:
    """Render the three-panel RCA + hallucination bar chart."""
    parser = get_base_parser()
    parser.description = (
        "Three-panel bar chart: RCA Accuracy at hard/universal "
        "granularities and % Complete Hallucination."
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    df = load_data(args.output_dir, args.effort, args.filter_xlsx)
    df = df[~df["_is_control"]]
    if df.empty:
        logger.error("No incident trials found.")
        return

    # Preserve the MODEL_ORDER baked into the Categorical _agent_model.
    model_order: list[str] = [
        str(m)
        for m in df["_agent_model"].cat.categories
        if m in df["_agent_model"].values
    ]

    # Color each model consistently across panels using the shared scatter-grid
    # palette.
    model_color = {m: _model_color(m) for m in model_order}
    x = np.arange(len(model_order))

    exact_rca = Q_COLS["RCA Accuracy"]
    halluc = Q_COLS["% Complete Hallucination"]
    # The ↑/↓ in each title states the polarity (higher / lower is better).
    panels: list[tuple[str, pd.DataFrame, str]] = [
        ("RCA Accuracy (Medium) ↑", df[df["_granularity"] == "medium"], exact_rca),
        ("RCA Accuracy (Hard) ↑", df[df["_granularity"] == "hard"], exact_rca),
        ("Hallucination Rate ↓", df, halluc),
    ]

    # Asymmetric layout: panels 0 & 1 share a y-axis and butt up against
    # each other (small inner wspace); panel 2 (inverted y-axis) keeps the
    # normal gap so its yticklabels have room.
    fig = plt.figure(figsize=(ICLR_WIDTH, 1.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.25)
    inner_left = outer[0].subgridspec(1, 2, wspace=0.08)
    ax0 = fig.add_subplot(inner_left[0])
    ax1 = fig.add_subplot(inner_left[1], sharey=ax0)
    ax2 = fig.add_subplot(outer[1])
    axes = [ax0, ax1, ax2]

    for i, (ax, (title, sub_df, field)) in enumerate(zip(axes, panels)):
        means, ses, ns = _bar_stats(sub_df, field, model_order)
        ax.bar(
            x,
            means,
            width=0.7,
            yerr=ses,
            color=[model_color[m] for m in model_order],
            edgecolor="white",
            linewidth=0.6,
            capsize=2,
            error_kw={"linewidth": 0.8, "ecolor": "#444444"},
        )
        for xi, m, e, n in zip(x, means, ses, ns):
            if n == 0:
                continue
            ax.text(
                float(xi),
                m + e + 1.5,
                f"{m:.1f}",
                ha="center",
                va="bottom",
                fontsize=TICK_FONT_SIZE,
                color="#444444",
            )

        ax.set_title(title, fontsize=LABEL_FONT_SIZE)
        ax.set_xticks([])
        # All three panels share the [0, 50] scale. Panels 0 & 1 share the
        # same YAxis (sharey), so only the leftmost owns its ticks/labels;
        # the middle hides them. Panel 2 (red) has its own YAxis.
        if i == 0 or i == 2:
            ax.set_ylim(0, 50)
            ax.set_yticks([0, 25, 50])
            ax.set_yticklabels(["0%", "25%", "50%"])
            ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
        else:
            # Middle panel shares ylim/yticks/labels with axes[0] via the
            # shared YAxis. Do NOT call set_ylim/set_yticks/set_yticklabels
            # here — they would mutate the shared YAxis and overwrite the
            # custom "%" labels set on axes[0]. Just hide this panel's
            # tick marks and label rendering.
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_position(("outward", 0))
        ax.spines["left"].set_position(("outward", OUTWARD))

    # Single-row legend shared across all three subplots.
    legend_handles = [
        mpatches.Patch(
            facecolor=model_color[m],
            edgecolor="white",
            label=model_alias(m),
        )
        for m in model_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.07),
    )

    # tight_layout isn't compatible with the subgridspec; keep manual margins.
    # Small bottom margin (no x-tick labels) sits the legend just under the
    # bars; tight top/side margins minimize whitespace around the figure.
    fig.subplots_adjust(bottom=0.10, top=0.90, left=0.06, right=0.99)

    fig_path = args.output_dir / "figures" / "rca_and_hallucination.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight", dpi=300, pad_inches=0.01)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
