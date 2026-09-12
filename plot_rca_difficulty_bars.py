r"""Two-panel RCA-by-difficulty bar chart: RCA Accuracy (Exact) and RCA Depth.

Side-by-side grouped bar charts (x-groups = agent_model) for two metrics,
broken out by prompt difficulty. The user-facing ``_granularity`` values are
already ``easy``/``medium``/``hard`` (``utils.GRANULARITY_ALIASES`` rewrites
the upstream ``easy``/``hard``/``universal`` labels).

1. **RCA Accuracy (Exact) ↑** — ``rca_accuracy`` mean per
   difficulty (Easy/Medium/Hard) on incident tasks, plus a **Control** bar
   showing control-task accuracy (control-task ``rca_depth``; RCA Accuracy is 0
   on control tasks by construction, so the control accuracy is the meaningful
   number, matching the control "Partial RCA" column of ``tab:full-rca-control``).
2. **RCA Depth (Partial) ↑** — ``rca_depth`` (0–3) mean per difficulty
   (Easy/Medium/Hard) on incident tasks, scaled to a percentage of 3. No
   control bar (RCA Depth has no score for control tasks).

Bars are colored by model using the shared palette from
``plot_rca_and_hallucination.py`` / ``plot_human_vs_llm_scatter_grid.py`` so
model identity is consistent across the paper (solid fill, no hatch). Each
bar is labeled with its difficulty (Control/Easy/Medium/Hard) drawn inside
the bar, or just above it when the bar is too short to fit the text.

Examples::

    cd examples/otel-demo
    uv run python plot_rca_difficulty_bars.py -od out-0522 -e high

"""

import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
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

# Match the font of plot_rca_and_hallucination.py (Fira Code, monospace).
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["Fira Code"]

# Per-model palette shared with plot_rca_and_hallucination.py /
# plot_human_vs_llm_scatter_grid.py so model identity reads consistently across
# the paper. Keyed by a substring of the ``_agent_model`` display name (which
# varies, e.g. ``claude-4.6-sonnet`` vs ``claude-sonnet-4.6``), checked in order.
SCATTER_COLORS: dict[str, str] = {
    "opus": "#2166ac",
    "sonnet": "#92c5de",
    "gpt": "#4daf4a",
    "glm": "#e41a1c",
    "deepseek": "#984ea3",
}

# Difficulty axis: fixed left→right order within each model group; also the
# order of the labels drawn inside each bar. "Control" only appears in the
# RCA-Accuracy panel.
DIFF_ORDER: list[str] = ["control", "easy", "medium", "hard"]
DIFF_LABELS: dict[str, str] = {
    "control": "Control",
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}

# Font size for the difficulty labels drawn inside (or just above) each bar.
IN_BAR_FONTSIZE = 4.5


def _model_color(model: str) -> str:
    """Return the scatter-grid palette color for ``model`` (grey fallback)."""
    name = model.lower()
    for key, color in SCATTER_COLORS.items():
        if key in name:
            return color
    return "#777777"


def _luminance(color: str) -> float:
    """Perceived luminance (0–1) of ``color``; used to pick black/white text."""
    r, g, b = to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _bar_stats(
    sub_df: pd.DataFrame,
    field: str,
    model_order: list[str],
    scale: float = 100.0,
) -> tuple[list[float], list[float], list[int]]:
    """Return (means, ses, n) per model in ``model_order``, scaled by ``scale``.

    Drops NaN values within each model slice before computing the mean and
    standard error; missing models render as bar height 0 with N=0. ``scale``
    is 100 for boolean rate fields and 100/3 for the 0–3 ``score`` field.
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
        means.append(float(vals.mean()) * scale)
        ses.append(float(sem(vals)) * scale if vals.size > 1 else 0.0)
        ns.append(int(vals.size))
    return means, ses, ns


def _plot_panel(
    ax: plt.Axes,
    title: str,
    cells: list[tuple[str, list[float], list[float], list[int]]],
    model_order: list[str],
    show_ylabels: bool = True,
) -> None:
    """Draw one grouped-bar panel.

    ``cells`` is an ordered list of ``(difficulty, means, ses, ns)`` tuples
    (one per difficulty present in this panel); each ``means``/``ses``/``ns``
    is per model in ``model_order``. When ``show_ylabels`` is False the panel
    shares its YAxis with the owning panel, so it neither sets the ylim/ticks
    (those would mutate the shared axis) nor renders its own y-tick labels.
    """
    x = np.arange(len(model_order))
    n_diff = len(cells)
    bar_width = 0.8 / n_diff

    # Data-units of bar height needed to fit a rotated (vertical) label inside
    # a bar, derived from the panel's height in points (ylim spans 0–100).
    # Requires the layout to be set (subplots_adjust called) beforehand.
    pos = ax.get_position()
    data_per_pt = 100.0 / (pos.height * ax.figure.get_figheight() * 72.0)

    for i, (diff, means, ses, ns) in enumerate(cells):
        offset = (i - (n_diff - 1) / 2) * bar_width
        for xi, model, mean_v, se_v in zip(x, model_order, means, ses):
            color = _model_color(model)
            ax.bar(
                xi + offset,
                mean_v,
                width=bar_width,
                yerr=se_v,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                capsize=1.5,
                error_kw={"linewidth": 0.7, "ecolor": "#444444"},
            )
            # Difficulty label inside the bar (rotated); on top when too short.
            # Rotated monospace text needs ~0.6*fontsize points per character.
            label = DIFF_LABELS[diff]
            need = len(label) * 0.6 * IN_BAR_FONTSIZE * data_per_pt * 1.15
            if mean_v >= need:
                txt_color = "white" if _luminance(color) < 0.6 else "#222222"
                ax.text(
                    xi + offset,
                    mean_v - 1.5,
                    label,
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=IN_BAR_FONTSIZE,
                    color=txt_color,
                )
            else:
                ax.text(
                    xi + offset,
                    mean_v + se_v + 1.5,
                    label,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=IN_BAR_FONTSIZE,
                    color="#333333",
                )

    ax.set_title(title, fontsize=LABEL_FONT_SIZE, pad=3)
    if show_ylabels:
        # 0–100 so the tall control-accuracy bars (~80%+) aren't clipped.
        # Only the y-owning panel sets these; the shared panel inherits them
        # (calling set_ylim/set_yticks there would mutate the shared YAxis).
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0%", "50%", "100%"])
        ax.set_yticks([25, 75], minor=True)
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
        ax.tick_params(axis="y", which="minor", length=0)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.yaxis.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.yaxis.grid(True, which="minor", linestyle=":", alpha=0.35, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 0))
    ax.spines["left"].set_position(("outward", OUTWARD))
    # Model identity is carried by the bottom color+hatch legend, so the
    # x-axis needs no tick labels (matching plot_rca_and_hallucination.py).
    ax.set_xticks([])


def main() -> None:
    """Render the two-panel RCA-by-difficulty bar chart."""
    parser = get_base_parser()
    parser.description = (
        "Two-panel bar chart: RCA Accuracy and RCA Depth "
        "by prompt difficulty, one model group per agent."
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    df = load_data(args.jobs_dir, args.effort)
    incident_df = df[~df["_is_control"]]
    control_df = df[df["_is_control"]]
    if incident_df.empty:
        logger.error("No incident trials found.")
        return

    # Preserve the MODEL_ORDER baked into the Categorical _agent_model.
    model_order: list[str] = [
        str(m)
        for m in df["_agent_model"].cat.categories
        if m in df["_agent_model"].values
    ]

    exact_rca = Q_COLS["RCA Accuracy"]
    incident_grans = ["easy", "medium", "hard"]

    # Panel 0 — RCA Accuracy: a Control bar (control-task score, the meaningful
    # control accuracy) followed by per-difficulty RCA Accuracy on incident tasks.
    exact_cells: list[tuple[str, list[float], list[float], list[int]]] = []
    for gran in incident_grans:
        sub = incident_df[incident_df["_granularity"] == gran]
        exact_cells.append(
            (gran, *_bar_stats(sub, exact_rca, model_order, scale=100.0))
        )

    # Panel 1 — RCA Depth (partial): per-difficulty mean score on incident
    # tasks only (no control bar).
    depth_cells: list[tuple[str, list[float], list[float], list[int]]] = [
        (
            "control",
            *_bar_stats(control_df, "rca_depth", model_order, scale=100.0 / 3.0),
        )
    ]
    for gran in incident_grans:
        sub = incident_df[incident_df["_granularity"] == gran]
        depth_cells.append(
            (gran, *_bar_stats(sub, "rca_depth", model_order, scale=100.0 / 3.0))
        )

    fig, axes = plt.subplots(1, 2, figsize=(ICLR_WIDTH, 1.5), sharey=True)
    # Set the layout before drawing: _plot_panel reads the axes geometry to
    # decide whether each difficulty label fits inside its bar.
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.10, wspace=0.08)
    _plot_panel(axes[0], "RCA Accuracy ↑", exact_cells, model_order)
    _plot_panel(axes[1], "RCA Depth ↑", depth_cells, model_order, show_ylabels=False)

    # Bottom legend — models by solid color (difficulty is labeled in-bar).
    model_handles = [
        mpatches.Patch(
            facecolor=_model_color(m), edgecolor="white", label=model_alias(m)
        )
        for m in model_order
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        ncol=len(model_handles),
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.08),
    )

    fig_path = args.output_dir / "figures" / "rca_difficulty_bars.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight", dpi=300, pad_inches=0.01)
    plt.close(fig)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
