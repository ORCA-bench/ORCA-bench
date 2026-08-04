r"""Compute average accuracy stratified by task granularity.

Reads each trial's ``verifier/reward-details.json`` under ``--jobs-dir`` and the
``granularity`` value from the published task's ``task.toml`` ``[metadata]``.
Prints one markdown table per ``(agent, agent_model, effort, judge_model)``
group with one row per granularity value. ``--output-dir`` is where the CSV and
figures are written.

Examples::

    cd examples/otel-demo
    uv run python format_acc_by_granularity.py -jd jobs/2026-05-05__04-10-26 -od out-0416
    uv run python format_acc_by_granularity.py -jd jobs -od out-0416 --effort medium

"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from harbor_utils.plotting_utils import (
    ICLR_WIDTH,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    OUTWARD,
    TICK_FONT_SIZE,
)
from utils import (
    Q_COLS,
    fmt_acc,
    fmt_mean_score,
    fmt_mean_score_pct,
    get_base_parser,
    load_data,
    model_alias,
    model_display,
    setup_logging,
)

logger = logging.getLogger(__name__)

GROUP_COLS = ["_agent", "_agent_model", "_effort", "_display_model"]

GRANULARITY_ORDER = ["easy", "medium", "hard"]

# easy → medium → hard, light → dark (sequential green palette).
GRANULARITY_COLORS = {
    "easy": "#a1d99b",
    "medium": "#41ab5d",
    "hard": "#005a32",
}

# Hatch style supplements color so bars stay distinguishable in greyscale
# prints or for color-vision-deficient readers. ``hard`` is left solid since
# the dark green hides any hatch pattern anyway.
GRANULARITY_HATCHES = {
    "easy": "...",
    "medium": "////",
    "hard": "",
}


def _granularity_key(value: str) -> tuple[int, str]:
    """Sort known granularities easy→hard; unknowns trail, alphabetized."""
    if value in GRANULARITY_ORDER:
        return (GRANULARITY_ORDER.index(value), value)
    return (len(GRANULARITY_ORDER), value)


def _plot_grouped_bars(
    section_df: pd.DataFrame,
    fig_path: Path,
    *,
    field: str,
    scale: float,
    ylabel: str,
) -> None:
    """Save a grouped bar plot: x-groups = model, bars within = granularity.

    Per-trial values are ``section_df[field] * scale`` (mean ± SE across
    trials in each model × granularity cell).
    """
    # ``load_data`` already orders rows by MODEL_ORDER (via the ordered
    # Categorical on ``_agent_model``); drop_duplicates preserves first-seen
    # order so model_keys come out in the canonical sort.
    model_keys = [
        tuple(row)
        for row in section_df[GROUP_COLS].drop_duplicates().itertuples(index=False)
    ]
    if not model_keys:
        return
    granularities = sorted(
        section_df["_granularity"].dropna().unique(), key=_granularity_key
    )
    if not granularities:
        return

    n_models = len(model_keys)
    n_grans = len(granularities)
    x = np.arange(n_models)
    bar_width = 0.8 / n_grans

    fig, ax = plt.subplots(figsize=(ICLR_WIDTH, 2.5))
    for i, gran in enumerate(granularities):
        means, errs = [], []
        for key in model_keys:
            mask = np.logical_and.reduce(
                [section_df[col] == val for col, val in zip(GROUP_COLS, key)]
            )
            g = section_df[mask & (section_df["_granularity"] == gran)]
            vals = (
                np.array(g[field].dropna().astype(float).tolist(), dtype=float) * scale
                if field in g.columns
                else np.array([], dtype=float)
            )
            if len(vals) == 0:
                means.append(np.nan)
                errs.append(0.0)
            else:
                means.append(float(np.mean(vals)))
                errs.append(
                    float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                    if len(vals) > 1
                    else 0.0
                )
        offset = (i - (n_grans - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=errs,
            color=GRANULARITY_COLORS.get(gran, f"C{i}"),
            hatch=GRANULARITY_HATCHES.get(gran, ""),
            label=gran.capitalize(),
            edgecolor="white",
            linewidth=0.5,
            error_kw={"linewidth": 0.7, "capsize": 1.5},
        )
        for xi, m, e in zip(x + offset, means, errs):
            if np.isnan(m):
                continue
            ax.text(
                xi,
                m + e + 1.5,
                f"{m:.1f}",
                ha="center",
                va="bottom",
                fontsize=max(TICK_FONT_SIZE - 1.5, 5),
            )

    tick_labels = [model_alias(agent_model) for _, agent_model, _, _ in model_keys]
    ax.set_xticks(x, labels=tick_labels)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_position(("outward", OUTWARD))
    ax.spines["left"].set_position(("outward", OUTWARD))
    ax.legend(
        title="Difficulty",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
    )

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {fig_path}")


def main() -> None:
    """Load score files and print average accuracy stratified by granularity."""
    parser = get_base_parser()
    parser.description = "Compute average accuracy stratified by granularity."
    args = parser.parse_args()
    setup_logging(args.log_level)

    df = load_data(args.jobs_dir, args.effort, args.filter_xlsx)

    missing_mask = df["_granularity"].isna()
    if missing_mask.any():
        missing = df.loc[missing_mask, "_trial_id"].tolist()
        logger.error(
            f"{len(missing)} trial(s) missing granularity metadata: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
        sys.exit(1)

    fig_dir = args.output_dir / "figures"
    all_rows = []
    for is_control, section_title in [
        (False, "Incident Tasks"),
        (True, "Control Tasks"),
    ]:
        section_df = df[df["_is_control"] == is_control]
        if section_df.empty:
            continue
        slug = "control" if is_control else "incident"
        _plot_grouped_bars(
            section_df,
            fig_dir / f"accuracy_by_granularity_{slug}.pdf",
            field=Q_COLS["RCA Accuracy"],
            scale=100.0,
            ylabel="RCA Accuracy (%)",
        )
        _plot_grouped_bars(
            section_df,
            fig_dir / f"partial_accuracy_by_granularity_{slug}.pdf",
            field="rca_depth",
            scale=100.0 / 3.0,
            ylabel="RCA Depth (% of 3)",
        )
        print(f"# {section_title}\n")
        for group_key, group_df in section_df.sort_values(GROUP_COLS).groupby(
            GROUP_COLS, observed=True
        ):
            agent, agent_model, effort, judge_model = group_key
            effort_str = f" ({effort})" if effort else ""
            label = (
                f"{agent} / {model_display(agent_model)}{effort_str} "
                f"(judge: {judge_model})"
            )
            print(f"## Average Accuracy by Granularity — {label}\n")
            table_data = []
            granularities = sorted(
                group_df["_granularity"].unique(), key=_granularity_key
            )
            for gran in granularities:
                g = group_df[group_df["_granularity"] == gran]
                row = {
                    "Granularity": gran,
                    "N": len(g),
                    "Avg GT Events": fmt_mean_score(g["_n_events"].tolist()),
                    "RCA Depth % (out of 3)": fmt_mean_score_pct(
                        g["rca_depth"].tolist()
                    ),
                    "% Score 0": fmt_acc([s == 0 for s in g["rca_depth"].tolist()]),
                }
                for qlabel, qfield in Q_COLS.items():
                    vals = (
                        g[qfield].where(g[qfield].notna(), None).tolist()
                        if qfield in g.columns
                        else []
                    )
                    row[qlabel] = fmt_acc(vals)
                table_data.append(row)
                all_rows.append(
                    {
                        "Task Type": section_title,
                        "Agent": agent,
                        "Agent Model": model_display(agent_model),
                        "Effort": effort or "-",
                        "Judge Model": judge_model,
                        **row,
                    }
                )
            print(tabulate(table_data, headers="keys", tablefmt="github"))
            print()

    csv_path = args.output_dir / "accuracy_by_granularity.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
