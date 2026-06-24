r"""Scatter plot: human-verified scores vs. LLM judge scores, 5×3 grid.

Rows = difficulty (Easy / Medium / Hard), columns = model (5 models).
Each panel shows one point per (task, model) pair with Spearman ρ and
weighted Cohen κ_w annotations.

Human scores are read from ``human_verified_scores_ag.json``; LLM judge
scores are loaded via :func:`utils.load_data`.

Examples::

    cd examples/otel-demo
    uv run python plot_human_vs_llm_scatter.py -od out-0522 -e high

"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from utils import get_base_parser, load_data, setup_logging

logger = logging.getLogger(__name__)

ICLR_WIDTH = 5.5

MODEL_ORDER = [
    ("claude-opus-4.7", "Opus 4.7", "#2166ac"),
    ("claude-4.6-sonnet", "Sonnet 4.6", "#92c5de"),
    ("gpt-5.5", "GPT-5.5", "#4daf4a"),
    ("glm-5", "GLM-5", "#e41a1c"),
    ("deepseek-v4-pro", "DeepSeek-V4-Pro", "#984ea3"),
]

DIFF_ORDER = [
    ("easy", "Easy"),
    ("medium", "Medium"),
    ("hard", "Hard"),
]


def main() -> None:
    """Entry point."""
    parser = get_base_parser()
    parser.add_argument(
        "--human-scores",
        type=Path,
        default=Path("human_verified_scores_ag.json"),
        help="Path to human_verified_scores_ag.json",
    )
    parser.add_argument(
        "--sampled-tasks",
        type=Path,
        default=Path("human-eval-sample/output/sampled_tasks.json"),
        help="Path to sampled_tasks.json (used for expected task set only)",
    )
    args = parser.parse_args()
    setup_logging()

    # --- human scores --------------------------------------------------
    with args.human_scores.open() as f:
        human_entries = json.load(f)
    human_lookup: dict[tuple[str, str], float] = {
        (e["task_name"], e["model"]): e["human_verified_score"]
        for e in human_entries
        if e["human_verified_score"] is not None
    }

    # --- LLM judge scores (already filtered to sampled tasks) ----------
    df = load_data(args.output_dir, args.effort)
    # columns used: task_name, _agent_model, score, _granularity

    # --- merge ---------------------------------------------------------
    rows = []
    for _, r in df.iterrows():
        key = (r["task_name"], str(r["_agent_model"]))
        h = human_lookup.get(key)
        if h is not None:
            rows.append(
                {
                    "task_name": r["task_name"],
                    "model": str(r["_agent_model"]),
                    "difficulty": r["_granularity"],
                    "human_score": h,
                    "llm_score": r["score"],
                }
            )
    merged = pd.DataFrame(rows)

    # --- sanity check --------------------------------------------------
    counts = merged.groupby("model").size()
    print(counts.to_string())
    # Expect ~40 per model; deepseek-v4-pro may be lower (empty reports excluded)

    # --- plot ----------------------------------------------------------
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = ["Fira Code"]

    point_size = 20

    fig, axes = plt.subplots(
        3, 5, figsize=(ICLR_WIDTH, 3.0), dpi=150, sharex=True, sharey=True
    )

    for col, (model_key, model_label, color) in enumerate(MODEL_ORDER):
        for row, (diff_key, diff_label) in enumerate(DIFF_ORDER):
            ax = axes[row, col]

            subset = merged[
                (merged["model"] == model_key) & (merged["difficulty"] == diff_key)
            ]

            if len(subset) > 0:
                human = subset["human_score"].to_numpy(dtype=float)
                llm = subset["llm_score"].to_numpy(dtype=float)

                ax.scatter(
                    human,
                    llm,
                    s=point_size,
                    color=color,
                    alpha=0.7,
                    edgecolors="white",
                    linewidths=0.3,
                    zorder=3,
                )

                ax.plot(
                    [0, 3],
                    [0, 3],
                    ls="--",
                    color="grey",
                    alpha=0.4,
                    linewidth=0.5,
                    zorder=1,
                )

                if len(subset) >= 3:
                    corr, _ = spearmanr(human, llm)
                    kappa = cohen_kappa_score(
                        np.round(human).astype(int),
                        np.round(llm).astype(int),
                        weights="quadratic",
                    )
                    ax.text(
                        0.05,
                        0.95,
                        f"$\\rho$={corr:.2f}\n$\\kappa_w$={kappa:.2f}",
                        transform=ax.transAxes,
                        fontsize=5,
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            edgecolor="none",
                        ),
                    )

                ax.text(
                    0.95,
                    0.05,
                    f"n={len(subset)}",
                    transform=ax.transAxes,
                    fontsize=5,
                    ha="right",
                    va="bottom",
                    color="#555555",
                )

            ax.set_xlim(-0.2, 3.2)
            ax.set_ylim(-0.2, 3.2)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_yticks([0, 1, 2, 3])
            ax.tick_params(labelsize=4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                ax.set_title(model_label, fontsize=6, color=color, pad=4)

            if col == 0:
                ax.set_ylabel(diff_label, fontsize=7)
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")

    fig.text(
        0.02,
        0.5,
        "LLM judge score",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=7,
    )
    fig.text(0.5, 0.0, "Human score", va="center", ha="center", fontsize=7)

    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    fig.subplots_adjust(left=0.12)

    scores_stem = args.human_scores.stem  # e.g. "human_verified_scores_ag"
    out_path = args.output_dir / "figures" / f"human_vs_llm_scatter_{scores_stem}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
