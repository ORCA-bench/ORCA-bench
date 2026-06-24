r"""Inter-rater agreement between human annotators and the LLM judge.

(a) Joins two ``human_verified_scores*.json`` files on ``(task_name, model)``
and reports Spearman rho and quadratic-weighted Cohen's kappa, overall and
broken out by difficulty (Easy/Medium/Hard, derived from
``sampled_tasks.json``).

(b) Averages the two human scores per ``(task_name, model)`` and compares
against the LLM judge score loaded via :func:`utils.load_data`, with the
same overall/per-difficulty breakdown.

Examples::

    cd examples/otel-demo
    uv run python compute_human_agreement.py \
        --scores-a human_verified_scores.json \
        --scores-b human_verified_scores_ag.json \
        -od out-0522 -e high

"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from tabulate import tabulate

from utils import get_base_parser, load_data, setup_logging

logger = logging.getLogger(__name__)

DIFF_ORDER = ["easy", "medium", "hard"]


def load_scores(path: Path) -> dict[tuple[str, str], float]:
    """Return ``{(task_name, model): human_verified_score}``, dropping nulls."""
    with path.open() as f:
        entries = json.load(f)
    return {
        (e["task_name"], e["model"]): e["human_verified_score"]
        for e in entries
        if e["human_verified_score"] is not None
    }


def agreement_stats(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Spearman rho and quadratic-weighted Cohen's kappa between two score columns."""
    a = df[col_a].to_numpy(dtype=float)
    b = df[col_b].to_numpy(dtype=float)
    if len(df) < 3:
        return {"n": len(df), "spearman": None, "kappa_w": None}
    corr, _ = spearmanr(a, b)
    kappa = cohen_kappa_score(
        np.round(a).astype(int), np.round(b).astype(int), weights="quadratic"
    )
    return {"n": len(df), "spearman": corr, "kappa_w": kappa}


def agreement_table(df: pd.DataFrame, col_a: str, col_b: str) -> list[dict]:
    """Overall + per-difficulty agreement table between two score columns."""
    table = [{"difficulty": "all", **agreement_stats(df, col_a, col_b)}]
    for diff in DIFF_ORDER:
        subset = df[df["difficulty"] == diff]
        table.append({"difficulty": diff, **agreement_stats(subset, col_a, col_b)})
    return table


def print_table(title: str, table: list[dict]) -> None:
    """Print a table title followed by its github-flavored markdown rendering."""
    print(title)
    print(
        tabulate(
            table,
            headers={
                "difficulty": "Difficulty",
                "n": "N",
                "spearman": "Spearman ρ",
                "kappa_w": "Weighted κ",
            },
            tablefmt="github",
            floatfmt=".3f",
        )
    )
    print()


def main() -> None:
    """Entry point."""
    parser = get_base_parser()
    parser.add_argument(
        "--scores-a", type=Path, default=Path("human_verified_scores.json")
    )
    parser.add_argument(
        "--scores-b", type=Path, default=Path("human_verified_scores_ag.json")
    )
    parser.add_argument(
        "--sampled-tasks",
        type=Path,
        default=Path("human-eval-sample/output/sampled_tasks.json"),
        help="Path to sampled_tasks.json (used for difficulty lookup)",
    )
    args = parser.parse_args()
    setup_logging()

    scores_a = load_scores(args.scores_a)
    scores_b = load_scores(args.scores_b)

    with args.sampled_tasks.open() as f:
        sampled: dict[str, dict] = json.load(f)
    granularity_aliases = {"easy": "easy", "hard": "medium", "universal": "hard"}
    task_difficulty = {
        t: granularity_aliases.get(v["difficulty"], v["difficulty"])
        for t, v in sampled.items()
    }

    common_keys = set(scores_a) & set(scores_b)

    # (a) pairwise inter-annotator agreement
    pairwise = pd.DataFrame(
        [
            {
                "task_name": task_name,
                "model": model,
                "difficulty": task_difficulty.get(task_name, "?"),
                "score_a": scores_a[(task_name, model)],
                "score_b": scores_b[(task_name, model)],
            }
            for task_name, model in common_keys
        ]
    )
    logger.info(
        f"(a) Matched {len(pairwise)} (task_name, model) pairs "
        f"({len(scores_a)} in {args.scores_a.name}, {len(scores_b)} in {args.scores_b.name})"
    )
    print_table(
        "(a) Pairwise inter-annotator agreement",
        agreement_table(pairwise, "score_a", "score_b"),
    )

    # (b) average human score vs. LLM judge
    avg_lookup = {key: (scores_a[key] + scores_b[key]) / 2 for key in common_keys}
    df = load_data(args.output_dir, args.effort)
    judge_rows = []
    for _, r in df.iterrows():
        key = (r["task_name"], str(r["_agent_model"]))
        h = avg_lookup.get(key)
        if h is not None:
            judge_rows.append(
                {
                    "difficulty": r["_granularity"],
                    "human_avg": h,
                    "llm_score": r["score"],
                }
            )
    judge_df = pd.DataFrame(judge_rows)
    logger.info(
        f"(b) Matched {len(judge_df)} (task_name, model) pairs against LLM judge scores"
    )
    print_table(
        "(b) LLM judge vs. average human score",
        agreement_table(judge_df, "human_avg", "llm_score"),
    )


if __name__ == "__main__":
    main()
