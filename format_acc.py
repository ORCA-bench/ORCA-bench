# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
r"""Compute average accuracy across Score and Q1-Q6 columns per model.

Reads score JSON files from ``<output_dir>/scores/`` and prints a markdown
table with one row per judge model showing mean Score and Q1-Q6 accuracy.

Examples::

    cd examples/otel-demo
    uv run python format_acc.py -od out-0318
    uv run python format_acc.py -od out-0318 --effort high

"""

# %%
import json
import logging
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from utils import (
    Q_COLS,
    fmt_acc,
    fmt_mean_score_pct,
    get_base_parser,
    load_data,
    setup_logging,
)

pd.set_option("display.max_colwidth", None)

# %%
logger = logging.getLogger(__name__)

# %%
GROUP_COLS = ["_agent", "_agent_model", "_effort", "_display_model"]
# Within each (agent, model, effort, judge) group, we split rows into control
# vs incident tasks. Granular per-section matches only apply to incident tasks
# (control tasks have no rubric); for control rows those columns render as "—".
TASK_TYPE_COL = "_is_control"

# %%
# Load score files and print average accuracy per model.
if True:
    parser = get_base_parser()
    parser.description = (
        "Compute average accuracy across Score and per-section matches per model."
    )
    args = parser.parse_args()
    setup_logging(args.log_level)
    output_dir = args.output_dir
    effort = args.effort
    filter_xlsx = args.filter_xlsx
else:
    output_dir = Path("out-0423-2")
    effort = "high"
    filter_xlsx = None

# %%
df = load_data(output_dir, effort, filter_xlsx)

# %%
df.columns

# %%
with (output_dir / "all-predictions.json").open() as f:
    preds = json.load(f)
trial_df = pd.DataFrame([p["result"] for p in preds.values()])[
    ["trial_name", "trial_uri"]
]
df = df.merge(trial_df, left_on="trial_id", right_on="trial_name", how="left")
df["_report_path"] = df["trial_uri"].str.removeprefix("file://") + "/verifier/report.md"

# %%
df["_agent_model"].unique()

# %%
sonnet_df = df[(df["_agent_model"] == "claude-4.6-sonnet")]

# %%
sonnet_df

# %%
sonnet_df[
    (
        sonnet_df["metrics_any_match"]
        | sonnet_df["logs_any_match"]
        | sonnet_df["traces_any_match"]
    )
    & (sonnet_df["flag"] == "recommendationCacheFailure")
][["task_name", "flag", "score", "_report_path", "_score_path"] + list(Q_COLS.values())]


# %%
def agg_group(g: pd.DataFrame, task_type: str) -> dict:
    """Aggregate one (agent, model, effort, judge, task_type) slice into a row."""
    row = {
        "Agent": g["_agent"].iloc[0],
        "Agent Model": g["_agent_model"].iloc[0],
        "Effort": g["_effort"].iloc[0] or "-",
        "Judge Model": g["_display_model"].iloc[0],
        "Task Type": task_type,
        "N": len(g),
        "RCA Depth % (out of 3)": fmt_mean_score_pct(g["score"].tolist()),
    }
    for label, qfield in Q_COLS.items():
        vals = g[qfield].where(g[qfield].notna(), None).tolist()
        row[label] = fmt_acc(vals)
    return row


# %%
table_data: list[dict] = []
for _, g in df.sort_values(GROUP_COLS).groupby(GROUP_COLS, observed=True):
    incident_g = g[~g[TASK_TYPE_COL]]
    control_g = g[g[TASK_TYPE_COL]]
    if len(incident_g):
        table_data.append(agg_group(incident_g, "incident"))
    if len(control_g):
        table_data.append(agg_group(control_g, "control"))
print("## Average Accuracy\n")
print(tabulate(table_data, headers="keys", tablefmt="github"))
print()

# %%
csv_path = args.output_dir / "accuracy.csv"
pd.DataFrame(table_data).to_csv(csv_path, index=False)
logger.info(f"Saved CSV to {csv_path}")


# %%
def dist_group(g: pd.DataFrame, task_type: str) -> dict:
    """Tally score distribution (0–3) for one group.

    Buckets by ``best_score`` (max across rubrics) because the per-trial
    ``score`` is a float mean and would silently miss integer buckets on
    multi-rubric tasks. ``best_score`` mirrors the historical pre-per-rubric
    "overall score" semantic.
    """
    counts = g["best_score"].value_counts()
    n = len(g)
    row = {
        "Agent": g["_agent"].iloc[0],
        "Agent Model": g["_agent_model"].iloc[0],
        "Effort": g["_effort"].iloc[0] or "-",
        "Judge Model": g["_display_model"].iloc[0],
        "Task Type": task_type,
        "N": n,
    }
    for s in (0, 1, 2, 3):
        c = int(counts.get(s, 0))
        row[f"Score {s}"] = f"{c} ({c / n * 100:.1f}%)"
    return row


# %%
dist_data: list[dict] = []
for _, g in df.sort_values(GROUP_COLS).groupby(GROUP_COLS, observed=True):
    incident_g = g[~g[TASK_TYPE_COL]]
    control_g = g[g[TASK_TYPE_COL]]
    if len(incident_g):
        dist_data.append(dist_group(incident_g, "incident"))
    if len(control_g):
        dist_data.append(dist_group(control_g, "control"))
print("## Score Distribution\n")
print(
    "> Buckets are sourced from `best_score` (max across rubrics) so multi-rubric "
    "trials with fractional `score` means don't silently miss integer buckets.\n"
)
print(tabulate(dist_data, headers="keys", tablefmt="github"))
print()

# %%
dist_csv_path = args.output_dir / "score_distribution.csv"
pd.DataFrame(dist_data).to_csv(dist_csv_path, index=False)
logger.info(f"Saved CSV to {dist_csv_path}")
