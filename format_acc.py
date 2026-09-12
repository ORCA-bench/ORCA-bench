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

Reads each trial's ``verifier/reward-details.json`` under ``--jobs-dir`` and
prints a markdown table with one row per judge model showing mean Score and
Q1-Q6 accuracy. ``--output-dir`` is where the CSVs are written.

Examples::

    cd examples/otel-demo
    uv run python format_acc.py -jd jobs/2026-05-05__04-10-26 -od out-0318
    uv run python format_acc.py -jd jobs/2026-05-05__04-10-26 -od out-0318 --effort high

"""

# %%
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
    jobs_dir = args.jobs_dir
    effort = args.effort
else:
    output_dir = Path("out-0423-2")
    jobs_dir = Path("jobs")
    effort = "high"

# %%
# Scores come from each trial's verifier/reward-details.json; output_dir is
# only the write location for the CSVs below. ``_report_path`` is set by
# load_trials, so no separate join is needed to reach the agent's report.
df = load_data(jobs_dir, effort)

# %%
df.columns

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
][
    ["task_name", "flag", "rca_depth", "_report_path", "_score_path"]
    + list(Q_COLS.values())
]


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
        "RCA Depth % (out of 3)": fmt_mean_score_pct(g["rca_depth"].tolist()),
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
csv_path = output_dir / "accuracy.csv"
output_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(table_data).to_csv(csv_path, index=False)
logger.info(f"Saved CSV to {csv_path}")
