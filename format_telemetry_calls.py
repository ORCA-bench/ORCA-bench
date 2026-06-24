"""Per-agent breakdown of telemetry-call labels (success/empty/error).

Reads ``output_dir/telemetry_classifications/{trial_id}/*.json`` per-call
files emitted by ``classify_telemetry_calls.py``, joins them with the
agent-group label derived from each batch's ``config.json`` under
``jobs_dir``, and prints two markdown tables to stdout: an overall
per-agent summary and a per-agent × telemetry-type breakdown
(metrics / logs / traces).

Examples::

    cd examples/otel-demo
    uv run python format_telemetry_calls.py -od out-test-0514 -jd jobs-sub
"""

import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from harbor_utils.plotting_utils import (
    ICLR_HEIGHT,
    ICLR_WIDTH,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    OUTWARD,
    TICK_FONT_SIZE,
)
from utils import (
    MODEL_ALIASES,
    format_group_label,
    get_base_parser,
    load_invalid_trial_ids,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Override for the half-width ICLR figures in this script (and the
# redundant-calls script that imports _plot_stacked) — smaller than the
# shared TICK_FONT_SIZE so the 5 model names fit without overlap.
TICK_FONT_SIZE_LOCAL = 5.5

LABELS = ("success", "empty", "error")
LABEL_COLORS = {
    "success": "#2ca02c",  # green
    "empty": "#7f7f7f",  # grey
    "error": "#d62728",  # red
}

# Canonical left-to-right order for tables and stacked-bar plots: frontier
# closed-source models first (Opus, Sonnet, GPT), then open-weight (GLM,
# DeepSeek). Keys are :data:`utils.MODEL_ALIASES` values.
MODEL_ORDER = (
    MODEL_ALIASES["claude-opus-4.7"],
    MODEL_ALIASES["claude-4.6-sonnet"],
    MODEL_ALIASES["gpt-5.5"],
    MODEL_ALIASES["glm-5"],
    MODEL_ALIASES["deepseek-v4-pro"],
)


def _group_sort_key(group: str) -> tuple[int, str]:
    """Sort key that orders groups by ``MODEL_ORDER`` (unknowns appended)."""
    label = _short_group_label(group)
    try:
        return (MODEL_ORDER.index(label), label)
    except ValueError:
        return (len(MODEL_ORDER), label)


# Telemetry-type buckets used for stratified breakdowns. Categories not listed
# here (e.g. telemetry:grafana_health, telemetry:grafana_dashboard) appear in
# the per-agent summary but are excluded from the per-type table.
TELEM_TYPES: dict[str, set[str]] = {
    "metrics": {"telemetry:metrics"},
    "logs": {"telemetry:logs", "telemetry:docker_logs"},
    "traces": {"telemetry:spans"},
}


def _short_group_label(group: str) -> str:
    """Return a display alias for the model in a ``terminus-N (model)[ (effort)]`` label.

    All trials currently use the terminus-2 harness, so the agent prefix is
    redundant noise; the trailing ``(effort)`` is also dropped. The extracted
    model name is then mapped through :data:`utils.MODEL_ALIASES` for compact,
    distinct display names (Sonnet 4.6, Opus 4.7, DeepSeek-V4-Pro, GLM-5,
    GPT-5.5). Unknown models fall through with their raw name.
    """
    m = re.match(r"^terminus-\d+ \((.+?)\)", group)
    model = m.group(1) if m else group
    return MODEL_ALIASES.get(model, model)


def classify_type(categories: list[str]) -> str | None:
    """Return the telemetry-type bucket for a row's categories, or None."""
    for ttype, cats in TELEM_TYPES.items():
        if any(c in cats for c in categories):
            return ttype
    return None


def _plot_stacked(
    groups: list[str],
    counts_by_group: dict[str, Counter[str]],
    mode: str,
    fig_path: Path,
    per_trial_calls_by_group: dict[str, list[int]] | None = None,
    labels: tuple[str, ...] = LABELS,
    label_colors: dict[str, str] = LABEL_COLORS,
) -> None:
    """Save a stacked bar chart of ``labels`` per agent.

    ``mode`` is ``"count"`` (mean per-trial counts when
    ``per_trial_calls_by_group`` is supplied; raw totals otherwise) or
    ``"rate"`` (% of telemetry calls). ``labels`` and ``label_colors``
    default to the success/empty/error scheme but can be overridden by
    sibling format scripts (e.g. unique/redundant).
    """
    n = len(groups)
    fig, ax = plt.subplots(figsize=(ICLR_WIDTH / 2, ICLR_HEIGHT))
    x = np.arange(n)
    bottoms = np.zeros(n)
    totals = np.array(
        [sum(counts_by_group[g][lab] for lab in labels) for g in groups],
        dtype=float,
    )
    n_trials = (
        np.array(
            [max(len(per_trial_calls_by_group.get(g, [])), 1) for g in groups],
            dtype=float,
        )
        if per_trial_calls_by_group is not None
        else np.ones(n)
    )
    for label in labels:
        raw = np.array([counts_by_group[g][label] for g in groups], dtype=float)
        if mode == "rate":
            vals = 100 * raw / np.maximum(totals, 1)
        else:  # count -> mean per trial
            vals = raw / n_trials
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            color=label_colors[label],
            label=label,
            edgecolor="white",
            linewidth=0.5,
            width=0.7,
        )
        if mode == "count":
            for i in range(n):
                if vals[i] <= 0 or totals[i] <= 0:
                    continue
                pct = 100 * raw[i] / totals[i]
                if pct < 4:  # skip tiny segments where text wouldn't fit
                    continue
                ax.text(
                    i,
                    bottoms[i] + vals[i] / 2,
                    f"{pct:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=max(TICK_FONT_SIZE - 1, 6),
                    color="white",
                )
        bottoms = bottoms + vals

    tick_labels = [_short_group_label(g) for g in groups]
    ax.set_xticks(x, labels=tick_labels)
    # tick_params is durable through tight_layout (set_xticks fontsize is not).
    # Match x-tick size to y-tick size for visual consistency.
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE_LOCAL)
    if mode == "rate":
        ax.set_ylabel("% of telemetry calls", fontsize=LABEL_FONT_SIZE)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    else:
        ax.set_ylabel("Mean telemetry calls", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE_LOCAL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", OUTWARD))
    ax.spines["left"].set_position(("outward", OUTWARD))
    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.18),
    )

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {fig_path}")
    plt.close(fig)


@dataclass
class TelemetryAggregation:
    """Per-group aggregates emitted by :func:`aggregate_telemetry_classifications`."""

    counts_by_group: dict[str, Counter[str]] = field(default_factory=dict)
    counts_by_group_type: dict[tuple[str, str], Counter[str]] = field(
        default_factory=dict
    )
    n_trials_by_group: Counter[str] = field(default_factory=Counter)
    per_trial_calls_by_group: dict[str, list[int]] = field(default_factory=dict)
    # One Counter per trial so callers can compute per-trial label fractions
    # (e.g. Empty% / Error% with per-model SEM) without re-walking the JSONs.
    per_trial_label_counts_by_group: dict[str, list[Counter[str]]] = field(
        default_factory=dict
    )
    n_classifier_errors_by_group: Counter[str] = field(default_factory=Counter)
    n_classifier_errors_by_group_type: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    n_unmapped: int = 0


def aggregate_telemetry_classifications(
    classifications_dir: Path,
    trial_to_group: dict[str, str],
    invalid_ids: set[str],
) -> TelemetryAggregation:
    """Walk ``classifications_dir`` and aggregate success/empty/error label counts per group.

    Reads ``{trial_id}/*.json`` records emitted by ``classify_telemetry_calls.py``
    and joins them against ``trial_to_group``. Returns the full set of
    aggregates so callers (this script's ``main`` and ``plot_efficiency.py``)
    can render whichever views they need without reopening the JSONs.
    """
    agg = TelemetryAggregation()
    for trial_dir in sorted(classifications_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        trial_id = trial_dir.name
        if trial_id in invalid_ids:
            continue
        group = trial_to_group.get(trial_id)
        if group is None:
            agg.n_unmapped += 1
            continue

        agg.n_trials_by_group[group] += 1
        c = agg.counts_by_group.setdefault(group, Counter())
        trial_calls = 0
        trial_label_counts: Counter[str] = Counter()
        for json_path in trial_dir.glob("*.json"):
            try:
                rec = json.loads(json_path.read_text())
            except Exception as exc:
                logger.warning(f"Could not parse {json_path}: {exc}")
                continue
            label = rec.get("label")
            ttype = classify_type(rec.get("categories", []) or [])
            if label in LABELS:
                c[label] += 1
                trial_calls += 1
                trial_label_counts[label] += 1
                if ttype is not None:
                    agg.counts_by_group_type.setdefault((group, ttype), Counter())[
                        label
                    ] += 1
            else:
                agg.n_classifier_errors_by_group[group] += 1
                if ttype is not None:
                    agg.n_classifier_errors_by_group_type[(group, ttype)] += 1
        agg.per_trial_calls_by_group.setdefault(group, []).append(trial_calls)
        agg.per_trial_label_counts_by_group.setdefault(group, []).append(
            trial_label_counts
        )
    return agg


def build_trial_to_group(jobs_dir: Path) -> dict[str, str]:
    """Return a ``{trial_id: agent_group_label}`` mapping by walking ``jobs_dir``."""
    jobs_dir = jobs_dir.resolve()
    if not jobs_dir.is_dir():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")

    trial_to_group: dict[str, str] = {}
    for batch_dir in sorted(jobs_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        agent_name, model_name = "unknown", "unknown"
        effort: str | None = None
        config_path = batch_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                if config.get("agents"):
                    agent_cfg = config["agents"][0]
                    agent_name = agent_cfg.get("name", "unknown")
                    model_name = agent_cfg.get("model_name", "unknown")
                    effort = (agent_cfg.get("kwargs") or {}).get("reasoning_effort")
            except Exception as exc:
                logger.warning(f"Could not parse {config_path}: {exc}")
        group = format_group_label(agent_name, model_name, effort)
        for trial_dir in batch_dir.iterdir():
            if trial_dir.is_dir():
                trial_to_group[trial_dir.name] = group
    return trial_to_group


def main() -> None:
    """Aggregate telemetry-call labels per agent group and print a markdown table."""
    parser = get_base_parser()
    parser.description = (
        "Per-agent breakdown of telemetry-call success/empty/error labels."
    )
    parser.add_argument(
        "--jobs-dir",
        "-jd",
        type=Path,
        default=Path("jobs"),
        help="Path to the Harbor jobs directory (default: jobs)",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    classifications_dir = args.output_dir / "telemetry_classifications"
    if not classifications_dir.is_dir():
        logger.error(f"No telemetry_classifications/ at {classifications_dir}")
        sys.exit(1)

    trial_to_group = build_trial_to_group(args.jobs_dir)
    invalid_ids: set[str] = (
        load_invalid_trial_ids(args.filter_xlsx) if args.filter_xlsx else set()
    )

    agg = aggregate_telemetry_classifications(
        classifications_dir, trial_to_group, invalid_ids
    )
    counts_by_group = agg.counts_by_group
    counts_by_group_type = agg.counts_by_group_type
    n_trials_by_group = agg.n_trials_by_group
    n_classifier_errors_by_group = agg.n_classifier_errors_by_group
    n_classifier_errors_by_group_type = agg.n_classifier_errors_by_group_type
    per_trial_calls_by_group = agg.per_trial_calls_by_group

    if agg.n_unmapped:
        logger.warning(
            f"{agg.n_unmapped} trial(s) had no batch mapping in {args.jobs_dir} (skipped)"
        )

    if not counts_by_group:
        logger.error("No classifications loaded; nothing to aggregate.")
        sys.exit(1)

    def fmt(n: int, total: int) -> str:
        return f"{n} ({100 * n / total:.1f}%)" if total else f"{n} (—)"

    rows = []
    for group in sorted(counts_by_group, key=_group_sort_key):
        c = counts_by_group[group]
        total = c["success"] + c["empty"] + c["error"]
        per_trial = per_trial_calls_by_group.get(group, [])
        if per_trial:
            mean = float(np.mean(per_trial))
            std = float(np.std(per_trial, ddof=1)) if len(per_trial) > 1 else 0.0
            avg_str = f"{mean:.1f} ± {std:.1f}"
        else:
            avg_str = "—"
        rows.append(
            [
                _short_group_label(group),
                n_trials_by_group[group],
                total,
                avg_str,
                fmt(c["success"], total),
                fmt(c["empty"], total),
                fmt(c["error"], total),
                n_classifier_errors_by_group[group] or "",
            ]
        )

    headers = [
        "Agent",
        "Trials",
        "Telemetry calls",
        "Avg calls/trial",
        "Success",
        "Empty",
        "Error",
        "Judge errors",
    ]
    print("## Per-agent")
    print(tabulate(rows, headers=headers, tablefmt="github"))

    strat_rows = []
    for group in sorted(counts_by_group, key=_group_sort_key):
        for ttype in TELEM_TYPES:
            c = counts_by_group_type.get((group, ttype), Counter())
            total = c["success"] + c["empty"] + c["error"]
            if not total and not n_classifier_errors_by_group_type[(group, ttype)]:
                continue
            strat_rows.append(
                [
                    _short_group_label(group),
                    ttype,
                    total,
                    fmt(c["success"], total),
                    fmt(c["empty"], total),
                    fmt(c["error"], total),
                    n_classifier_errors_by_group_type[(group, ttype)] or "",
                ]
            )
    if strat_rows:
        print("\n## Per-agent × telemetry type")
        print(
            tabulate(
                strat_rows,
                headers=[
                    "Agent",
                    "Type",
                    "Telemetry calls",
                    "Success",
                    "Empty",
                    "Error",
                    "Judge errors",
                ],
                tablefmt="github",
            )
        )

    fig_dir = args.output_dir / "figures"
    groups_sorted = sorted(counts_by_group, key=_group_sort_key)
    _plot_stacked(
        groups_sorted,
        counts_by_group,
        "count",
        fig_dir / "telemetry_calls_count.pdf",
        per_trial_calls_by_group=per_trial_calls_by_group,
    )
    _plot_stacked(
        groups_sorted,
        counts_by_group,
        "rate",
        fig_dir / "telemetry_calls_rate.pdf",
        per_trial_calls_by_group=per_trial_calls_by_group,
    )


if __name__ == "__main__":
    main()
