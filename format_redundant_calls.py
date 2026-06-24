"""Per-agent breakdown of redundant telemetry-call rates.

Reads ``output_dir/redundant_telemetry_calls/{trial_id}/{type}.json``
per-(trial, type) files emitted by ``classify_redundant_calls.py``, joins
them with the agent-group label derived from each batch's ``config.json``
under ``jobs_dir``, and prints two markdown tables to stdout: an overall
per-agent summary and a per-agent × telemetry-type breakdown
(metrics / logs / traces). Also saves stacked-bar PDFs (count + rate) at
``output_dir/figures/redundant_calls_{count,rate}.pdf``.

Examples::

    cd examples/otel-demo
    uv run python format_redundant_calls.py -od out-0514 -jd jobs-sub
"""

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tabulate import tabulate

from format_telemetry_calls import (
    TELEM_TYPES,
    _group_sort_key,
    _plot_stacked,
    _short_group_label,
    build_trial_to_group,
)
from utils import get_base_parser, load_invalid_trial_ids, setup_logging

logger = logging.getLogger(__name__)

LABELS = ("unique", "redundant")
LABEL_COLORS = {
    "unique": "#2ca02c",  # green — canonical / first occurrence
    "redundant": "#d62728",  # red — repeats of an earlier query
}


@dataclass
class RedundantAggregation:
    """Per-group aggregates emitted by :func:`aggregate_redundant_classifications`."""

    counts_by_group: dict[str, Counter[str]] = field(default_factory=dict)
    counts_by_group_type: dict[tuple[str, str], Counter[str]] = field(
        default_factory=dict
    )
    n_trials_by_group: Counter[str] = field(default_factory=Counter)
    per_trial_calls_by_group: dict[str, list[int]] = field(default_factory=dict)
    # One Counter per trial so callers can compute per-trial redundancy
    # fractions (with per-model SEM) without re-walking the JSONs.
    per_trial_label_counts_by_group: dict[str, list[Counter[str]]] = field(
        default_factory=dict
    )
    n_unmapped: int = 0
    n_failed_groups: int = 0


def aggregate_redundant_classifications(
    classifications_dir: Path,
    trial_to_group: dict[str, str],
    invalid_ids: set[str],
) -> RedundantAggregation:
    """Walk ``classifications_dir`` and aggregate unique/redundant counts per group.

    Reads ``{trial_id}/{type}.json`` records emitted by
    ``classify_redundant_calls.py`` and joins them against ``trial_to_group``.
    Returns the full set of aggregates so callers (this script's ``main`` and
    ``plot_efficiency.py``) can render whichever views they need without
    reopening the JSONs.
    """
    agg = RedundantAggregation()
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
                d = json.loads(json_path.read_text())
            except Exception as exc:
                logger.warning(f"Could not parse {json_path}: {exc}")
                continue
            if "error" in d:
                agg.n_failed_groups += 1
                continue
            for rec in d.values():
                ttype = rec.get("telem_type")
                label = "redundant" if rec.get("is_redundant") else "unique"
                c[label] += 1
                trial_calls += 1
                trial_label_counts[label] += 1
                if ttype is not None:
                    agg.counts_by_group_type.setdefault((group, ttype), Counter())[
                        label
                    ] += 1
        agg.per_trial_calls_by_group.setdefault(group, []).append(trial_calls)
        agg.per_trial_label_counts_by_group.setdefault(group, []).append(
            trial_label_counts
        )
    return agg


def main() -> None:
    """Aggregate per-agent redundant-call rates and emit tables + plots."""
    parser = get_base_parser()
    parser.description = "Per-agent breakdown of redundant telemetry-call rates."
    parser.add_argument(
        "--jobs-dir",
        "-jd",
        type=Path,
        default=Path("jobs"),
        help="Path to the Harbor jobs directory (default: jobs)",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    classifications_dir = args.output_dir / "redundant_telemetry_calls"
    if not classifications_dir.is_dir():
        logger.error(f"No redundant_telemetry_calls/ at {classifications_dir}")
        sys.exit(1)

    trial_to_group = build_trial_to_group(args.jobs_dir)
    invalid_ids: set[str] = (
        load_invalid_trial_ids(args.filter_xlsx) if args.filter_xlsx else set()
    )

    agg = aggregate_redundant_classifications(
        classifications_dir, trial_to_group, invalid_ids
    )
    counts_by_group = agg.counts_by_group
    counts_by_group_type = agg.counts_by_group_type
    n_trials_by_group = agg.n_trials_by_group
    per_trial_calls_by_group = agg.per_trial_calls_by_group

    if agg.n_unmapped:
        logger.warning(
            f"{agg.n_unmapped} trial(s) had no batch mapping in {args.jobs_dir} (skipped)"
        )
    if agg.n_failed_groups:
        logger.warning(
            f"{agg.n_failed_groups} (trial, type) JSONs had an error field "
            "(skipped; re-run classify_redundant_calls.py to retry)"
        )
    if not counts_by_group:
        logger.error("No classifications loaded; nothing to aggregate.")
        sys.exit(1)

    def fmt(n: int, total: int) -> str:
        return f"{n} ({100 * n / total:.1f}%)" if total else f"{n} (—)"

    rows = []
    for group in sorted(counts_by_group, key=_group_sort_key):
        c = counts_by_group[group]
        total = c["unique"] + c["redundant"]
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
                fmt(c["unique"], total),
                fmt(c["redundant"], total),
            ]
        )

    headers = [
        "Agent",
        "Trials",
        "Telemetry calls",
        "Avg calls/trial",
        "Unique",
        "Redundant",
    ]
    print("## Per-agent")
    print(tabulate(rows, headers=headers, tablefmt="github"))

    strat_rows = []
    for group in sorted(counts_by_group, key=_group_sort_key):
        for ttype in TELEM_TYPES:
            c = counts_by_group_type.get((group, ttype), Counter())
            total = c["unique"] + c["redundant"]
            if not total:
                continue
            strat_rows.append(
                [
                    _short_group_label(group),
                    ttype,
                    total,
                    fmt(c["unique"], total),
                    fmt(c["redundant"], total),
                ]
            )
    if strat_rows:
        print("\n## Per-agent × telemetry type")
        print(
            tabulate(
                strat_rows,
                headers=["Agent", "Type", "Telemetry calls", "Unique", "Redundant"],
                tablefmt="github",
            )
        )

    fig_dir = args.output_dir / "figures"
    groups_sorted = sorted(counts_by_group, key=_group_sort_key)
    _plot_stacked(
        groups_sorted,
        counts_by_group,
        "count",
        fig_dir / "redundant_calls_count.pdf",
        per_trial_calls_by_group=per_trial_calls_by_group,
        labels=LABELS,
        label_colors=LABEL_COLORS,
    )
    _plot_stacked(
        groups_sorted,
        counts_by_group,
        "rate",
        fig_dir / "redundant_calls_rate.pdf",
        per_trial_calls_by_group=per_trial_calls_by_group,
        labels=LABELS,
        label_colors=LABEL_COLORS,
    )


if __name__ == "__main__":
    main()
