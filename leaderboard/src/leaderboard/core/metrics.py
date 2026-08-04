"""metrics: leaderboard metric computation.

compute_metrics / is_success / submission_by_task are pure functions over
per-task reward lists. compute_submission_metrics is the one exception: it
fetches the submission's trials from the hub (via core.hub) and is shared by
the static checker (ci.static_analysis) and the promote step, so the metrics
shown on the PR and the metrics written into the submission can't diverge. Also
home to the whole-submission resource totals (tokens / cost) computed from bulk
trial metadata.

ORCA-bench rewards are graded in [0, 1], not pass/fail -- see compute_metrics.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from leaderboard.core.hub import (
    RCA_ACCURACY_METRIC,
    REWARD_METRIC,
    submission_trials,
    trial_metric,
    trial_reward,
)
from leaderboard.core.task_groups import TaskLabel, label_for, task_labels


class RewardRangeError(ValueError):
    """A trial reward fell outside [0, 1]."""


def is_success(reward) -> bool:
    """A trial 'succeeded' iff reward > 0. NOT the metric -- ORCA-bench rewards
    are graded, so any partial credit counts here. Used only for the trial
    error breakdown in the PR comment."""
    return bool(reward) and float(reward) > 0


def _reward(r) -> float:
    """A trial's reward as a float in [0, 1]. A missing reward (errored trial)
    is 0.0 -- errored trials are not excluded from the metric."""
    value = 0.0 if r is None else float(r)
    if not 0.0 <= value <= 1.0:
        raise RewardRangeError(
            f"reward {value} outside [0, 1]; ORCA-bench verifiers must emit a "
            "normalized reward (see check_prediction.py). Refusing to compute "
            "a metric that would be silently skewed."
        )
    return value


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _sample_variance(xs: list[float]) -> float:
    """Unbiased (ddof=1) sample variance; 0.0 for fewer than 2 points."""
    k = len(xs)
    if k < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (k - 1)


def compute_metrics(by_task: dict[str, list]) -> tuple[float, float]:
    """Return (accuracy, accuracy_stderr) as percentages.

    ORCA-bench rewards are graded in [0, 1] (a normalized rubric score), not
    pass/fail, so accuracy is the mean reward over all trials -- a trial scoring
    0.33 contributes a third of a point, not a full success.

    accuracy_stderr uses whichever of two estimators the trial layout supports:

    * **Within-task** (some task has k >= 2): the sampling error of the mean,
      s^2 = (1/n^2) * sum_i var(rewards_i) / k_i, where var is the ddof=1 sample
      variance. For pass/fail rewards this reduces exactly to the classic
      p_i (1 - p_i) / (k_i - 1), so binary submissions score identically to
      Harbor's own importer.
    * **Between-task** (every task has exactly one trial): within-task variance
      is undefined at k = 1, which would report a meaningless 0.00. Fall back to
      the spread across tasks, s^2 = var(task means) / n. This is the standard
      SE for a mean over a task set and is what MIN_TRIALS_PER_TASK = 1 yields.
    """
    rewards_by_task = {t: [_reward(r) for r in rs] for t, rs in by_task.items()}
    all_rewards = [r for rs in rewards_by_task.values() for r in rs]
    accuracy = 100.0 * _mean(all_rewards) if all_rewards else 0.0

    n = len(rewards_by_task)
    if not n:
        return accuracy, 0.0

    if any(len(rs) >= 2 for rs in rewards_by_task.values()):
        variance = (
            sum(_sample_variance(rs) / len(rs) for rs in rewards_by_task.values()) / n**2
        )
    else:
        variance = _sample_variance([rs[0] for rs in rewards_by_task.values()]) / n

    return accuracy, 100.0 * variance**0.5


def stderr_basis(by_task: dict[str, list]) -> str:
    """Which estimator compute_metrics used, for the PR comment."""
    return (
        "within-task"
        if any(len(rs) >= 2 for rs in by_task.values())
        else "between-task"
    )


def submission_by_task(
    trials: list[dict], submission: dict
) -> tuple[dict[str, list], int, int]:
    """Per-task reward lists with maintainer overrides joined in.

    - `disqualified_trials` -> reward 0 (dock spurious pass)
    - `credited_trials` -> reward 1 (credit spurious fail)

    ORCA-bench has no automated trajectory judge, so these lists are edited by
    hand on the bot PR when a trial is found to be scored wrongly. The metric
    never mutates the hub -- those lists are the record. If a trial appears in
    both, disqualification wins. Returns (by_task, n_disqualified, n_credited).

    Un-overridden rewards come from the eval metric *named* `reward` (via
    core.hub.trial_reward), never the row's derived `reward` scalar -- the Hub
    derives that positionally and can publish a companion metric in its place.
    """
    disq = {d["trial_id"] for d in (submission.get("disqualified_trials") or [])}
    credited = {d["trial_id"] for d in (submission.get("credited_trials") or [])}
    by_task: dict[str, list] = defaultdict(list)
    n_disq = 0
    n_credited = 0
    for t in trials:
        tid = t.get("id")
        if tid in disq:
            by_task[t.get("task_name")].append(0)
            n_disq += 1
        elif tid in credited:
            by_task[t.get("task_name")].append(1)
            n_credited += 1
        else:
            by_task[t.get("task_name")].append(trial_reward(t))
    return by_task, n_disq, n_credited


# The subset breakdown: (metrics-key suffix, display label, predicate over a
# TaskLabel). Shared by the computation and by every renderer so the published
# keys, the PR comment and the merge comment cannot drift apart. `accuracy`
# (reward over all tasks) is computed separately -- it is the ranking metric and
# predates the breakdown.
SUBSETS: tuple[tuple[str, str, Callable[[TaskLabel], bool]], ...] = (
    ("incident", "Incident", lambda t: not t.is_control),
    ("control", "Control", lambda t: t.is_control),
    ("incident_easy", "Incident · easy", lambda t: not t.is_control and t.difficulty == "easy"),
    ("incident_medium", "Incident · medium", lambda t: not t.is_control and t.difficulty == "medium"),
    ("incident_hard", "Incident · hard", lambda t: not t.is_control and t.difficulty == "hard"),
)

# Which per-trial eval metric each column of the breakdown aggregates, and the
# metrics-key prefix it is published under. `rca_accuracy` is defined only over
# incident tasks: control tasks have no root cause to name, so the verifier
# scores it 0 there by construction and a control column would be a constant.
SUBSET_METRICS: tuple[tuple[str, str, str, bool], ...] = (
    # (key prefix, eval metric name, display header, incident-only)
    ("accuracy", REWARD_METRIC, "Reward", False),
    ("rca_accuracy", RCA_ACCURACY_METRIC, "RCA accuracy", True),
)


# Comment-table headers for the resource totals, shared by every renderer
# (static analysis, merge submit) so the tables can't drift apart.
RESOURCE_HEADERS = ("Tokens", "Cost")


def format_resource_cells(metrics: dict) -> tuple[str, str]:
    """Display cells for the resource totals, in RESOURCE_HEADERS order;
    em-dash for a field the metrics dict does not carry."""

    def fmt(key: str, spec: str) -> str:
        value = metrics.get(key)
        return spec.format(value) if value is not None else "—"

    return (
        fmt("total_tokens", "{:,}"),
        fmt("total_cost_usd", "${:,.2f}"),
    )


class EmptySubsetError(ValueError):
    """A breakdown subset has no trials, so its metric would read 0.00%."""


def compute_subset_metrics(
    trials: list[dict], submission: dict, labels: dict[str, TaskLabel] | None = None
) -> dict[str, float]:
    """The per-subset breakdown: reward and rca_accuracy by task group.

    Returns `{"<prefix>_<subset>": percentage}` for every (metric, subset) pair
    in SUBSET_METRICS x SUBSETS, e.g. `accuracy_incident_easy`. Values are means
    over the subset's trials, computed by `compute_metrics` so they inherit its
    contract (a missing value counts as 0, and a value outside [0, 1] is
    rejected rather than silently skewing the result).

    The disqualified / credited overrides apply to the **reward** subsets only,
    matching `accuracy`. Those lists record that a trial's *score* was wrong;
    they say nothing about whether the agent named the root cause, so rewriting
    `rca_accuracy` from them would invent a verdict no judge produced.

    Every subset is populated by construction -- a submission must cover all of
    the dataset's tasks, which span both kinds and all three difficulties -- so
    an empty one means the labels and the trials disagree. That raises rather
    than publishing a plausible-looking 0.00%.
    """
    labels = task_labels() if labels is None else labels
    disq = {d["trial_id"] for d in (submission.get("disqualified_trials") or [])}
    credited = {d["trial_id"] for d in (submission.get("credited_trials") or [])}

    out: dict[str, float] = {}
    for prefix, metric, _header, incident_only in SUBSET_METRICS:
        overridable = metric == REWARD_METRIC
        for suffix, _label, predicate in SUBSETS:
            if incident_only and suffix == "control":
                continue
            by_task: dict[str, list] = defaultdict(list)
            for t in trials:
                if not predicate(label_for(labels, t.get("task_name"))):
                    continue
                tid = t.get("id")
                if overridable and tid in disq:
                    value = 0
                elif overridable and tid in credited:
                    value = 1
                else:
                    value = trial_metric(t, metric)
                by_task[t.get("task_name")].append(value)
            if not by_task:
                raise EmptySubsetError(
                    f"no trials for subset {suffix!r}; the submission's tasks "
                    "and the pinned dataset's labels disagree, so "
                    f"{prefix}_{suffix} would be published as 0.00%."
                )
            mean, _ = compute_metrics(by_task)
            out[f"{prefix}_{suffix}"] = round(mean, 2)
    return out


def format_subset_table(metrics: dict, counts: dict[str, int] | None = None) -> list[str]:
    """The breakdown as markdown rows: one line per subset, one column per
    metric. Shared by the static-analysis comment and the merge comment so the
    two cannot drift; `counts` adds an `n` column when the caller has it.
    """
    headers = [h for _p, _m, h, _i in SUBSET_METRICS]
    head = "| Subset | " + " | ".join(headers) + (" | n |" if counts else " |")
    rule = "|" + " --- |" * (1 + len(headers) + (1 if counts else 0))
    rows = [head, rule]
    for suffix, label, _predicate in SUBSETS:
        cells = []
        for prefix, _metric, _header, incident_only in SUBSET_METRICS:
            value = metrics.get(f"{prefix}_{suffix}")
            cells.append("—" if value is None else f"{value:.2f}%")
            del incident_only  # control cells are simply absent -> em-dash
        line = f"| {label} | " + " | ".join(cells)
        if counts:
            line += f" | {counts.get(suffix, 0)} |"
        else:
            line += " |"
        rows.append(line)
    return rows


def subset_counts(trials: list[dict], labels: dict[str, TaskLabel]) -> dict[str, int]:
    """Trial count per subset, for the `n` column of the breakdown table."""
    counts: dict[str, int] = {}
    for suffix, _label, predicate in SUBSETS:
        counts[suffix] = sum(
            1 for t in trials if predicate(label_for(labels, t.get("task_name")))
        )
    return counts


def compute_resource_metrics(trials: list[dict]) -> dict:
    """Whole-submission resource totals from bulk trial metadata: tokens
    (input + output) and dollars.

    Every trial that ran counts -- disqualified / credited trials still
    consumed tokens and dollars. A trial missing a field is skipped for that
    total; a field no trial reports is omitted entirely, so the result always
    satisfies the leaderboard metrics_schema (these fields are optional there).
    """
    tokens = [
        (t.get("input_tokens") or 0) + (t.get("output_tokens") or 0)
        for t in trials
        if t.get("input_tokens") is not None or t.get("output_tokens") is not None
    ]
    costs = [t["cost_usd"] for t in trials if t.get("cost_usd") is not None]
    out: dict = {}
    if tokens:
        out["total_tokens"] = sum(tokens)
    if costs:
        out["total_cost_usd"] = round(sum(costs), 2)
    return out


def compute_submission_metrics(submission: dict) -> dict:
    """The metrics record written into the submission JSON at promote.
    Honors disqualified_trials (reward 0) and credited_trials (reward 1) via
    the join. Fields match the leaderboard metrics_schema: accuracy +
    accuracy_stderr, the per-subset breakdown (reward and rca_accuracy by task
    group, see compute_subset_metrics), plus the whole-submission resource
    totals (tokens / cost), which count every trial that ran -- overrides change
    a trial's reward, not its resource usage."""
    trials = submission_trials(submission)
    by_task, _, _ = submission_by_task(trials, submission)
    accuracy, stderr = compute_metrics(by_task)
    return {
        "accuracy": round(accuracy, 2),
        "accuracy_stderr": round(stderr, 2),
        **compute_subset_metrics(trials, submission),
        **compute_resource_metrics(trials),
    }
