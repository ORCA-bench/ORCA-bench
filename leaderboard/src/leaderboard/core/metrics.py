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

from leaderboard.core.hub import submission_trials, trial_reward


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
    accuracy_stderr, plus the whole-submission resource totals (tokens / cost),
    which count every trial that ran -- overrides change a trial's reward, not
    its resource usage."""
    trials = submission_trials(submission)
    by_task, _, _ = submission_by_task(trials, submission)
    accuracy, stderr = compute_metrics(by_task)
    return {
        "accuracy": round(accuracy, 2),
        "accuracy_stderr": round(stderr, 2),
        **compute_resource_metrics(trials),
    }
