"""metrics: leaderboard metric computation.

compute_metrics / submission_by_task are pure functions over per-task value
lists. compute_submission_metrics is the one exception: it fetches the
submission's trials from the hub (via core.hub) and is shared by the static
checker (ci.static_analysis) and the promote step, so the metrics shown on the
PR and the metrics written into the submission can't diverge. Also home to the
whole-submission resource totals (tokens / cost) computed from bulk trial
metadata.

What the leaderboard publishes is the METRICS list below -- the three panels of
plot_rca_and_hallucination.py. `submission_by_task` and the reward it reads are
no longer published; they survive because the coverage check counts the tasks
that produced a scored trial.

ORCA-bench rewards are graded in [0, 1], not pass/fail -- see compute_metrics.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from leaderboard.core.hub import (
    HALLUCINATE_METRIC,
    PRIVATE,
    RCA_ACCURACY_METRIC,
    REWARD_METRIC,
    Board,
    private_trial_finished,
    submission_board,
    submission_trials,
    trial_metric,
    trial_reward,
)
from leaderboard.core.task_groups import (
    TaskLabel,
    TaskLabelError,
    label_for,
    task_labels,
)


class RewardRangeError(ValueError):
    """A trial reward fell outside [0, 1]."""


def _reward(r) -> float:
    """A trial's reward as a float in [0, 1].

    `None` is not accepted: an unscored trial is dropped by `submission_by_task`
    before it reaches here, so a `None` arriving means a caller bypassed that
    filter and is about to average a trial that has no verdict. Raising keeps
    the old "missing reward counts as 0" behaviour from creeping back in.
    """
    if r is None:
        raise RewardRangeError(
            "reward is None; unscored trials are excluded by submission_by_task "
            "and must never reach the metric. Scoring one 0 would understate the "
            "result by counting a trial that was never judged."
        )
    value = float(r)
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

    Trials that produced no verdict at all are **excluded**, so the metric is a
    mean over scored trials -- the same convention as utils.load_trials, which
    the analysis pipeline (format_acc.py) uses. An override still wins over
    exclusion: `credited_trials` on an errored trial is a maintainer's explicit
    judgement and must not be silently discarded. The number excluded is
    `len(trials)` minus the trials in `by_task`.
    """
    disq, credited = _overrides(submission)
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
            reward = trial_reward(t)
            # An unscored trial (no evals -- the run died before the verifier
            # produced a verdict) is dropped, not counted as 0. Matches
            # utils.load_trials, which skips a trial with no reward-details.json
            # or no nested.rubrics. A task left with no scored trial disappears
            # from by_task and fails the coverage check, which is the intended
            # loud failure.
            if reward is None:
                continue
            by_task[t.get("task_name")].append(reward)
    return by_task, n_disq, n_credited


def _overrides(submission: dict) -> tuple[set[str], set[str]]:
    """(disqualified, credited) trial ids from the submission's override lists."""
    disq = {d["trial_id"] for d in (submission.get("disqualified_trials") or [])}
    credited = {d["trial_id"] for d in (submission.get("credited_trials") or [])}
    return disq, credited


def submission_coverage(
    trials: list[dict], submission: dict, board: Board
) -> dict[str, int]:
    """Per-task count of the trials that cover it, for the coverage check.

    On the public board that is exactly what `submission_by_task` keeps: a
    trial with a verdict, or one a maintainer overrode. On the private board
    no trial has a verdict on the Hub by construction -- the `-hidden` tasks
    are scored out-of-band by /judge -- so a trial that reached its verifier
    (core.hub.private_trial_finished: no error, or one harbor verifies after,
    such as an agent timeout) covers its task as well. A private trial that
    failed before that still does not: there is no report for /judge to score.
    Overridden trials are already counted by `submission_by_task`, so they are
    not counted again here.
    """
    by_task, _, _ = submission_by_task(trials, submission)
    covered = {task: len(rs) for task, rs in by_task.items()}
    if board is PRIVATE:
        disq, credited = _overrides(submission)
        for t in trials:
            if t.get("id") in disq or t.get("id") in credited:
                continue
            if private_trial_finished(t):
                covered[t.get("task_name")] = covered.get(t.get("task_name"), 0) + 1
    return covered


@dataclass(frozen=True)
class MetricSpec:
    """One published metric: what to average, over which tasks, shown how."""

    key: str
    metric: str
    header: str
    subset: Callable[[TaskLabel], bool]


def _incident(label: TaskLabel) -> bool:
    return not label.is_control


def _incident_at(difficulty: str) -> Callable[[TaskLabel], bool]:
    return lambda label: not label.is_control and label.difficulty == difficulty


# Everything the leaderboard publishes. These three are the panels of
# plot_rca_and_hallucination.py: the two harder RCA tiers, and the hallucination
# rate over every incident task. Easy is deliberately absent -- it looks
# deceptively good and pulls attention from the realistic setting.
#
# All three are restricted to incident tasks. A control task has no root cause,
# so `rca_accuracy` is 0 there by construction and `hallucinate_any` is not
# emitted at all; including controls would either flatten the metric or raise.
#
# The two RCA metrics are higher-is-better; the hallucination rate is lower-is-
# better. Shared by the computation and every renderer so the published keys and
# the comment tables cannot drift apart.
METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "rca_accuracy_incident_medium",
        RCA_ACCURACY_METRIC,
        "RCA Accuracy (Medium)",
        _incident_at("medium"),
    ),
    MetricSpec(
        "rca_accuracy_incident_hard",
        RCA_ACCURACY_METRIC,
        "RCA Accuracy (Hard)",
        _incident_at("hard"),
    ),
    MetricSpec(
        "hallucinate_any_incident",
        HALLUCINATE_METRIC,
        "Hallucination Rate",
        _incident,
    ),
)

# A leaderboard cell renders exactly one stored field, so `mean ± se` has to be
# pre-formatted. Each metric is therefore published three ways: `<key>` (the
# numeric mean, which also ranks the board), `<key>_stderr`, and `<key>_display`
# -- the only one with a column. Ranking must use the numeric mean: the Hub
# compares strings lexicographically, so ranking on a display value would sort
# "9.5" above "26.6".
STDERR_SUFFIX = "_stderr"
DISPLAY_SUFFIX = "_display"


def format_measurement(mean: float, stderr: float) -> str:
    """The cell for one metric: `**83.8%** ± 1.2%`.

    One decimal is all these measurements support -- the standard errors run
    around a point, so a second decimal would imply precision that isn't there.
    Both halves carry the percent sign, since the cell has no unit elsewhere.

    Markdown, with the mean bold so the eye lands on it rather than the error
    bar. That means the columns showing this must declare
    `display_type: "markdown"` in leaderboard.json -- a text cell renders the
    asterisks literally. It also renders correctly in the PR and merge comments,
    which are markdown too.
    """
    return f"**{mean:.1f}%** ± {stderr:.1f}%"


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
) -> dict[str, float | str]:
    """Every metric the leaderboard publishes, three fields per METRICS entry.

    Returns `{key: mean, key_stderr: se, key_display: "mean ± se"}` for each
    spec -- percentages, means over the spec's subset, computed by
    `compute_metrics` so they inherit its contract (a value outside [0, 1] is
    rejected rather than silently skewing the result).

    `submission` is read only for its board (to pick the pinned dataset's
    labels). Its disqualified / credited lists are NOT applied: they force a
    trial's *reward*, and none of the published metrics is reward-derived any
    more. Overriding `rca_accuracy` or `hallucinate_any` from them would invent
    a verdict no judge produced.

    Unscored trials are excluded rather than counted as 0, matching
    `submission_by_task` and utils.load_trials.

    Every subset is populated by construction -- a submission must cover all of
    the dataset's tasks, which span both kinds and all three difficulties -- so
    an empty one means the labels and the trials disagree, or every trial in it
    went unscored. That raises rather than publishing a plausible-looking 0.00%.
    """
    # See docstring: no published metric is reward-derived, so only the board
    # is read off the submission.
    labels = task_labels(submission_board(submission)) if labels is None else labels
    out: dict[str, float | str] = {}
    for spec in METRICS:
        by_task: dict[str, list] = defaultdict(list)
        for t in trials:
            if not spec.subset(label_for(labels, t.get("task_name"))):
                continue
            value = trial_metric(t, spec.metric)
            # Unscored trials are excluded, exactly as in submission_by_task --
            # a trial with no verdict has nothing to contribute.
            if value is None:
                continue
            by_task[t.get("task_name")].append(value)
        if not by_task:
            raise EmptySubsetError(
                f"no trials for {spec.key!r}; the submission's tasks and the "
                "pinned dataset's labels disagree, or every trial in the subset "
                "went unscored, so it would be published as 0.00%."
            )
        mean, stderr = compute_metrics(by_task)
        out[spec.key] = round(mean, 2)
        out[spec.key + STDERR_SUFFIX] = round(stderr, 2)
        out[spec.key + DISPLAY_SUFFIX] = format_measurement(mean, stderr)
    return out


def format_metrics_table(metrics: dict, counts: dict[str, int] | None = None) -> list[str]:
    """The published metrics as markdown rows: one line per METRICS entry.

    Renders the `_display` field, so the comment shows exactly the cell the
    leaderboard will show. Shared by the static-analysis comment and the merge
    comment so the two cannot drift; `counts` adds an `n` column when the caller
    has it.
    """
    head = "| Metric | Result |" + (" n |" if counts else "")
    rows = [head, "|" + " --- |" * (2 + (1 if counts else 0))]
    for spec in METRICS:
        value = metrics.get(spec.key + DISPLAY_SUFFIX)
        line = f"| {spec.header} | {value if value is not None else '—'} |"
        if counts:
            line += f" {counts.get(spec.key, 0)} |"
        rows.append(line)
    return rows


def metric_counts(
    trials: list[dict], labels: dict[str, TaskLabel]
) -> dict[str, int]:
    """Trial count per metric, for the `n` column.

    Counts the trials that actually reached each metric, so `n` is the
    denominator the value beside it was computed over: a trial outside the
    subset, or one that produced no value for that metric, is not counted.
    """
    counts: dict[str, int] = {}
    for spec in METRICS:
        counts[spec.key] = sum(
            1
            for t in trials
            if spec.subset(label_for(labels, t.get("task_name")))
            and trial_metric(t, spec.metric) is not None
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


# ---------------------------------------------------------------------------
# The private board: verdicts come from /judge, not from the trials.
#
# A -hidden task has no in-container verifier, so its Hub row carries no
# metrics (`evals: {}`). /judge scores the agent's report against the held-back
# ground truth and commits one record per trial to leaderboard/scores/<name>
# .json (see .github/workflows/leaderboard-judge.yml). Each record is exactly
# what the public verifier would have written to reward.json -- run_llm_judge
# builds it with check_prediction.build_rewards -- plus the judge's `mode`.

# The judge's mode on a control task. task_groups cannot tell a private control
# task from an incident one: the -hidden task.toml keeps no `events`, because
# whether an incident happened *is* the answer. Only the judge knows.
JUDGE_CONTROL_MODE = "no_incident_llm_judge"
# Modes recorded when the judge could not score a trial. The record carries a
# reward of 0 so every score file has one shape, but there is no verdict in it:
# such a trial is unscored, not a 0.
JUDGE_ERROR_MODES = frozenset({"llm_judge_error", "hub_fetch_error"})
# The record fields that are verifier metrics (build_rewards' keys).
JUDGE_METRIC_KEYS = (REWARD_METRIC, RCA_ACCURACY_METRIC, HALLUCINATE_METRIC)


def load_scores(path: Path) -> dict[str, dict]:
    """The /judge scores file for a submission: `{trial_id: record}`."""
    scores = json.loads(path.read_text())
    if not isinstance(scores, dict) or not all(
        isinstance(v, dict) and v.get("trial_id") == k for k, v in scores.items()
    ):
        raise ValueError(f"{path} is not a /judge scores file ({{trial_id: record}})")
    return scores


def scores_coverage_failure(trials: list[dict], scores: dict[str, dict]) -> str:
    """Why `scores` does not give every trial a verdict, or "" if it does.

    A private submission must not be merged on a partial judge run: a trial
    with no record, or one the judge could not score, would silently drop out
    of the mean. The maintainer re-runs /judge instead.
    """
    missing = [t["id"] for t in trials if t.get("id") not in scores]
    errored = [
        t["id"]
        for t in trials
        if t.get("id") in scores and scores[t["id"]].get("mode") in JUDGE_ERROR_MODES
    ]
    if not missing and not errored:
        return ""
    parts = []
    if missing:
        parts.append(f"{len(missing)} trial(s) have no judge score")
    if errored:
        parts.append(f"{len(errored)} the judge could not score")
    return ", ".join(parts) + f" (of {len(trials)}); re-run /judge"


def judged_trials(trials: list[dict], scores: dict[str, dict]) -> list[dict]:
    """Bulk rows with the judge's verdict where the verifier's would have been.

    A judge record is what the in-container verifier writes to reward.json, so
    it goes where the Hub puts that: the row's `evals`, in the nested shape
    bulk rows carry. Every other field of the row (id, task_name, tokens, cost,
    error_type) is kept, so `compute_subset_metrics`, `metric_counts` and
    `compute_resource_metrics` read a judged private trial exactly as they read
    a verified public one. A trial with no record, or an errored one, keeps
    `evals: {}` and stays unscored -- excluded by the metric, and named by
    `scores_coverage_failure`. The judge's mode rides along as `judge_mode` for
    `private_task_labels`.
    """
    out: list[dict] = []
    for t in trials:
        row = dict(t)
        rec = scores.get(t.get("id"))
        if rec is not None and rec.get("mode") not in JUDGE_ERROR_MODES:
            row["evals"] = {
                k: {"metrics": [{k: rec[k]}]} for k in JUDGE_METRIC_KEYS if k in rec
            }
            row["judge_mode"] = rec.get("mode")
        out.append(row)
    return out


def private_task_labels(board: Board, judged: list[dict]) -> dict[str, TaskLabel]:
    """Labels for the private board: difficulty from the task, control from the judge.

    The -hidden task.toml keeps `difficulty`, so that half comes from
    `task_labels` as usual; its `events` are stripped, so `is_control` there
    is True for every task and is replaced by the judge's verdict on the task's
    trials (`JUDGE_CONTROL_MODE`). Control-ness is a property of the task, so
    every judged trial of a task must agree; a task none of whose trials was
    judged cannot be labelled and raises, which `scores_coverage_failure` will
    already have reported.
    """
    labels = dict(task_labels(board))
    control_by_task: dict[str, set[bool]] = defaultdict(set)
    for t in judged:
        if "judge_mode" in t:
            control_by_task[t.get("task_name")].add(t["judge_mode"] == JUDGE_CONTROL_MODE)
    disagree = sorted(task for task, kinds in control_by_task.items() if len(kinds) > 1)
    if disagree:
        raise TaskLabelError(
            f"{len(disagree)} task(s) judged as both control and incident, e.g. "
            f"{disagree[:3]}; a task's trials cannot disagree on whether an "
            "incident happened."
        )
    unjudged = sorted(set(t.get("task_name") for t in judged) - control_by_task.keys())
    if unjudged:
        raise TaskLabelError(
            f"{len(unjudged)} task(s) have no judged trial, e.g. {unjudged[:3]}; "
            "control/incident is known only from the judge, so they cannot be "
            "placed in a subset."
        )
    for task, kinds in control_by_task.items():
        base = label_for(labels, task)
        labels[task] = TaskLabel(difficulty=base.difficulty, is_control=kinds.pop())
    return labels


def compute_submission_metrics(
    submission: dict, scores: dict[str, dict] | None = None
) -> dict:
    """The metrics record written into the submission JSON.

    Fields match the leaderboard metrics_schema: three fields per METRICS entry
    (mean, stderr, display -- see compute_subset_metrics), plus the
    whole-submission resource totals (tokens / cost), which count every trial
    that ran including the unscored ones, since an errored trial still burned
    tokens.

    Public board: written at promote, from the trials' own verdicts. Private
    board: written by /judge on the bot PR, from its `scores` -- required
    there, and they must cover every trial (`scores_coverage_failure`), since
    a partial mean would be published as the whole.

    The disqualified / credited lists no longer affect anything published: they
    force a trial's *reward*, and no published metric is reward-derived.
    """
    board = submission_board(submission)
    trials = submission_trials(submission)
    if board is PRIVATE:
        if scores is None:
            raise ValueError(
                "private-board metrics come from the /judge scores file; none given"
            )
        bad = scores_coverage_failure(trials, scores)
        if bad:
            raise ValueError(f"judge scores do not cover the submission: {bad}")
        judged = judged_trials(trials, scores)
        labels = private_task_labels(board, judged)
        return {
            **compute_subset_metrics(judged, submission, labels),
            **compute_resource_metrics(trials),
        }
    return {
        **compute_subset_metrics(trials, submission),
        **compute_resource_metrics(trials),
    }
