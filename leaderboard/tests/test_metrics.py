"""Unit tests for the reward override join and the graded-reward metric."""

from __future__ import annotations

import unittest
import unittest.mock

from leaderboard.core.metrics import (
    DISPLAY_SUFFIX,
    STDERR_SUFFIX,
    EmptySubsetError,
    RewardRangeError,
    compute_metrics,
    compute_subset_metrics,
    format_measurement,
    JUDGE_CONTROL_MODE,
    judged_trials,
    private_task_labels,
    scores_coverage_failure,
    submission_by_task,
    submission_coverage,
)
from leaderboard.core.hub import PRIVATE, PUBLIC, private_trial_finished, trial_metric
from leaderboard.core.task_groups import TaskLabel, TaskLabelError


_UNSET = object()


def _trial(tid: str, task: str, reward, *, evals=_UNSET) -> dict:
    """A bulk hub trial row: the `reward` scalar the Hub derives positionally,
    plus the `evals` block the metric is actually read from.

    By default the two agree. Pass ``evals=None`` for an errored trial (no
    evals at all), or an explicit dict to make them disagree -- which is what
    the Hub really does when a companion metric sorts ahead of `reward`.
    """
    if evals is _UNSET:
        evals = (
            None if reward is None else {"reward": {"metrics": [{"reward": reward}]}}
        )
    row = {"id": tid, "task_name": task, "reward": reward}
    if evals is not None:
        row["evals"] = evals
    return row


class SubmissionByTaskTests(unittest.TestCase):
    def test_disqualified_forces_reward_zero(self):
        trials = [
            _trial("t-pass", "org/a", 1.0),
            _trial("t-fp", "org/b", 1.0),
        ]
        sub = {"disqualified_trials": [{"trial_id": "t-fp", "reason": "spurious_pass"}]}
        by_task, n_disq, n_cred = submission_by_task(trials, sub)
        self.assertEqual(n_disq, 1)
        self.assertEqual(n_cred, 0)
        self.assertEqual(by_task["org/b"], [0])
        self.assertEqual(by_task["org/a"], [1.0])

    def test_credited_forces_reward_one(self):
        trials = [
            _trial("t-fail", "org/a", 0.0),
            _trial("t-fn", "org/b", None),
            _trial("t-ok", "org/c", 1.0),
        ]
        sub = {
            "credited_trials": [
                {"trial_id": "t-fail", "reason": "spurious_fail"},
                {"trial_id": "t-fn", "reason": "spurious_fail"},
            ]
        }
        by_task, n_disq, n_cred = submission_by_task(trials, sub)
        self.assertEqual(n_disq, 0)
        self.assertEqual(n_cred, 2)
        self.assertEqual(by_task["org/a"], [1])
        self.assertEqual(by_task["org/b"], [1])
        self.assertEqual(by_task["org/c"], [1.0])
        acc, _ = compute_metrics(by_task)
        self.assertEqual(acc, 100.0)

    def test_disqualify_wins_over_credit(self):
        trials = [_trial("t-both", "org/a", 0.0)]
        sub = {
            "disqualified_trials": [{"trial_id": "t-both", "reason": "spurious_pass"}],
            "credited_trials": [{"trial_id": "t-both", "reason": "spurious_fail"}],
        }
        by_task, n_disq, n_cred = submission_by_task(trials, sub)
        self.assertEqual(by_task["org/a"], [0])
        self.assertEqual(n_disq, 1)
        self.assertEqual(n_cred, 0)

    def test_missing_override_lists_are_noop(self):
        trials = [_trial("t1", "org/a", 0.0)]
        by_task, n_disq, n_cred = submission_by_task(trials, {})
        self.assertEqual(by_task["org/a"], [0.0])
        self.assertEqual((n_disq, n_cred), (0, 0))

    def test_reward_comes_from_the_named_eval_metric_not_the_row_scalar(self):
        """Regression guard for the 12.05%-vs-49.43% bug: the Hub derives a
        row's `reward` scalar from the alphabetically first metric, so
        `hallucinate_any` was published as the trial's score. The metric must
        read the eval metric named `reward` instead."""
        trials = [
            _trial(
                "t1",
                "org/a",
                0,  # what the Hub derived -- the hallucinate_any flag
                evals={
                    "reward": {"metrics": [{"reward": 0.75}]},
                    "rca_accuracy": {"metrics": [{"rca_accuracy": 1}]},
                    "hallucinate_any": {"metrics": [{"hallucinate_any": 0}]},
                },
            )
        ]
        by_task, _, _ = submission_by_task(trials, {})
        self.assertEqual(by_task["org/a"], [0.75])
        acc, _ = compute_metrics(by_task)
        self.assertEqual(acc, 75.0)

    def test_unscored_trial_is_excluded_not_zeroed(self):
        """A trial that produced no verdict is dropped, matching
        utils.load_trials -- it must not drag the mean down as a 0."""
        trials = [
            _trial("t-ok", "org/a", 1.0),
            _trial("t-err", "org/a", None, evals=None),
        ]
        by_task, _, _ = submission_by_task(trials, {})
        self.assertEqual(by_task["org/a"], [1.0])
        acc, _ = compute_metrics(by_task)
        self.assertEqual(acc, 100.0)

    def test_task_with_only_unscored_trials_disappears(self):
        """So the coverage check fails loudly rather than the task silently
        vanishing from the denominator."""
        trials = [
            _trial("t-ok", "org/a", 1.0),
            _trial("t-err", "org/b", None, evals=None),
        ]
        by_task, _, _ = submission_by_task(trials, {})
        self.assertEqual(sorted(by_task), ["org/a"])

    def test_excluded_count_is_recoverable(self):
        """The renderers report exclusions as len(trials) - trials in by_task;
        pin that arithmetic."""
        trials = [
            _trial("t-ok", "org/a", 1.0),
            _trial("t-err", "org/a", None, evals=None),
        ]
        by_task, _, _ = submission_by_task(trials, {})
        self.assertEqual(len(trials) - sum(len(v) for v in by_task.values()), 1)

    def test_credited_unscored_trial_still_counts(self):
        """An override is an explicit maintainer judgement and beats exclusion."""
        trials = [_trial("t-err", "org/a", None, evals=None)]
        sub = {"credited_trials": [{"trial_id": "t-err", "reason": "spurious_fail"}]}
        by_task, _, n_cred = submission_by_task(trials, sub)
        self.assertEqual(by_task["org/a"], [1])
        self.assertEqual(n_cred, 1)

    def test_disqualified_unscored_trial_still_counts(self):
        trials = [_trial("t-err", "org/a", None, evals=None)]
        sub = {"disqualified_trials": [{"trial_id": "t-err", "reason": "spurious_pass"}]}
        by_task, n_disq, _ = submission_by_task(trials, sub)
        self.assertEqual(by_task["org/a"], [0])
        self.assertEqual(n_disq, 1)

    def test_overrides_ignore_the_eval_metric(self):
        """The override join short-circuits before the reward is read."""
        trials = [
            _trial(
                "t-fp",
                "org/a",
                0,
                evals={"reward": {"metrics": [{"reward": 1.0}]}},
            )
        ]
        sub = {"disqualified_trials": [{"trial_id": "t-fp", "reason": "spurious_pass"}]}
        by_task, n_disq, _ = submission_by_task(trials, sub)
        self.assertEqual(by_task["org/a"], [0])
        self.assertEqual(n_disq, 1)


def _scored(tid: str, task: str, reward, rca, halluc=None) -> dict:
    """A trial row carrying the eval metrics the verifier publishes. Control
    trials get no `hallucinate_any` -- the verifier omits it there."""
    evals = {
        "reward": {"metrics": [{"reward": reward}]},
        "rca_accuracy": {"metrics": [{"rca_accuracy": rca}]},
    }
    if halluc is not None:
        evals["hallucinate_any"] = {"metrics": [{"hallucinate_any": halluc}]}
    return {"id": tid, "task_name": task, "reward": reward, "evals": evals}


# One task per (kind, difficulty) cell, so every metric's subset is populated.
_LABELS = {
    "org/inc-easy": TaskLabel("easy", False),
    "org/inc-medium": TaskLabel("medium", False),
    "org/inc-hard": TaskLabel("hard", False),
    "org/ctl": TaskLabel("hard", True),
}


def _trials() -> list[dict]:
    return [
        _scored("t-easy", "org/inc-easy", 1.0, 1, halluc=0),
        _scored("t-medium", "org/inc-medium", 0.5, 1, halluc=0),
        _scored("t-hard", "org/inc-hard", 0.0, 0, halluc=1),
        _scored("t-ctl", "org/ctl", 1.0, 0),
    ]


def _hidden_trial(tid: str, task: str, *, error_type=None, source=PRIVATE.dataset) -> dict:
    """A bulk row as the Hub stores a -hidden trial: empty rewards map, so
    `evals` is `{}` and the derived `reward` is null. Observed on the first
    uploaded private job (6b469112-157f-5d55-a7c2-25417c2a226d): 324/324 rows
    looked exactly like this, `error_type` None. An errored trial has the same
    `reward`/`evals` and differs only in `error_type`."""
    return {
        "id": tid,
        "task_name": task,
        "source": source,
        "reward": None,
        "evals": {},
        "error_type": error_type,
        "status": "completed",
        "is_scored": True,
    }


class PrivateTrialFinishedTests(unittest.TestCase):
    def test_clean_hidden_row_is_finished(self):
        self.assertTrue(private_trial_finished(_hidden_trial("t", "org/a-hidden")))

    def test_errored_row_is_not(self):
        """Same reward/evals shape; only error_type tells them apart."""
        self.assertFalse(
            private_trial_finished(
                _hidden_trial("t", "org/a-hidden", error_type="RuntimeError")
            )
        )

    def test_agent_errors_harbor_still_verifies_are_finished(self):
        """harbor's SingleStepTrial._run_agent swallows exactly these two and
        proceeds to the verifier, so the report is judged -- the public board
        scores such a trial (job 2026-05-05__05-48-18 has 204 timeouts), and
        the private board must count it too."""
        for error in ("AgentTimeoutError", "NonZeroAgentExitCodeError"):
            with self.subTest(error=error):
                self.assertTrue(
                    private_trial_finished(
                        _hidden_trial("t", "org/a-hidden", error_type=error)
                    )
                )

    def test_public_row_with_the_same_shape_is_not(self):
        """A public trial whose verifier produced nothing is excluded, not
        covered -- only the private dataset has no verdict by construction."""
        self.assertFalse(
            private_trial_finished(_hidden_trial("t", "org/a", source=PUBLIC.dataset))
        )

    def test_scored_row_is_not(self):
        self.assertFalse(private_trial_finished(_trial("t", "org/a", 1.0)))

    def test_missing_evals_is_not(self):
        """`evals` absent (not `{}`) is the errored-before-verifier shape."""
        row = _hidden_trial("t", "org/a-hidden")
        del row["evals"]
        self.assertFalse(private_trial_finished(row))


class SubmissionCoverageTests(unittest.TestCase):
    def test_public_coverage_is_the_scored_and_overridden_trials(self):
        trials = [
            _trial("t1", "org/a", 1.0),
            _trial("t2", "org/a", 0.0),
            _trial("t-err", "org/b", None, evals=None),
            _trial("t-cred", "org/c", None, evals=None),
        ]
        sub = {"credited_trials": [{"trial_id": "t-cred"}]}
        self.assertEqual(
            submission_coverage(trials, sub, PUBLIC), {"org/a": 2, "org/c": 1}
        )

    def test_private_finished_trials_cover_their_task(self):
        trials = [
            _hidden_trial("t1", "org/a-hidden"),
            _hidden_trial("t2", "org/a-hidden"),
            _hidden_trial("t3", "org/b-hidden"),
        ]
        self.assertEqual(
            submission_coverage(trials, {}, PRIVATE),
            {"org/a-hidden": 2, "org/b-hidden": 1},
        )
        # ... while submission_by_task still keeps none of them: no verdict.
        by_task, _, _ = submission_by_task(trials, {})
        self.assertEqual(by_task, {})

    def test_private_errored_trial_does_not_cover(self):
        trials = [
            _hidden_trial("t1", "org/a-hidden"),
            _hidden_trial("t-err", "org/b-hidden", error_type="RuntimeError"),
            _hidden_trial("t-to", "org/c-hidden", error_type="AgentTimeoutError"),
        ]
        self.assertEqual(
            submission_coverage(trials, {}, PRIVATE),
            {"org/a-hidden": 1, "org/c-hidden": 1},
        )

    def test_private_override_is_counted_once(self):
        """A credited/disqualified private trial is already in by_task; it
        must not be counted a second time as finished-unscored."""
        trials = [_hidden_trial("t1", "org/a-hidden")]
        sub = {"credited_trials": [{"trial_id": "t1"}]}
        self.assertEqual(
            submission_coverage(trials, sub, PRIVATE), {"org/a-hidden": 1}
        )

    def test_private_board_does_not_cover_public_shaped_rows(self):
        """The dataset guard: a row from another dataset with the empty shape
        is not coverage even when the board is private."""
        trials = [_hidden_trial("t1", "org/a", source=PUBLIC.dataset)]
        self.assertEqual(submission_coverage(trials, {}, PRIVATE), {})


def _score(tid: str, task: str, mode: str, **metrics) -> dict:
    """One /judge record as leaderboard-judge.yml commits it: build_rewards'
    keys (reward always; rca_accuracy / hallucinate_any when defined) plus the
    judge's mode. Nothing that names the answer."""
    return {"trial_id": tid, "task_name": task, "mode": mode, **metrics}


class JudgeScoresTests(unittest.TestCase):
    def setUp(self):
        self.trials = [
            _hidden_trial("t-inc", "org/a-hidden"),
            _hidden_trial("t-ctl", "org/b-hidden"),
            _hidden_trial("t-empty", "org/c-hidden", error_type="AgentTimeoutError"),
        ]
        self.scores = {
            "t-inc": _score("t-inc", "org/a-hidden", "llm_judge",
                            reward=0.67, rca_accuracy=1, hallucinate_any=0),
            # A control task: no root cause, so hallucinate_any is undefined
            # and absent -- exactly what build_rewards emits.
            "t-ctl": _score("t-ctl", "org/b-hidden", JUDGE_CONTROL_MODE,
                            reward=1.0, rca_accuracy=1),
            # An incident task whose agent wrote nothing: scored, not control.
            "t-empty": _score("t-empty", "org/c-hidden", "empty_report",
                              reward=0.0, rca_accuracy=0, hallucinate_any=0),
        }

    def test_full_coverage_passes(self):
        self.assertEqual(scores_coverage_failure(self.trials, self.scores), "")

    def test_missing_and_errored_records_are_named(self):
        scores = dict(self.scores)
        del scores["t-ctl"]
        scores["t-inc"] = _score("t-inc", "org/a-hidden", "llm_judge_error", reward=0.0)
        msg = scores_coverage_failure(self.trials, scores)
        self.assertIn("1 trial(s) have no judge score", msg)
        self.assertIn("1 the judge could not score", msg)
        self.assertIn("re-run /judge", msg)

    def test_judged_rows_read_like_verified_rows(self):
        """The judge record lands in `evals`, so trial_metric reads it by name
        exactly as it reads a public trial's verifier output."""
        judged = judged_trials(self.trials, self.scores)
        inc = next(t for t in judged if t["id"] == "t-inc")
        self.assertEqual(trial_metric(inc, "reward"), 0.67)
        self.assertEqual(trial_metric(inc, "rca_accuracy"), 1.0)
        self.assertEqual(trial_metric(inc, "hallucinate_any"), 0.0)
        self.assertEqual(inc["judge_mode"], "llm_judge")
        # Untouched row fields survive.
        self.assertEqual(inc["source"], PRIVATE.dataset)
        # Absent metric stays absent: a control row has no hallucinate_any.
        ctl = next(t for t in judged if t["id"] == "t-ctl")
        with self.assertRaises(Exception):
            trial_metric(ctl, "hallucinate_any")

    def test_errored_or_missing_record_leaves_the_row_unscored(self):
        scores = dict(self.scores)
        scores["t-inc"] = _score("t-inc", "org/a-hidden", "hub_fetch_error", reward=0.0)
        del scores["t-ctl"]
        judged = judged_trials(self.trials, scores)
        for tid in ("t-inc", "t-ctl"):
            row = next(t for t in judged if t["id"] == tid)
            self.assertEqual(row["evals"], {})
            self.assertNotIn("judge_mode", row)
            self.assertIsNone(trial_metric(row, "reward"))

    def test_inputs_are_not_mutated(self):
        judged_trials(self.trials, self.scores)
        self.assertEqual(self.trials[0]["evals"], {})

    def test_control_comes_from_the_judge_and_difficulty_from_the_task(self):
        base = {
            "org/a-hidden": TaskLabel(difficulty="hard", is_control=True),
            "org/b-hidden": TaskLabel(difficulty="easy", is_control=True),
            "org/c-hidden": TaskLabel(difficulty="medium", is_control=True),
        }
        with unittest.mock.patch("leaderboard.core.metrics.task_labels", return_value=base):
            labels = private_task_labels(PRIVATE, judged_trials(self.trials, self.scores))
        self.assertEqual(labels["org/a-hidden"], TaskLabel("hard", is_control=False))
        self.assertEqual(labels["org/b-hidden"], TaskLabel("easy", is_control=True))
        # empty_report is an incident trial that scored 0, not a control.
        self.assertEqual(labels["org/c-hidden"], TaskLabel("medium", is_control=False))

    def test_task_trials_disagreeing_on_control_raise(self):
        trials = [_hidden_trial("t1", "org/a-hidden"), _hidden_trial("t2", "org/a-hidden")]
        scores = {
            "t1": _score("t1", "org/a-hidden", "llm_judge", reward=1.0, rca_accuracy=1, hallucinate_any=0),
            "t2": _score("t2", "org/a-hidden", JUDGE_CONTROL_MODE, reward=1.0, rca_accuracy=1),
        }
        base = {"org/a-hidden": TaskLabel(difficulty="hard", is_control=True)}
        with unittest.mock.patch("leaderboard.core.metrics.task_labels", return_value=base):
            with self.assertRaises(TaskLabelError):
                private_task_labels(PRIVATE, judged_trials(trials, scores))

    def test_task_with_no_judged_trial_raises(self):
        trials = [_hidden_trial("t1", "org/a-hidden")]
        base = {"org/a-hidden": TaskLabel(difficulty="hard", is_control=True)}
        with unittest.mock.patch("leaderboard.core.metrics.task_labels", return_value=base):
            with self.assertRaises(TaskLabelError):
                private_task_labels(PRIVATE, judged_trials(trials, {}))


class FormatMeasurementTests(unittest.TestCase):
    """The cell format is user-visible, so pin it here rather than only
    asserting self-consistency."""

    def test_renders_percent_on_both_halves_with_one_decimal(self):
        self.assertEqual(format_measurement(83.8, 1.2), "**83.8%** ± 1.2%")

    def test_rounds_to_one_decimal(self):
        self.assertEqual(format_measurement(26.61, 3.004), "**26.6%** ± 3.0%")

    def test_pads_to_one_decimal(self):
        """A whole number must not collapse to `1%` -- the column reads as a
        measurement, so the precision stays visible."""
        self.assertEqual(format_measurement(1.0, 0.7), "**1.0%** ± 0.7%")

    def test_only_the_mean_is_bold(self):
        cell = format_measurement(9.0, 2.0)
        self.assertTrue(cell.startswith("**9.0%**"))
        self.assertEqual(cell.count("**"), 2)


class PublishedMetricsTests(unittest.TestCase):
    def test_each_metric_reports_mean_stderr_and_display(self):
        m = compute_subset_metrics(_trials(), {}, _LABELS)
        # One incident task per tier, so each RCA metric is a single value.
        self.assertEqual(m["rca_accuracy_incident_medium"], 100.0)
        self.assertEqual(m["rca_accuracy_incident_hard"], 0.0)
        # Hallucination spans all three incident tasks: (0 + 0 + 1) / 3.
        self.assertEqual(m["hallucinate_any_incident"], 33.33)
        for key in (
            "rca_accuracy_incident_medium",
            "rca_accuracy_incident_hard",
            "hallucinate_any_incident",
        ):
            self.assertIn(key + STDERR_SUFFIX, m)
            self.assertEqual(
                m[key + DISPLAY_SUFFIX],
                format_measurement(m[key], m[key + STDERR_SUFFIX]),
            )
            # The mean is bold so the eye lands on it, not the error bar.
            self.assertTrue(m[key + DISPLAY_SUFFIX].startswith("**"))

    def test_publishes_nothing_else(self):
        """The board is exactly three metrics; a stray key would be rejected by
        the hub schema at merge time."""
        m = compute_subset_metrics(_trials(), {}, _LABELS)
        expected = {
            k + suffix
            for k in (
                "rca_accuracy_incident_medium",
                "rca_accuracy_incident_hard",
                "hallucinate_any_incident",
            )
            for suffix in ("", STDERR_SUFFIX, DISPLAY_SUFFIX)
        }
        self.assertEqual(set(m), expected)

    def test_control_tasks_never_reach_any_metric(self):
        """All three metrics are incident-only. The control trial here carries
        no `hallucinate_any` at all, so a leak would raise rather than skew."""
        m = compute_subset_metrics(_trials(), {}, _LABELS)
        self.assertEqual(m["hallucinate_any_incident"], 33.33)  # 3 incident trials, not 4

    def test_control_task_difficulty_does_not_leak_into_a_tier(self):
        """The control task above is difficulty `hard`; it must not join the
        hard RCA metric."""
        m = compute_subset_metrics(_trials(), {}, _LABELS)
        self.assertEqual(m["rca_accuracy_incident_hard"], 0.0)

    def test_overrides_do_not_touch_published_metrics(self):
        """The override lists force a trial's *reward*, and no published metric
        is reward-derived any more."""
        sub = {"disqualified_trials": [{"trial_id": "t-medium", "reason": "spurious_pass"}]}
        self.assertEqual(
            compute_subset_metrics(_trials(), sub, _LABELS),
            compute_subset_metrics(_trials(), {}, _LABELS),
        )

    def test_unscored_trial_is_excluded(self):
        """Same rule as submission_by_task: no verdict, no contribution."""
        trials = _trials() + [{"id": "t-err", "task_name": "org/inc-medium", "reward": None}]
        m = compute_subset_metrics(trials, {}, _LABELS)
        self.assertEqual(m["rca_accuracy_incident_medium"], 100.0)

    def test_empty_subset_raises_rather_than_reporting_zero(self):
        """Full task coverage is enforced upstream, so an empty subset means
        the labels and the trials disagree."""
        trials = [t for t in _trials() if t["task_name"] != "org/inc-hard"]
        with self.assertRaises(EmptySubsetError):
            compute_subset_metrics(trials, {}, _LABELS)


class ComputeMetricsTests(unittest.TestCase):
    """ORCA-bench rewards are graded in [0, 1], so accuracy is the mean reward
    and the SE generalizes the binary formula."""

    def test_binary_rewards_match_the_classic_formula(self):
        """Regression guard: on pass/fail rewards the generalized estimators
        must reproduce accuracy = successes/total and
        s^2 = (1/n^2) * sum_i p_i (1-p_i) / (k_i - 1) exactly."""
        by_task = {
            "org/a": [1.0, 1.0, 0.0, 0.0],  # p = 0.5
            "org/b": [1.0, 1.0, 1.0, 1.0],  # p = 1.0
        }
        acc, se = compute_metrics(by_task)
        self.assertAlmostEqual(acc, 75.0)  # 6 of 8

        n, expected_var = 2, 0.0
        for p, k in ((0.5, 4), (1.0, 4)):
            expected_var += p * (1.0 - p) / (k - 1)
        expected_var /= n**2
        self.assertAlmostEqual(se, 100.0 * expected_var**0.5)

    def test_graded_rewards_average_rather_than_threshold(self):
        """A partial score must contribute its fraction, not a full success --
        the whole reason is_success is not the metric."""
        by_task = {"org/a": [1.0 / 3.0], "org/b": [2.0 / 3.0], "org/c": [1.0]}
        acc, _ = compute_metrics(by_task)
        self.assertAlmostEqual(acc, 100.0 * (2.0 / 3.0))

    def test_single_trial_per_task_uses_between_task_stderr(self):
        """With MIN_TRIALS_PER_TASK = 1 the within-task variance is undefined;
        the SE must still be meaningful rather than a misleading 0.00."""
        by_task = {"org/a": [0.0], "org/b": [1.0], "org/c": [0.5], "org/d": [1.0]}
        acc, se = compute_metrics(by_task)
        self.assertAlmostEqual(acc, 62.5)

        means = [0.0, 1.0, 0.5, 1.0]
        m = sum(means) / 4
        sample_var = sum((x - m) ** 2 for x in means) / 3
        self.assertAlmostEqual(se, 100.0 * (sample_var / 4) ** 0.5)
        self.assertGreater(se, 0.0)

    def test_none_reward_is_rejected(self):
        """submission_by_task drops unscored trials, so a None reaching
        compute_metrics means that filter was bypassed. Fail loudly rather than
        quietly reinstating the old "missing counts as 0" behaviour."""
        with self.assertRaises(RewardRangeError):
            compute_metrics({"org/a": [1.0, None]})

    def test_reward_outside_unit_interval_is_rejected(self):
        """A verifier emitting an un-normalized score must fail loudly rather
        than silently skew the leaderboard."""
        with self.assertRaises(RewardRangeError):
            compute_metrics({"org/a": [3.0]})
        with self.assertRaises(RewardRangeError):
            compute_metrics({"org/a": [-0.5]})

    def test_empty_submission_is_zero(self):
        self.assertEqual(compute_metrics({}), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
