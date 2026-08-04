"""Unit tests for the reward override join and the graded-reward metric."""

from __future__ import annotations

import unittest

from leaderboard.core.metrics import (
    EmptySubsetError,
    RewardRangeError,
    compute_metrics,
    compute_subset_metrics,
    stderr_basis,
    submission_by_task,
)
from leaderboard.core.task_groups import TaskLabel


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

    def test_errored_trial_without_evals_still_scores_zero(self):
        """A trial that produced no evals at all is an errored trial, not a
        contract violation: it stays None and counts as 0."""
        trials = [_trial("t-err", "org/a", None, evals=None)]
        by_task, _, _ = submission_by_task(trials, {})
        self.assertEqual(by_task["org/a"], [None])
        acc, _ = compute_metrics(by_task)
        self.assertEqual(acc, 0.0)

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


def _scored(tid: str, task: str, reward, rca) -> dict:
    """A trial row carrying both eval metrics, as the verifier publishes them."""
    return {
        "id": tid,
        "task_name": task,
        "reward": reward,
        "evals": {
            "reward": {"metrics": [{"reward": reward}]},
            "rca_accuracy": {"metrics": [{"rca_accuracy": rca}]},
        },
    }


# One task per (kind, difficulty) cell, so every subset is populated.
_LABELS = {
    "org/inc-easy": TaskLabel("easy", False),
    "org/inc-medium": TaskLabel("medium", False),
    "org/inc-hard": TaskLabel("hard", False),
    "org/ctl": TaskLabel("hard", True),
}


def _trials() -> list[dict]:
    return [
        _scored("t-easy", "org/inc-easy", 1.0, 1),
        _scored("t-medium", "org/inc-medium", 0.5, 0),
        _scored("t-hard", "org/inc-hard", 0.0, 0),
        _scored("t-ctl", "org/ctl", 1.0, 0),
    ]


class SubsetMetricsTests(unittest.TestCase):
    def test_every_subset_is_reported(self):
        m = compute_subset_metrics(_trials(), {}, _LABELS)
        self.assertEqual(m["accuracy_incident"], 50.0)  # (1.0 + 0.5 + 0.0) / 3
        self.assertEqual(m["accuracy_control"], 100.0)
        self.assertEqual(m["accuracy_incident_easy"], 100.0)
        self.assertEqual(m["accuracy_incident_medium"], 50.0)
        self.assertEqual(m["accuracy_incident_hard"], 0.0)
        self.assertEqual(m["rca_accuracy_incident"], round(100 / 3, 2))
        self.assertEqual(m["rca_accuracy_incident_easy"], 100.0)
        self.assertEqual(m["rca_accuracy_incident_medium"], 0.0)
        self.assertEqual(m["rca_accuracy_incident_hard"], 0.0)

    def test_rca_accuracy_has_no_control_subset(self):
        """It is 0 by construction on control tasks, so the column would be a
        constant; the reward breakdown does have one."""
        m = compute_subset_metrics(_trials(), {}, _LABELS)
        self.assertNotIn("rca_accuracy_control", m)
        self.assertIn("accuracy_control", m)

    def test_control_tasks_are_excluded_from_difficulty_subsets(self):
        """The control task above is difficulty `hard`; it must not leak into
        accuracy_incident_hard."""
        m = compute_subset_metrics(_trials(), {}, _LABELS)
        self.assertEqual(m["accuracy_incident_hard"], 0.0)

    def test_overrides_move_reward_subsets_only(self):
        """A disqualified trial is forced to reward 0, but its rca_accuracy
        verdict is left alone -- the override says the score was wrong, not
        that the root cause went unnamed."""
        sub = {"disqualified_trials": [{"trial_id": "t-easy", "reason": "spurious_pass"}]}
        m = compute_subset_metrics(_trials(), sub, _LABELS)
        self.assertEqual(m["accuracy_incident_easy"], 0.0)
        self.assertEqual(m["rca_accuracy_incident_easy"], 100.0)

    def test_credited_trial_is_forced_to_full_reward(self):
        sub = {"credited_trials": [{"trial_id": "t-hard", "reason": "spurious_fail"}]}
        m = compute_subset_metrics(_trials(), sub, _LABELS)
        self.assertEqual(m["accuracy_incident_hard"], 100.0)

    def test_empty_subset_raises_rather_than_reporting_zero(self):
        """Full task coverage is enforced upstream, so an empty subset means
        the labels and the trials disagree."""
        trials = [t for t in _trials() if t["task_name"] != "org/ctl"]
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
        self.assertEqual(stderr_basis(by_task), "within-task")

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
        self.assertEqual(stderr_basis(by_task), "between-task")

    def test_missing_reward_counts_as_zero(self):
        """Errored trials have no reward and are not excluded from the metric."""
        acc, _ = compute_metrics({"org/a": [1.0, None]})
        self.assertAlmostEqual(acc, 50.0)

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
