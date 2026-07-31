"""Unit tests for the reward override join and the graded-reward metric."""

from __future__ import annotations

import unittest

from leaderboard.core.metrics import (
    RewardRangeError,
    compute_metrics,
    stderr_basis,
    submission_by_task,
)


def _trial(tid: str, task: str, reward) -> dict:
    return {"id": tid, "task_name": task, "reward": reward}


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
