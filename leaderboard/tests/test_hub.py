"""Unit tests for reading a trial's reward out of its eval metrics.

The Hub derives a bulk trial row's `reward` scalar positionally -- the
alphabetically first numeric entry of the verifier's rewards map -- so it can
publish a companion metric as the trial's score. `trial_reward` reads the
metric by name instead. See core.hub.trial_reward.
"""

from __future__ import annotations

import unittest

from leaderboard.core.hub import (
    MissingRewardMetricError,
    trial_metric,
    trial_reward,
)


def _row(evals, reward=None) -> dict:
    row = {"id": "t1", "task_name": "org/a", "reward": reward}
    if evals is not None:
        row["evals"] = evals
    return row


class TrialRewardTests(unittest.TestCase):
    def test_reads_the_metric_named_reward_from_nested_evals(self):
        """The shape bulk trial rows carry, with companion metrics alongside."""
        row = _row(
            {
                "reward": {"metrics": [{"reward": 0.75}]},
                "rca_accuracy": {"metrics": [{"rca_accuracy": 1}]},
                "hallucinate_any": {"metrics": [{"hallucinate_any": 0}]},
            },
            reward=0,
        )
        self.assertEqual(trial_reward(row), 0.75)

    def test_reads_the_metric_named_reward_from_row_oriented_evals(self):
        """The other shape harbor emits; `group_by` is not a metrics container
        and must be skipped rather than crash."""
        row = _row(
            {
                "group_by": ["dataset", "agent"],
                "rows": [
                    {
                        "metrics": [
                            {"hallucinate_any": 0},
                            {"reward": 0.5},
                        ]
                    }
                ],
            }
        )
        self.assertEqual(trial_reward(row), 0.5)

    def test_legacy_single_metric_evals(self):
        """Jobs uploaded before the verifier published companion metrics."""
        self.assertEqual(
            trial_reward(_row({"reward": {"metrics": [{"reward": 1.0}]}}, reward=1.0)),
            1.0,
        )

    def test_row_scalar_is_ignored_when_it_disagrees(self):
        row = _row({"reward": {"metrics": [{"reward": 1.0}]}}, reward=0)
        self.assertEqual(trial_reward(row), 1.0)

    def test_no_evals_and_no_scalar_is_none(self):
        """A genuinely errored trial: metrics._reward maps None to 0.0."""
        self.assertIsNone(trial_reward(_row(None)))

    def test_zero_reward_is_preserved_not_confused_with_missing(self):
        self.assertEqual(trial_reward(_row({"reward": {"metrics": [{"reward": 0}]}})), 0.0)

    def test_evals_without_a_reward_metric_raises(self):
        row = _row(
            {
                "rca_accuracy": {"metrics": [{"rca_accuracy": 1}]},
                "hallucinate_any": {"metrics": [{"hallucinate_any": 0}]},
            }
        )
        with self.assertRaises(MissingRewardMetricError) as ctx:
            trial_reward(row)
        # The message must name what it did find, so the contract break is obvious.
        self.assertIn("hallucinate_any", str(ctx.exception))
        self.assertIn("rca_accuracy", str(ctx.exception))

    def test_missing_evals_with_a_reward_scalar_raises(self):
        """Schema-drift guard: a row carrying a scalar but no evals would zero
        every trial silently."""
        with self.assertRaises(MissingRewardMetricError):
            trial_reward(_row(None, reward=0.5))

    def test_empty_evals_with_a_reward_scalar_raises(self):
        with self.assertRaises(MissingRewardMetricError):
            trial_reward(_row({}, reward=0.5))

    def test_non_numeric_reward_metric_raises(self):
        with self.assertRaises(MissingRewardMetricError):
            trial_reward(_row({"reward": {"metrics": [{"reward": None}]}}))

    def test_boolean_reward_metric_raises(self):
        """A bool is an int in Python; it is not a graded reward."""
        with self.assertRaises(MissingRewardMetricError):
            trial_reward(_row({"reward": {"metrics": [{"reward": True}]}}))


class TrialMetricTests(unittest.TestCase):
    """`trial_metric` is `trial_reward` with the metric name parameterized; the
    subset metrics use it to read `rca_accuracy` the same way."""

    def _row_with_both(self, reward, rca) -> dict:
        return _row(
            {
                "reward": {"metrics": [{"reward": reward}]},
                "rca_accuracy": {"metrics": [{"rca_accuracy": rca}]},
                "hallucinate_any": {"metrics": [{"hallucinate_any": 0}]},
            },
            reward=0,
        )

    def test_reads_the_named_companion_metric(self):
        row = self._row_with_both(0.75, 1)
        self.assertEqual(trial_metric(row, "rca_accuracy"), 1.0)
        self.assertEqual(trial_metric(row, "reward"), 0.75)

    def test_defaults_to_reward(self):
        self.assertEqual(trial_metric(self._row_with_both(0.75, 1)), 0.75)
        self.assertEqual(trial_reward(self._row_with_both(0.75, 1)), 0.75)

    def test_missing_companion_metric_raises(self):
        """The verifier emits rca_accuracy on every trial, control tasks
        included, so its absence means the contract broke."""
        row = _row({"reward": {"metrics": [{"reward": 1.0}]}}, reward=1.0)
        with self.assertRaises(MissingRewardMetricError) as ctx:
            trial_metric(row, "rca_accuracy")
        self.assertIn("rca_accuracy", str(ctx.exception))

    def test_errored_trial_is_none_for_any_metric(self):
        self.assertIsNone(trial_metric(_row(None), "rca_accuracy"))


if __name__ == "__main__":
    unittest.main()
