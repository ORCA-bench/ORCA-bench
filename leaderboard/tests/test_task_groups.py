"""Unit tests for the per-task labels behind the subset metrics.

The labels come from each task's `task.toml` `[metadata]`. Two things are easy
to get wrong and both are pinned here: control tasks carry a difficulty (so the
difficulty subsets must intersect with "not control"), and the tasks carry a
second, rival tier ladder under `granularity`.
"""

from __future__ import annotations

import unittest

from leaderboard.core.task_groups import (
    DIFFICULTIES,
    TaskLabel,
    TaskLabelError,
    _label,
    check_difficulties,
)


class LabelTests(unittest.TestCase):
    def test_empty_events_is_a_control_task(self):
        self.assertTrue(_label({"events": [], "difficulty": "easy"}).is_control)

    def test_missing_events_is_a_control_task(self):
        self.assertTrue(_label({"difficulty": "easy"}).is_control)

    def test_non_empty_events_is_an_incident_task(self):
        self.assertFalse(
            _label({"events": [{"root_cause": "adFailure"}], "difficulty": "hard"}).is_control
        )

    def test_difficulty_is_read_verbatim(self):
        self.assertEqual(_label({"events": [], "difficulty": "medium"}).difficulty, "medium")

    def test_granularity_is_not_used(self):
        """The rival ladder must be ignored: this task is `medium` difficulty
        even though its granularity says `hard`."""
        label = _label({"events": [1], "difficulty": "medium", "granularity": "hard"})
        self.assertEqual(label.difficulty, "medium")

    def test_control_tasks_still_carry_a_difficulty(self):
        """Which is why difficulty subsets intersect with `not is_control`
        rather than partitioning on difficulty alone."""
        label = _label({"events": [], "difficulty": "hard"})
        self.assertTrue(label.is_control)
        self.assertEqual(label.difficulty, "hard")


class DifficultyGuardTests(unittest.TestCase):
    def _labels(self, difficulties):
        return {f"orca-bench/t{i}": TaskLabel(d, False) for i, d in enumerate(difficulties)}

    def test_expected_ladder_passes(self):
        check_difficulties(self._labels(DIFFICULTIES))  # must not raise

    def test_granularity_ladder_is_rejected(self):
        """Reading `granularity` by mistake yields easy/hard/universal, which
        overlaps the real ladder on two of three labels -- the guard exists to
        catch exactly this."""
        with self.assertRaises(TaskLabelError) as ctx:
            check_difficulties(self._labels(("easy", "hard", "universal")))
        self.assertIn("granularity", str(ctx.exception))

    def test_missing_tier_is_rejected(self):
        with self.assertRaises(TaskLabelError):
            check_difficulties(self._labels(("easy", "hard")))

    def test_unlabelled_task_is_rejected(self):
        with self.assertRaises(TaskLabelError):
            check_difficulties(self._labels(("easy", "medium", "hard", None)))


if __name__ == "__main__":
    unittest.main()
