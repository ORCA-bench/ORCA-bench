"""Unit tests for the job-config checks in ci.static_analysis.

These pin two behaviours discovered against a real submission (PR #2), where
harbor recorded a config that the checks ported from harbor-index rejected even
though the run was clean.
"""

from __future__ import annotations

import unittest

from leaderboard.core.hub import PRIVATE, PUBLIC
from leaderboard.ci.static_analysis import (
    dataset_entry_failure,
    timeout_multiplier_ok,
    trial_config_failures,
)


class TimeoutMultiplierTests(unittest.TestCase):
    """harbor omits the key entirely when it was never set, and unset == 1.0."""

    def test_explicit_one_is_ok(self):
        self.assertTrue(timeout_multiplier_ok({"timeout_multiplier": 1.0}))

    def test_none_is_ok(self):
        self.assertTrue(timeout_multiplier_ok({"timeout_multiplier": None}))

    def test_absent_is_ok(self):
        self.assertTrue(timeout_multiplier_ok({}))

    def test_scaled_is_rejected(self):
        self.assertFalse(timeout_multiplier_ok({"timeout_multiplier": 2.0}))
        self.assertFalse(timeout_multiplier_ok({"timeout_multiplier": 0.5}))

    def test_per_trial_accepts_default_but_still_catches_overrides(self):
        """Accepting None must not weaken the override checks beside it."""
        self.assertEqual(trial_config_failures({"timeout_multiplier": None}), [])
        fails = trial_config_failures(
            {"timeout_multiplier": None, "agent": {"override_timeout_sec": 9000}}
        )
        self.assertEqual(fails, ["agent.override_timeout_sec=9000"])


class DatasetEntryTests(unittest.TestCase):
    """The recorded config names its datasets but does not always carry a
    resolved ref; version pinning then falls to the task-digest check."""

    def test_name_only_passes(self):
        cfg = {"datasets": [{"name": PUBLIC.dataset, "task_names": ["*"]}]}
        self.assertEqual(dataset_entry_failure(cfg, PUBLIC), "")

    def test_matching_ref_passes(self):
        cfg = {"datasets": [{"name": PUBLIC.dataset, "ref": PUBLIC.ref}]}
        self.assertEqual(dataset_entry_failure(cfg, PUBLIC), "")

    def test_wrong_ref_is_rejected(self):
        cfg = {"datasets": [{"name": PUBLIC.dataset, "ref": "sha256:deadbeef"}]}
        self.assertEqual(dataset_entry_failure(cfg, PUBLIC), "ref mismatch")

    def test_other_dataset_is_rejected(self):
        cfg = {"datasets": [{"name": "someone-else/bench", "ref": PUBLIC.ref}]}
        self.assertEqual(
            dataset_entry_failure(cfg, PUBLIC), f"did not run {PUBLIC.dataset}"
        )

    def test_no_datasets_is_rejected(self):
        self.assertEqual(
            dataset_entry_failure({}, PUBLIC), f"did not run {PUBLIC.dataset}"
        )

    def test_our_dataset_alongside_others_passes(self):
        cfg = {
            "datasets": [
                {"name": "someone-else/bench"},
                {"name": PUBLIC.dataset, "task_names": ["*"]},
            ]
        }
        self.assertEqual(dataset_entry_failure(cfg, PUBLIC), "")

    def test_boards_are_checked_against_their_own_dataset(self):
        """A private-split job passes the private board's check and fails the
        public board's, and vice versa -- the two datasets are disjoint by
        construction, so a run on one can never be a submission to the other."""
        private_cfg = {"datasets": [{"name": PRIVATE.dataset, "ref": PRIVATE.ref}]}
        public_cfg = {"datasets": [{"name": PUBLIC.dataset, "ref": PUBLIC.ref}]}
        self.assertEqual(dataset_entry_failure(private_cfg, PRIVATE), "")
        self.assertEqual(
            dataset_entry_failure(private_cfg, PUBLIC), f"did not run {PUBLIC.dataset}"
        )
        self.assertEqual(
            dataset_entry_failure(public_cfg, PRIVATE), f"did not run {PRIVATE.dataset}"
        )

    def test_private_ref_is_pinned(self):
        cfg = {"datasets": [{"name": PRIVATE.dataset, "ref": "sha256:deadbeef"}]}
        self.assertEqual(dataset_entry_failure(cfg, PRIVATE), "ref mismatch")


if __name__ == "__main__":
    unittest.main()
