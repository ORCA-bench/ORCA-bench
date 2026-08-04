"""Unit tests for Hub metadata whitelisting in ci.submit."""

from __future__ import annotations

import unittest

import re

from leaderboard.ci.submit import (
    LEADERBOARD_NAME,
    LEADERBOARD_PACKAGE,
    hub_metadata,
    row_create_payload,
)


class HubMetadataTests(unittest.TestCase):
    def test_keeps_allowed_keys_drops_audit_and_nulls(self):
        submission = {
            "metadata": {
                "agent_display": "Terminus 2",
                "model_display": "Claude Opus 4.7",
                "agent_org": "Harbor",
                "model_org": "Anthropic",
                "date": "2026-07-14",
                "reasoning_effort": None,  # optional; null must not be sent
                "pr": "[#72](https://example.com/72)",
                "notes": "audit only",
                "rerun_job": "job-xyz",
                "prior_hub_copy": "old",
            }
        }
        got = hub_metadata(submission)
        self.assertEqual(
            got,
            {
                "agent_display": "Terminus 2",
                "model_display": "Claude Opus 4.7",
                "agent_org": "Harbor",
                "model_org": "Anthropic",
                "date": "2026-07-14",
                "pr": "[#72](https://example.com/72)",
            },
        )
        self.assertNotIn("notes", got)
        self.assertNotIn("rerun_job", got)
        self.assertNotIn("reasoning_effort", got)

    def test_drops_judge_key(self):
        """ORCA-bench has no trajectory judge, so `judge` is not in the hub
        leaderboard's metadata_schema (additionalProperties: false) -- a
        leftover value must be stripped rather than rejected by the API."""
        submission = {
            "metadata": {
                "agent_display": "Terminus 2",
                "model_display": "GLM-5",
                "agent_org": "Harbor",
                "model_org": "Z.ai",
                "date": "2026-07-14",
                "judge": "[Job](https://example.com/job)",
            }
        }
        self.assertNotIn("judge", hub_metadata(submission))

    def test_keeps_reasoning_effort_when_set(self):
        submission = {
            "metadata": {
                "agent_display": "Terminus 2",
                "model_display": "GPT-5.5",
                "agent_org": "Harbor",
                "model_org": "OpenAI",
                "date": "2026-07-14",
                "reasoning_effort": "medium",
            }
        }
        self.assertEqual(hub_metadata(submission)["reasoning_effort"], "medium")


class RowCreatePayloadTests(unittest.TestCase):
    """The row-create API selects the package by its slug. The slug must stay
    lowercase to be accepted -- see #3, where the old uppercase `ORCA-bench`
    slug produced HTTP 400 after the PR had already merged."""

    # The pattern the hub enforces on a `package` selector.
    HUB_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_.-]*$")

    def _submission(self) -> dict:
        return {
            "metadata": {
                "agent_display": "Terminus 2",
                "model_display": "Claude Sonnet 4.6",
                "agent_org": "Harbor",
                "model_org": "Anthropic",
                "date": "2026-05-05",
            },
            "metrics": {"accuracy": 72.72, "accuracy_stderr": 1.62},
            "trials": ["trial-a", "trial-b"],
        }

    def test_display_slug_is_accepted_by_the_hub(self):
        """The slug is the API selector, so renaming the package to anything
        the hub's lowercase-only pattern rejects breaks every submission."""
        self.assertIsNotNone(self.HUB_SLUG_RE.match(LEADERBOARD_PACKAGE))

    def test_payload_selects_by_package_slug(self):
        payload = row_create_payload(self._submission())
        self.assertEqual(payload["package"], LEADERBOARD_PACKAGE)
        self.assertEqual(payload["name"], LEADERBOARD_NAME)
        self.assertNotIn("package_id", payload)

    def test_payload_carries_row_metrics_and_trials(self):
        payload = row_create_payload(self._submission())
        (row,) = payload["rows"]
        self.assertEqual(row["metrics"]["accuracy"], 72.72)
        self.assertEqual(row["trial_ids"], ["trial-a", "trial-b"])
        self.assertEqual(row["status"], "display")
        self.assertNotIn("notes", row["metadata"])


if __name__ == "__main__":
    unittest.main()
