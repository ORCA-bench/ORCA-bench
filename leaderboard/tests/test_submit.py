"""Unit tests for Hub metadata whitelisting in ci.submit."""

from __future__ import annotations

import unittest

from leaderboard.ci.submit import hub_metadata


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


if __name__ == "__main__":
    unittest.main()
