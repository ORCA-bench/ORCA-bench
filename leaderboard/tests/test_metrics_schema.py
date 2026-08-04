"""The metrics dict must satisfy the checked-in leaderboard metrics_schema.

`ci/submit.py` passes `metrics` to `leaderboard-row-create` verbatim -- there is
no key whitelist -- and the hub schema sets `additionalProperties: false`. A key
the live schema doesn't know is therefore rejected at *merge* time, after the PR
has already merged. This test is the cheap guard against that: if the computed
metrics and the schema drift apart, it fails here instead.

It also checks the reverse direction -- every metric the code can emit has a
column -- so a new metric can't land invisibly on the board.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from leaderboard.core.metrics import SUBSET_METRICS, SUBSETS

LEADERBOARD_JSON = Path(__file__).resolve().parents[1] / "leaderboard.json"


def _definition() -> dict:
    return json.loads(LEADERBOARD_JSON.read_text())


def _expected_metric_keys() -> set[str]:
    """Every key compute_submission_metrics can emit."""
    keys = {"accuracy", "accuracy_stderr", "total_tokens", "total_cost_usd"}
    for prefix, _metric, _header, incident_only in SUBSET_METRICS:
        for suffix, _label, _predicate in SUBSETS:
            if incident_only and suffix == "control":
                continue
            keys.add(f"{prefix}_{suffix}")
    return keys


class MetricsSchemaTests(unittest.TestCase):
    def test_schema_is_valid(self):
        Draft202012Validator.check_schema(_definition()["metrics_schema"])

    def test_every_computed_metric_is_in_the_schema(self):
        allowed = set(_definition()["metrics_schema"]["properties"])
        missing = _expected_metric_keys() - allowed
        self.assertFalse(
            missing,
            f"metrics_schema is missing {sorted(missing)}; a row carrying these "
            "would be rejected by leaderboard-row-create after the PR merged.",
        )

    def test_schema_declares_no_metric_the_code_never_emits(self):
        allowed = set(_definition()["metrics_schema"]["properties"])
        extra = allowed - _expected_metric_keys()
        self.assertFalse(extra, f"metrics_schema declares unused keys {sorted(extra)}")

    def test_every_metric_has_a_column(self):
        columns = {
            c["accessor"].split(".", 1)[1]
            for c in _definition()["columns"]
            if c["accessor"].startswith("metrics.")
        }
        missing = _expected_metric_keys() - columns
        self.assertFalse(missing, f"no leaderboard column for {sorted(missing)}")

    def test_a_representative_metrics_dict_validates(self):
        metrics = {k: 12.34 for k in _expected_metric_keys()}
        metrics["total_tokens"] = 1_126_726_966
        metrics["total_cost_usd"] = 42.5
        Draft202012Validator(_definition()["metrics_schema"]).validate(metrics)

    def test_an_unknown_metric_is_rejected(self):
        """Proves additionalProperties: false is doing its job, so the test
        above is meaningful."""
        validator = Draft202012Validator(_definition()["metrics_schema"])
        with self.assertRaises(Exception):
            validator.validate({"accuracy": 1.0, "accuracy_stderr": 0.1, "nope": 1})


if __name__ == "__main__":
    unittest.main()
