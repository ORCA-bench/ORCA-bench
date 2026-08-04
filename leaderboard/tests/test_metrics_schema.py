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

from leaderboard.core.metrics import DISPLAY_SUFFIX, METRICS, STDERR_SUFFIX

LEADERBOARD_JSON = Path(__file__).resolve().parents[1] / "leaderboard.json"


def _definition() -> dict:
    return json.loads(LEADERBOARD_JSON.read_text())


def _expected_metric_keys() -> set[str]:
    """Every key compute_submission_metrics can emit: three per metric spec,
    plus the optional resource totals."""
    keys = {"total_tokens", "total_cost_usd"}
    for spec in METRICS:
        keys |= {spec.key, spec.key + STDERR_SUFFIX, spec.key + DISPLAY_SUFFIX}
    return keys


def _display_keys() -> set[str]:
    """The only keys that get a column; means and stderrs are stored but not
    shown."""
    return {spec.key + DISPLAY_SUFFIX for spec in METRICS}


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

    def _metric_columns(self) -> set[str]:
        return {
            c["accessor"].split(".", 1)[1]
            for c in _definition()["columns"]
            if c["accessor"].startswith("metrics.")
        }

    def test_every_display_metric_has_a_column(self):
        missing = _display_keys() - self._metric_columns()
        self.assertFalse(missing, f"no leaderboard column for {sorted(missing)}")

    def test_columns_show_only_display_and_resource_metrics(self):
        """Means and stderrs are stored for machines, not shown: a column on a
        raw mean would render a bare number beside the mean ± se cell."""
        extra = self._metric_columns() - _display_keys() - {"total_tokens", "total_cost_usd"}
        self.assertFalse(extra, f"unexpected metric columns {sorted(extra)}")

    def test_ranking_uses_a_numeric_metric(self):
        """The Hub compares strings lexicographically, so ranking on a display
        value would sort "9.5" above "26.6"."""
        props = _definition()["metrics_schema"]["properties"]
        for rule in _definition()["rank_by"]:
            key = rule["accessor"].split(".", 1)[1]
            self.assertEqual(
                props[key]["type"], "number", f"rank_by {key} is not numeric"
            )

    def test_a_representative_metrics_dict_validates(self):
        metrics = {k: 12.34 for k in _expected_metric_keys()}
        metrics.update({k: "12.34 ± 1.00" for k in _display_keys()})
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
