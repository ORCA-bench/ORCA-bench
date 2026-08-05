"""hub: shared low-level access to the Harbor Hub + registry.

Constants and I/O helpers used by both the submission generator (filter) and
the static checker (static_analysis):

  * hub_json / hub_job_trials -- the `harbor hub` CLI (job show, job trials)
  * dataset_task_digests      -- the canonical per-task sha256 digests for
                                 DATASET@DATASET_REF, fetched from the registry
                                 at runtime (a metadata query, no task content)
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# The dataset this leaderboard repo tracks. Filtering is anchored to it: a job
# that did not run this dataset is rejected, so trials from other datasets are
# never considered. DATASET_REF is the sha256 of the canonical dataset version;
# checking it proves every trial ran the official, unmodified tasks.
#
# NOTE: submissions must run the dataset as published on the Harbor Hub. The
# HuggingFace-registry config in configs/harbor_job_orca_bench.yaml is for
# local development only — its trials carry a different `source` and are
# filtered out by submission_trials below.
#
# Re-pin after republishing the dataset (see leaderboard/SETUP.md):
#
#     uv run python -c "import asyncio; \
#       from harbor.registry.client.package import PackageDatasetClient; \
#       print(asyncio.run(PackageDatasetClient().get_dataset_metadata( \
#         'orca-bench/orca-bench@latest')).version)"
#
# The dataset is currently private, so this needs an authenticated harbor
# session (`harbor auth login`) or HARBOR_API_KEY -- an anonymous caller sees
# "Tag 'latest' not found", which reads as "no such dataset" but is not.
DATASET = "orca-bench/orca-bench"
DATASET_REF = "sha256:2add497ac2f93468dba1b83977c295a784c89031754e6a35520195345ad62ada"

# Sentinel guarding an unpinned checkout; DATASET_REF above is a real ref, so
# this only fires if someone blanks it out.
_REF_PLACEHOLDER = "sha256:TODO-PIN-ORCA-BENCH-DATASET-REF"

# Base URL for human-facing hub links (trial pages, the leaderboard). API
# access does not go through this -- harbor resolves its own Supabase URL.
HUB_URL = "https://hub.harborframework.com"


def job_uuid(link: str) -> str:
    """Extract the job UUID from a hub link or a bare UUID."""
    return link.rstrip("/").split("/")[-1]


def hub_json(cmd_args: list[str]) -> dict:
    """Run `harbor hub <args> --json` and return the parsed response."""
    try:
        proc = subprocess.run(
            ["harbor", "hub", *cmd_args, "--json"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        sys.exit("`harbor` CLI not found; install it to query the Hub.")
    if proc.returncode != 0:
        sys.exit(f"`harbor hub {' '.join(cmd_args)}` failed:\n{proc.stderr.strip()}")
    # harbor colorizes --json output even when captured; strip ANSI codes.
    return json.loads(_ANSI.sub("", proc.stdout))


def hub_job_trials(uuid: str, page_size: int = 500) -> list[dict]:
    """Page through the latest-attempt trials of a job (metadata only)."""
    items: list[dict] = []
    page = 1
    while True:
        d = hub_json(
            ["job", "trials", uuid, "--limit", str(page_size), "--page", str(page)]
        )
        items.extend(d.get("items", []))
        if page >= d.get("total_pages", 1):
            break
        page += 1
    return items


def trial_model(trial: dict) -> str | None:
    """Full 'provider/model' id for a bulk trial row. The rows store the
    provider and bare model name split; job configs and submissions use the
    joined id (e.g. 'openai/gpt-5.5'), so join here for exact comparison.
    A missing or 'unknown' provider (an agent that didn't report one) joins
    to the bare model name, matching configs that name the model without a
    provider (e.g. 'glm-5.1')."""
    provider, name = trial.get("model_provider"), trial.get("model_name")
    if not provider or provider == "unknown":
        return name
    return f"{provider}/{name}"


# The eval metric a trial's score is read from. ORCA-bench verifiers publish it
# alongside companion diagnostics (see check_prediction.build_rewards); the
# subset metrics also read `rca_accuracy`, which the verifier emits on every
# trial, control tasks included.
REWARD_METRIC = "reward"
RCA_ACCURACY_METRIC = "rca_accuracy"
# Emitted on incident trials only -- a control task has no root cause to
# hallucinate about, so the verifier omits it there (check_prediction
# .aggregate_judge_response). Only ever read over incident tasks.
HALLUCINATE_METRIC = "hallucinate_any"


class MissingRewardMetricError(ValueError):
    """A trial reported eval metrics, but none of them has the wanted name."""


def _eval_metric_maps(evals: dict) -> list[dict]:
    """Every {metric_name: value} map in an `evals` block, in Hub order.

    Tolerates both shapes harbor emits (cf. harbor.hub.models.primary_reward):
    the nested `{name: {"metrics": [{name: value}]}}` that bulk trial rows
    carry, and the row-oriented `{"group_by": [...], "rows": [{"metrics": [...]}]}`.
    Non-dict values (e.g. the `group_by` list) are skipped, not an error.
    """
    rows = evals.get("rows")
    containers = rows if isinstance(rows, list) else evals.values()
    return [
        m
        for c in containers
        if isinstance(c, dict)
        for m in (c.get("metrics") or [])
        if isinstance(m, dict)
    ]


def trial_metric(trial: dict, metric: str = REWARD_METRIC) -> float | None:
    """The eval metric NAMED `metric` for a bulk trial row.

    Read by name, never positionally. The Hub derives a row's own `reward`
    scalar from the *alphabetically first* numeric entry of the verifier's
    rewards map (harbor.hub.models._first_numeric_reward), so a verifier that
    publishes several metrics can have a companion metric published as the
    trial's score. That is not hypothetical: `hallucinate_any` sorts ahead of
    `reward`, so the hallucination flag became the trial reward and this
    leaderboard computed 12.05% where the true accuracy was 49.43%.

    Returns None only for a trial that produced no evals at all -- a genuinely
    errored trial -- which metrics._reward maps to 0.0, as intended. A trial
    that reported metrics but not this one raises instead of scoring 0: that
    means the verifier contract changed, which skews the whole submission
    uniformly, and a silently zeroed metric is the exact failure this function
    exists to prevent. The ORCA-bench verifier emits both `reward` and
    `rca_accuracy` on every trial, so neither is ever legitimately absent.
    """
    evals = trial.get("evals")
    if not isinstance(evals, dict) or not evals:
        if trial.get("reward") is not None:
            raise MissingRewardMetricError(
                f"trial {trial.get('id')} carries a reward scalar "
                f"{trial.get('reward')!r} but no `evals` block; the Hub row "
                "shape changed and reading a metric by name would silently "
                "score every trial 0. Refusing to compute a zeroed metric."
            )
        return None
    for metrics in _eval_metric_maps(evals):
        if metric in metrics:
            value = metrics[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MissingRewardMetricError(
                    f"trial {trial.get('id')} reports a non-numeric "
                    f"`{metric}` metric {value!r}"
                )
            return float(value)
    found = sorted({k for m in _eval_metric_maps(evals) for k in m})
    raise MissingRewardMetricError(
        f"trial {trial.get('id')} reports eval metrics {found} but none named "
        f"`{metric}`; ORCA-bench verifiers must emit it (see "
        "check_prediction.build_rewards). Refusing to score the trial 0."
    )


def trial_reward(trial: dict) -> float | None:
    """The metric named `reward` -- the trial's graded score."""
    return trial_metric(trial, REWARD_METRIC)


def submission_trials(submission: dict) -> list[dict]:
    """Bulk trial metadata for a submission: every latest-attempt trial of its
    source_jobs on the dataset matching the submission's source_filter (agent,
    agent version, model). Shared by static_analysis (checks + metrics) and
    clone_trials, so what is checked is exactly what gets cloned."""
    sf = submission["source_filter"]
    return [
        t
        for link in submission["source_jobs"]
        for t in hub_job_trials(job_uuid(link))
        if t.get("source") == DATASET
        and t.get("agent_name") == sf["agent"]
        and t.get("agent_version") == sf["agent_version"]
        and trial_model(t) == sf["model_name"]
    ]


def dataset_task_digests(dataset: str = DATASET, ref: str = DATASET_REF) -> dict[str, str]:
    """Canonical {task_name: sha256 ref} for the dataset version, from the
    registry. A metadata-only query (no task content downloaded)."""
    if ref == _REF_PLACEHOLDER:
        sys.exit(
            "DATASET_REF is still the placeholder in leaderboard/core/hub.py.\n"
            "Pin the published dataset version before running any check -- see\n"
            'leaderboard/SETUP.md, "Pinning DATASET_REF".'
        )

    async def _fetch() -> dict[str, str]:
        from harbor.registry.client.package import PackageDatasetClient

        meta = await PackageDatasetClient().get_dataset_metadata(f"{dataset}@{ref}")
        return {f"{t.org}/{t.name}": t.ref for t in meta.task_ids}

    return asyncio.run(_fetch())
