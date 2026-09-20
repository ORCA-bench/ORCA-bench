"""hub: shared low-level access to the Harbor Hub + registry.

Constants and I/O helpers used by both the submission generator (filter) and
the static checker (static_analysis):

  * Board / BOARDS            -- the leaderboards this repo submits to, each
                                 pinned to one dataset version
  * hub_json / hub_job_trials -- the `harbor hub` CLI (job show, job trials)
  * dataset_task_digests      -- the canonical per-task sha256 digests for a
                                 board's dataset@ref, fetched from the registry
                                 at runtime (a metadata query, no task content)
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from dataclasses import dataclass

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Base URL for human-facing hub links (trial pages, the leaderboard). API
# access does not go through this -- harbor resolves its own Supabase URL.
HUB_URL = "https://hub.harborframework.com"


@dataclass(frozen=True)
class Board:
    """One leaderboard and the dataset version it scores.

    Filtering is anchored to `dataset`: a job that did not run it is rejected,
    so trials from other datasets are never considered. `ref` is the sha256 of
    the canonical dataset version; checking it proves every trial ran the
    official, unmodified tasks. The hub leaderboard lives on the same package
    (`leaderboard_package` == `dataset`), under `leaderboard_name`.
    """

    # The value a submission's `board` field carries; also the filename suffix
    # `filter` appends for every board except the public one.
    key: str
    dataset: str
    ref: str
    leaderboard_name: str

    @property
    def leaderboard_package(self) -> str:
        # The slug is lowercase, so it satisfies the hub's `package` pattern
        # (/^[a-z0-9][a-z0-9_-]*\/[a-z0-9][a-z0-9_.-]*$/) and doubles as the
        # API selector -- no `package_id` indirection needed.
        return self.dataset

    @property
    def url(self) -> str:
        return (
            f"{HUB_URL}/datasets/{self.leaderboard_package}/latest"
            f"?tab=leaderboard&leaderboard={self.leaderboard_name}"
        )


# The two boards. Definitions live in leaderboard.json / leaderboard-private.json
# (see leaderboard/SETUP.md).
#
# NOTE: submissions must run the dataset as published on the Harbor Hub. The
# HuggingFace-registry config in configs/harbor_job_orca_bench.yaml is for
# local development only -- its trials carry a different `source` and are
# filtered out by submission_trials below.
#
# Re-pin after republishing a dataset (see leaderboard/SETUP.md):
#
#     uv run python -c "import asyncio; \
#       from harbor.registry.client.package import PackageDatasetClient; \
#       print(asyncio.run(PackageDatasetClient().get_dataset_metadata( \
#         'orca-bench/orca-bench@latest')).version)"
#
# A private dataset needs an authenticated harbor session (`harbor auth login`)
# or HARBOR_API_KEY -- an anonymous caller sees "Tag 'latest' not found", which
# reads as "no such dataset" but is not.

# The released split: 755 tasks with their verifiers, scored in-container.
PUBLIC = Board(
    key="public",
    dataset="orca-bench/orca-bench",
    ref="sha256:2add497ac2f93468dba1b83977c295a784c89031754e6a35520195345ad62ada",
    leaderboard_name="orca-bench",
)

# The held-out split: 324 answer-free `-hidden` tasks (see build_harbor_tasks
# .build_hidden_task), scored out-of-band by the /judge workflow.
PRIVATE = Board(
    key="private",
    dataset="orca-bench/orca-bench-private",
    ref="sha256:ab44e871420e7c32540c36c8c2af5f8cb75bf48e371e3f19791540339dfdac81",
    leaderboard_name="orca-bench-private",
)

BOARDS: dict[str, Board] = {b.key: b for b in (PUBLIC, PRIVATE)}

# Sentinel guarding an unpinned checkout; the refs above are real, so this only
# fires if someone blanks one out.
_REF_PLACEHOLDER = "sha256:TODO-PIN-ORCA-BENCH-DATASET-REF"


def submission_board(submission: dict) -> Board:
    """The board a submission targets, from its `board` field.

    Files written before the private board existed carry no `board` and are
    public: every one of them was filtered against the public dataset, so the
    default is a statement of fact, not a guess.
    """
    key = submission.get("board", PUBLIC.key)
    try:
        return BOARDS[key]
    except KeyError:
        sys.exit(
            f"submission targets unknown board {key!r}; expected one of "
            f"{sorted(BOARDS)} (leaderboard/core/hub.py)"
        )


def board_for_dataset(dataset: str | None) -> Board | None:
    """The board whose dataset a trial's `source` names, if any."""
    return next((b for b in BOARDS.values() if b.dataset == dataset), None)


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


# Errors after which harbor still runs the verifier, so the trial's report --
# whatever the agent managed to write -- is judged like any other. This is the
# `except (AgentTimeoutError, NonZeroAgentExitCodeError)` in harbor's
# SingleStepTrial._run_agent (harbor/trial/single_step.py): those two are
# recorded and the trial proceeds to _run_verifier; any other exception
# propagates and the verifier never runs. On the public board a trial with one
# of these is scored (job 2026-05-05__05-48-18: all 462 timed-out trials carry
# a reward, mostly 0 from an empty report), and the private board must count it
# the same way or the two boards disagree on what "finished" means. Any other
# error_type (a setup failure, e.g. RuntimeError from tmux) means there is no
# report for /judge to score.
JUDGEABLE_ERRORS = frozenset({"AgentTimeoutError", "NonZeroAgentExitCodeError"})


def private_trial_finished(trial: dict) -> bool:
    """True for a private-board trial that reached its verifier without a verdict.

    A `-hidden` task has no in-container verifier: its test.sh writes an empty
    rewards map (build_harbor_tasks.HIDDEN_TEST_SH), and the score arrives
    out-of-band from /judge. The Hub stores that map as `evals: {}` with
    `reward: null` -- confirmed on the first uploaded private job
    (6b469112-157f-5d55-a7c2-25417c2a226d, 2026-09-19): 324/324 rows had that
    exact shape. It is also the shape of a trial that errored before its
    verifier ran, and `status` ("completed") and `is_scored` (true) do not
    separate the two; `error_type` does -- None (or a JUDGEABLE_ERRORS entry)
    here, the exception class name ("RuntimeError") on the errored one. The
    dataset guard keeps a public trial with the same shape (a run whose
    verifier produced nothing) from being counted as finished: on the public
    board no verdict means excluded.
    """
    if trial.get("source") != PRIVATE.dataset:
        return False
    evals = trial.get("evals")
    error = trial.get("error_type")
    return (
        isinstance(evals, dict)
        and not evals
        and trial.get("reward") is None
        and (error is None or error in JUDGEABLE_ERRORS)
    )


def submission_trials(submission: dict) -> list[dict]:
    """Bulk trial metadata for a submission: every latest-attempt trial of its
    source_jobs on the submission's board dataset matching its source_filter
    (agent, agent version, model). Shared by static_analysis (checks + metrics)
    and clone_trials, so what is checked is exactly what gets cloned."""
    sf = submission["source_filter"]
    dataset = submission_board(submission).dataset
    return [
        t
        for link in submission["source_jobs"]
        for t in hub_job_trials(job_uuid(link))
        if t.get("source") == dataset
        and t.get("agent_name") == sf["agent"]
        and t.get("agent_version") == sf["agent_version"]
        and trial_model(t) == sf["model_name"]
    ]


def dataset_task_digests(board: Board) -> dict[str, str]:
    """Canonical {task_name: sha256 ref} for the board's dataset version, from
    the registry. A metadata-only query (no task content downloaded)."""
    if board.ref == _REF_PLACEHOLDER:
        sys.exit(
            f"the {board.key} board's ref is still the placeholder in "
            "leaderboard/core/hub.py.\nPin the published dataset version before "
            'running any check -- see leaderboard/SETUP.md, "Re-pinning a board".'
        )

    async def _fetch() -> dict[str, str]:
        from harbor.registry.client.package import PackageDatasetClient

        meta = await PackageDatasetClient().get_dataset_metadata(
            f"{board.dataset}@{board.ref}"
        )
        return {f"{t.org}/{t.name}": t.ref for t in meta.task_ids}

    return asyncio.run(_fetch())
