"""task_groups: per-task labels for the leaderboard's subset metrics.

Hub trial rows carry only a task's hashed name (`orca-bench/<sha256[:16]>`, see
build_harbor_tasks.task_name_hash), and neither the bulk rows nor the dataset
metadata query expose task metadata. The labels therefore come from the task
content itself: each task's `task.toml` `[metadata]`, fetched from the registry
at the refs the pinned dataset records -- the same source utils.load_task_metadata
uses for the analysis pipeline.

Reading at the *pinned* refs keeps the labels version-exact: they describe
exactly the task content each trial ran, the same guarantee the per-trial digest
check provides.

Two labels, both straight out of `[metadata]`:

  * ``is_control`` -- a task is a control (no-incident) task iff its ``events``
    list is empty. Same rule as utils.py's ``_is_control``.
  * ``difficulty``  -- the tier ladder ``easy`` / ``medium`` / ``hard``.

CAREFUL: the tasks also carry a ``granularity`` field holding a *different*
ladder (``easy`` / ``hard`` / ``universal``) for the same tiers. The two share
the labels ``easy`` and ``hard``, so reading the wrong field is silent on those
tiers and wrong only in the middle -- which is why DIFFICULTIES is asserted
below rather than trusted.
"""

from __future__ import annotations

import asyncio
import sys
import tomllib
from dataclasses import dataclass

from leaderboard.core.hub import DATASET, DATASET_REF, dataset_task_digests

# The metadata field the subsets key on, and the exact ladder it must contain.
# `granularity` holds a rival ladder for the same tiers; the assertion below is
# what catches reading it by mistake, since the two differ only by
# `medium` (this one) vs `universal` (that one).
DIFFICULTY_FIELD = "difficulty"
DIFFICULTIES = ("easy", "medium", "hard")


class TaskLabelError(RuntimeError):
    """The task metadata does not match what the subset metrics assume."""


@dataclass(frozen=True)
class TaskLabel:
    """What a task contributes to the subset metrics."""

    difficulty: str | None
    is_control: bool


_CACHE: dict[str, TaskLabel] | None = None


def _label(metadata: dict) -> TaskLabel:
    """One task's label from its `task.toml` `[metadata]` table."""
    return TaskLabel(
        difficulty=metadata.get(DIFFICULTY_FIELD),
        # A control task is one with no incident to find. Control tasks still
        # carry a difficulty, so the difficulty subsets must intersect with
        # `not is_control` rather than partition on difficulty alone.
        is_control=not bool(metadata.get("events")),
    )


def _fetch_labels() -> dict[str, TaskLabel]:
    """Download the pinned dataset's tasks and read their `[metadata]`.

    Mirrors utils.load_task_metadata: harbor caches task packages by digest
    under TASK_CACHE_DIR, so repeat runs in a warm environment hit no network
    (CI caches that directory keyed on DATASET_REF).
    """
    from harbor.models.task.id import PackageTaskId
    from harbor.tasks.client import TaskClient

    digests = dataset_task_digests()
    task_ids = {
        name: PackageTaskId(org=name.split("/")[0], name=name.split("/")[1], ref=ref)
        for name, ref in digests.items()
    }
    asyncio.run(TaskClient().download_tasks(list(task_ids.values()), export=False))

    labels: dict[str, TaskLabel] = {}
    missing: list[str] = []
    for name, task_id in task_ids.items():
        toml_path = task_id.get_local_path() / "task.toml"
        if not toml_path.is_file():
            missing.append(name)
            continue
        metadata = tomllib.loads(toml_path.read_text()).get("metadata", {})
        labels[name] = _label(metadata)

    if missing:
        raise TaskLabelError(
            f"no task.toml for {len(missing)} task(s) of {DATASET}@{DATASET_REF}, "
            f"e.g. {missing[:3]}; the subset metrics cannot be computed without "
            "every task's labels."
        )
    check_difficulties(labels)
    return labels


def check_difficulties(labels: dict[str, TaskLabel]) -> None:
    """Fail unless the difficulty ladder is exactly DIFFICULTIES.

    Guards against reading `granularity` instead of `difficulty`, and against a
    republished dataset that renames the ladder -- either would silently
    redefine an already-published column (the `hard` column would swap between
    the hardest and the middle tier).
    """
    seen = {label.difficulty for label in labels.values()}
    if seen != set(DIFFICULTIES):
        raise TaskLabelError(
            f"tasks of {DATASET}@{DATASET_REF} report difficulties "
            f"{sorted(map(str, seen))}, expected {list(DIFFICULTIES)}. If this "
            "is the `easy`/`hard`/`universal` ladder, the labels are being read "
            "from `granularity` instead of `difficulty`; if the dataset renamed "
            "the ladder, the published subset columns change meaning and the "
            "leaderboard schema must be updated deliberately."
        )


def task_labels() -> dict[str, TaskLabel]:
    """`{hub task_name: TaskLabel}` for every task in the pinned dataset.

    Memoized: static_analysis and the promote step both need the labels, and
    fetching is the one expensive step in computing the metrics.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = _fetch_labels()
    return _CACHE


def label_for(labels: dict[str, TaskLabel], task_name: str) -> TaskLabel:
    """The label for one trial's task, or exit with a clear message.

    A miss means the submission ran a task the pinned dataset does not contain,
    which the coverage and digest checks should already have rejected.
    """
    try:
        return labels[task_name]
    except KeyError:
        sys.exit(
            f"task {task_name!r} is not in {DATASET}@{DATASET_REF}; the pinned "
            "dataset and the submitted trials have diverged."
        )
