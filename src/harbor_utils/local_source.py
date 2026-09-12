r"""Read judge inputs from a local Harbor jobs directory.

The local counterpart of :mod:`harbor_utils.hub_source`, for scoring trials that
live on disk rather than on the Hub -- a job still running, a job never uploaded,
or an archived tree like the backfilled private-split runs.

Only the *report* half comes from disk. Ground truth still resolves through
``hub_source.load_ground_truth``, because a trial dir carries no rubric: the
task package does, and it is fetched by name exactly as in Hub mode. So this
module produces the same row shape ``hub_source.list_job_trials`` does, and
``run_llm_judge`` differs between the two modes only in where the rows and the
report text come from.

Layout read (the one ``harbor run`` writes)::

    <jobs_dir>/<job_name>/<trial_name>/config.json
    <jobs_dir>/<job_name>/<trial_name>/verifier/report.md

``config.json``'s ``task.name`` is the hashed package name (``orca-bench/<hash>``)
-- the same namespace the Hub rows use, and what ``hub_source.oracle_package``
keys on, so a ``-hidden`` trial resolves to its oracle twin here too.

The trial *directory name* is the id. It is what ``utils.load_trials`` and
``backfill_verifier_outputs`` already pair trials by, so score files written from
a jobs dir line up with those tools. Names must be unique across the whole tree;
a collision would have two trials share one score file, so it raises.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPORT_RELPATH = "verifier/report.md"


def list_trials(jobs_dir: Path) -> list[dict[str, Any]]:
    """Every trial under ``jobs_dir`` as ``{id, task_name, ...}``.

    Accepts a tree of job directories (``<jobs_dir>/<job>/<trial>/``). Trials
    with no ``config.json`` are skipped with a warning -- they carry no task
    identity, so there is nothing to resolve ground truth against.
    """
    jobs_dir = Path(jobs_dir)
    if not jobs_dir.is_dir():
        raise FileNotFoundError(f"jobs directory not found: {jobs_dir}")

    rows: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    n_skipped = 0
    for job_dir in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
            config_path = trial_dir / "config.json"
            if not config_path.is_file():
                n_skipped += 1
                continue
            config = json.loads(config_path.read_text())
            task = config.get("task") or {}
            agent = config.get("agent") or {}

            trial_id = trial_dir.name
            if trial_id in seen:
                raise ValueError(
                    f"duplicate trial name {trial_id!r} in {seen[trial_id]} and "
                    f"{trial_dir}; score files are keyed by it, so the two would "
                    "overwrite each other"
                )
            seen[trial_id] = trial_dir

            rows.append(
                {
                    "id": trial_id,
                    "task_name": task.get("name", ""),
                    "source": task.get("source"),
                    "agent_name": agent.get("name"),
                    "agent_version": agent.get("version"),
                    "model_name": agent.get("model_name"),
                    "model_provider": agent.get("model_provider"),
                    "error_type": ((config.get("exception_info") or {}) or {}).get(
                        "exception_type"
                    ),
                    "job_id": job_dir.name,
                    # Local-only: where read_report finds the text.
                    "_report_path": str(trial_dir / REPORT_RELPATH),
                }
            )

    if n_skipped:
        logger.warning(f"Skipped {n_skipped} trial dir(s) with no config.json")
    missing = [r["id"] for r in rows if not r["task_name"]]
    if missing:
        raise ValueError(
            f"{len(missing)} trial(s) have no task.name in config.json, e.g. "
            f"{missing[:3]}; ground truth cannot be resolved for them"
        )
    logger.info(f"Found {len(rows)} trial(s) under {jobs_dir}")
    return rows


def report_reader(rows: list[dict[str, Any]]):
    """An async reader with :func:`hub_source.fetch_report`'s signature.

    Closes over the paths ``list_trials`` already resolved, so the caller can
    swap sources without knowing which one it has. ``work_dir`` is accepted and
    ignored -- nothing is downloaded, so there is no scratch space to use.

    Returns ``""`` for a trial with no ``report.md``, matching Hub mode: the
    agent wrote nothing, or errored before the verifier ran, and the judge
    treats an empty report as its own outcome.
    """
    paths = {r["id"]: Path(r["_report_path"]) for r in rows}

    async def read(trial_id: str, work_dir: Path) -> str:
        del work_dir  # no download, nothing to stage
        path = paths[trial_id]
        return path.read_text() if path.is_file() else ""

    return read
