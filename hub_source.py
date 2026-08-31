#!/usr/bin/env python3
r"""Fetch LLM-judge inputs straight from Harbor Hub.

No local jobs directory is involved -- both halves of a judge input come from
the Hub:

  * **the report** -- ``verifier/report.md`` from inside the trial archive.
    There is no per-artifact API (``TrialDetail`` models DB fields only;
    archive contents live in storage), so the whole archive comes down and is
    deleted as soon as the report is read. Peak disk is one trial.
  * **the ground truth** -- ``task.toml`` ``[metadata]``, ``tests/expected.json``
    and ``tests/rubrics/*.json`` from the task package, fetched into harbor's
    digest-keyed task cache. Deduplicated across trials, so a job of N trials
    over M tasks downloads M packages.

Task resolution follows the published split (see build_harbor_tasks.py):

  * ``orca-bench/<hash>``        -- carries its own ground truth.
  * ``orca-bench/<hash>-hidden`` -- the answer-free variant of the held-out
    split; its ground truth lives in ``orca-bench/<hash>``.

Ground truth is read at ``latest``, not at the ref the trial ran. The hidden and
oracle packages are built together but published separately, so a hidden trial's
own ref identifies no oracle revision. Consequence worth knowing: republishing
the oracle tasks changes what already-judged trials were scored against.

Reading the oracle tasks needs credentials that can see them -- HARBOR_API_KEY,
or a ``harbor auth login`` session. Call :func:`check_hub_access` first: an
unauthenticated caller gets "Tag 'latest' not found", which reads as "no such
dataset" but means "not logged in".
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tomllib
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

# Package-name suffix of the answer-free variant; see build_harbor_tasks.py.
HIDDEN_SUFFIX = "-hidden"

# Where the judge's report lives inside a downloaded trial dir.
REPORT_RELPATH = "verifier/report.md"


def oracle_package(task_name: str) -> str:
    """The package holding ``task_name``'s ground truth.

    Identity for a normal task; the oracle twin for a ``-hidden`` one.
    """
    return task_name.removesuffix(HIDDEN_SUFFIX)


async def check_hub_auth() -> str:
    """Fail fast when there is no usable Harbor session. Returns the user id."""
    from harbor.upload.db_client import UploadDB

    try:
        user_id = await UploadDB().get_user_id()
    except Exception as exc:
        raise SystemExit(
            f"Harbor authentication failed ({type(exc).__name__}: {exc}).\n"
            "Set HARBOR_API_KEY or run `harbor auth login`."
        ) from exc
    logger.info(f"Harbor Hub: authenticated as {user_id}")
    return user_id


async def check_task_access(package: str) -> None:
    """Fail fast when the session cannot read the ground-truth tasks.

    Separate from :func:`check_hub_auth` because the failure is different and
    much less legible: an authenticated caller without access to the task sees
    "Tag 'latest' not found", which reads as "no such task" rather than "no
    access" (leaderboard/SETUP.md documents the same trap for the dataset).
    """
    try:
        await _download_packages([package])
    except Exception as exc:
        raise SystemExit(
            f"Could not read the ground-truth task {package!r} "
            f"({type(exc).__name__}: {exc}).\n"
            "A 'Tag latest not found' here usually means the credentials cannot "
            "see the task, not that it is missing. Check `harbor auth status` "
            "and that this account has access to the oracle tasks."
        ) from exc
    logger.info(f"Harbor Hub: ground-truth tasks readable (checked {package})")


async def list_job_trials(job_ids: list[str], *, page_size: int = 200) -> list[dict]:
    """Every latest-attempt trial of ``job_ids`` as ``{id, task_name, ...}``.

    Metadata only -- one paginated RPC, no trial content. ``task_name`` on these
    rows is the hashed package name (``orca-bench/<sha256[:16]>``), which is what
    :func:`oracle_package` keys on.
    """
    from harbor.hub.client import HubClient

    client = HubClient()
    rows: list[dict] = []
    page = 1
    while True:
        result = await client.get_job_trials(job_ids, page=page, page_size=page_size)
        rows.extend(
            {
                "id": t.id,
                "task_name": t.task_name,
                "source": t.source,
                "agent_name": t.agent_name,
                "agent_version": t.agent_version,
                "model_provider": t.model_provider,
                "model_name": t.model_name,
                "error_type": t.error_type,
                "job_id": t.job_id,
            }
            for t in result.items
        )
        if page >= max(result.total_pages, 1):
            break
        page += 1
    logger.info(f"Found {len(rows)} trial(s) across {len(job_ids)} job(s)")
    return rows


async def _download_packages(package_names: list[str]) -> dict[str, Path]:
    """Download task packages at ``latest``; return ``{name: local dir}``.

    harbor caches by digest, so a warm cache makes this a no-op. Results come
    back in input order (BatchDownloadResult reassembles them), so zip is safe.
    """
    from harbor.models.task.id import PackageTaskId
    from harbor.tasks.client import TaskClient

    ids = [
        PackageTaskId(org=n.split("/")[0], name=n.split("/", 1)[1], ref="latest")
        for n in package_names
    ]
    batch = await TaskClient().download_tasks(ids, export=False)
    return {name: r.path for name, r in zip(package_names, batch.results)}


def _read_ground_truth(task_dir: Path, package: str) -> dict[str, Any]:
    """``task_meta`` / ``expected`` / ``rubric`` out of a downloaded task dir.

    Rubrics are named by the event ids in ``[metadata].events``, one file per
    plausible root cause (see build_harbor_tasks.build_task).
    """
    toml_path = task_dir / "task.toml"
    expected_path = task_dir / "tests" / "expected.json"
    if not toml_path.is_file():
        raise FileNotFoundError(f"{package}: no task.toml at {toml_path}")
    if not expected_path.is_file():
        raise FileNotFoundError(
            f"{package}: no tests/expected.json at {expected_path}. This package "
            "carries no ground truth -- if it is a `-hidden` variant, the oracle "
            "package should have been resolved instead."
        )

    task_meta = tomllib.loads(toml_path.read_text()).get("metadata", {})
    expected = json.loads(expected_path.read_text())

    rubrics: list[dict] = []
    for event in task_meta.get("events") or []:
        event_id = event.get("event_id")
        if not event_id:
            continue
        path = task_dir / "tests" / "rubrics" / f"{event_id}.json"
        if not path.is_file():
            logger.warning(f"{package}: no rubric at {path}")
            continue
        rubrics.append(json.loads(path.read_text()))
    return {"task_meta": task_meta, "expected": expected, "rubric": rubrics}


async def load_ground_truth(task_names: list[str]) -> dict[str, dict[str, Any]]:
    """``{task_name: {task_meta, expected, rubric}}`` for every task named.

    ``-hidden`` names resolve to their oracle twin, so a hidden and a public
    trial of the same incident share one download.
    """
    wanted = sorted({oracle_package(n) for n in task_names})
    logger.info(f"Fetching ground truth for {len(wanted)} task package(s)...")
    dirs = await _download_packages(wanted)
    truth = {pkg: _read_ground_truth(path, pkg) for pkg, path in dirs.items()}
    return {name: truth[oracle_package(name)] for name in set(task_names)}


async def fetch_report(trial_id: str, work_dir: Path) -> str:
    """The agent's ``report.md`` for one trial, via its Hub archive.

    The archive is deleted before returning -- it carries the full trajectory,
    and only the report is needed -- so concurrent callers cost one archive
    each, not one per job.

    Returns ``""`` when the trial produced no report (an agent that wrote
    nothing, or errored before the verifier ran); the judge treats an empty
    report as its own outcome.
    """
    from harbor.download.downloader import Downloader

    dest = work_dir / trial_id
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        result = await Downloader().download_trial(UUID(trial_id), dest, overwrite=True)
        report = result.output_dir / REPORT_RELPATH
        return report.read_text() if report.is_file() else ""
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def build_entry(row: dict, truth: dict[str, Any], predictions: str) -> dict[str, Any]:
    """One trial in the shape the judge consumes.

    ``predictions`` (the agent's report) + ``result`` (trial identity) + the
    three ground-truth keys ``task_meta`` / ``expected`` / ``rubric``, which is
    what :func:`run_llm_judge.process_trial` reads.
    """
    return {
        "predictions": predictions,
        "result": {
            "task_name": row.get("task_name", ""),
            "trial_id": row.get("id"),
            "job_id": row.get("job_id"),
            "agent_name": row.get("agent_name"),
            "agent_version": row.get("agent_version"),
            "model_name": row.get("model_name"),
            "model_provider": row.get("model_provider"),
            "error_type": row.get("error_type"),
            "source": row.get("source"),
        },
        **truth,
    }
