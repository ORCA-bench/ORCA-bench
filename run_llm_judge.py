#!/usr/bin/env python3
r"""Run LLM-as-a-judge evaluation on Harbor Hub trials.

Scores every trial of one or more Hub jobs, reading both halves of each judge
input straight from the Hub (see ``harbor_utils.hub_source``): the agent's
``report.md`` from the trial archive, and ``expected.json`` +
``tests/rubrics/`` from the task package. Per-trial results are written under
OUTPUT_DIR.

Usage::

    # Score every trial of a job
    uv run python run_llm_judge.py -od out-0224 --job <job-uuid>

    # Score batch 2 of size 50, across two jobs
    uv run python run_llm_judge.py -od out-0224 \\
        --job <uuid-a> --job <uuid-b> --batch-size 50 --batch-number 2

Trials of the answer-free ``-hidden`` split are scored against their oracle
twin's ground truth, which is what makes this script the scoring path for
``orca-bench/orca-bench-private`` -- those tasks ship without a rubric and
cannot be scored in-container (see build_harbor_tasks.py).

Model and reasoning effort default to what the in-container verifier uses
(``openai-gpt-5.4``, effort ``high``), so scores are comparable with the ones
the public split's verifier produced. Override with ``-m`` / ``-e``.

Requires HARBOR_API_KEY (or a ``harbor auth login`` session) with read access to
the oracle tasks, plus OPENAI_API_KEY for the judge itself. Access is checked
before any trial is downloaded.

Trials already scored under OUTPUT_DIR/scores are skipped *before* their archive
is fetched, so a re-run costs neither API calls nor downloads. Pass --force to
re-score.
"""

import asyncio
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from harbor_utils import hub_source
from utils import get_base_parser, setup_logging

# Add the directory containing check_prediction.py to sys.path so we can
# import the judge helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_prediction import (
    DEFAULT_MODEL,
    aggregate_judge_response,
    judge,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


def _trial_score_path(
    scores_dir: Path, model: str, effort: str | None, trial_id: str
) -> Path:
    """Return the per-trial output path under ``scores_dir``."""
    return scores_dir / f"judge-{model}-{effort or 'none'}-{trial_id}.json"


def _trial_prompt_path(
    prompts_dir: Path, model: str, effort: str | None, trial_id: str
) -> Path:
    """Return the per-trial prompt markdown path under ``prompts_dir``."""
    return prompts_dir / f"judge-{model}-{effort or 'none'}-{trial_id}.md"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via tmp + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via tmp + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


async def process_trial(
    client: AsyncOpenAI,
    judge_sem: asyncio.Semaphore,
    download_sem: asyncio.Semaphore,
    row: dict[str, Any],
    truth: dict[str, Any],
    model: str,
    reasoning_effort: str | None,
    bs: int,
    batch_num: int,
    scores_dir: Path,
    prompts_dir: Path,
    work_dir: Path,
    force: bool,
) -> tuple[str, dict[str, Any] | None]:
    """Score a single trial and write its result to a per-trial JSON file.

    Also writes the judge prompt to a sibling ``.md`` under ``prompts_dir``.

    The already-scored check runs *before* the report is fetched, so a re-run
    neither downloads archives nor calls the judge. Returns ``(trial_id, None)``
    when the trial is skipped that way; the prompt markdown is backfilled from
    the existing JSON if missing.
    """
    trial_id = row["id"]
    task_name = row.get("task_name", "")
    out_path = _trial_score_path(scores_dir, model, reasoning_effort, trial_id)
    prompt_path = _trial_prompt_path(prompts_dir, model, reasoning_effort, trial_id)
    if out_path.is_file() and not force:
        existing = json.loads(out_path.read_text())
        if existing.get("mode") not in ("llm_judge_error", "hub_fetch_error"):
            if not prompt_path.is_file():
                prompt_text = existing.get("judge_prompt")
                if prompt_text:
                    _atomic_write_text(prompt_path, prompt_text)
            return trial_id, None
        logger.info(f"Trial {trial_id}: previous run errored, re-scoring")

    task_meta = truth.get("task_meta", {})
    flag = task_meta.get("flag", "")
    rubric_data = truth.get("rubric") or []
    expected = truth.get("expected", {})

    def _error_payload(mode: str, exc: Exception) -> dict[str, Any]:
        return {
            "trial_id": trial_id,
            "task_name": task_name,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_response": getattr(exc, "raw_response", None),
            "reward": 0.0,
            "mode": mode,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "batch_size": bs,
            "batch_number": batch_num,
            "rca_depth": 0,
            "rubric_used": bool(rubric_data),
            "flag": flag,
        }

    # Bounded separately from the judge: a download is disk- and
    # bandwidth-bound and pulls a whole trial archive for one markdown file.
    async with download_sem:
        try:
            predictions = await hub_source.fetch_report(trial_id, work_dir)
        except Exception as exc:
            logger.error(f"Trial {trial_id} ({task_name}) download failed: {exc}")
            payload = _error_payload("hub_fetch_error", exc)
            _atomic_write_json(out_path, payload)
            return trial_id, payload

    async with judge_sem:
        try:
            result = await judge(
                client,
                expected,
                predictions,
                rubric_data,
                model=model,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            logger.error(f"Trial {trial_id} ({task_name}) failed: {exc}")
            payload = _error_payload("llm_judge_error", exc)
            _atomic_write_json(out_path, payload)
            return trial_id, payload

    # Materialize the post-hoc rca_depth (mean across rubrics) so the runtime
    # log and the saved JSON both surface it. Downstream consumers (load_trials
    # in utils.py) re-derive it from ``nested`` anyway, but having it at the top
    # level matches the historical schema and makes per-trial files greppable.
    agg = aggregate_judge_response(result["nested"], mode=result.get("mode"))
    rca_depth = agg["rca_depth"] if agg["rca_depth"] is not None else 0
    payload = {
        "trial_id": trial_id,
        "task_name": task_name,
        "reasoning_effort": reasoning_effort,
        "batch_size": bs,
        "batch_number": batch_num,
        "flag": flag,
        **result,
        "rca_depth": rca_depth,
    }
    _atomic_write_json(out_path, payload)
    prompt_text = result.get("judge_prompt")
    if prompt_text:
        _atomic_write_text(prompt_path, prompt_text)
    return trial_id, payload


async def process_batch(
    client: AsyncOpenAI,
    judge_sem: asyncio.Semaphore,
    download_sem: asyncio.Semaphore,
    batch: list[dict[str, Any]],
    truth: dict[str, dict[str, Any]],
    batch_num: int,
    n_batches: int,
    scores_dir: Path,
    prompts_dir: Path,
    work_dir: Path,
    force: bool,
    model: str,
    reasoning_effort: str | None,
    bs: int,
) -> None:
    """Process all trials in a batch concurrently; write per-trial JSONs."""
    logger.info(f"=== Batch {batch_num}/{n_batches}: {len(batch)} trials ===")

    tasks = [
        process_trial(
            client,
            judge_sem,
            download_sem,
            row,
            truth[row["task_name"]],
            model,
            reasoning_effort,
            bs,
            batch_num,
            scores_dir,
            prompts_dir,
            work_dir,
            force,
        )
        for row in batch
    ]
    results = await asyncio.gather(*tasks)

    n_skipped = sum(1 for _, r in results if r is None)
    n_scored = len(results) - n_skipped
    for i, (trial_id, result) in enumerate(results):
        if result is None:
            continue
        logger.info(
            f"  [{i + 1}/{len(batch)}] {result.get('task_name', '')}: "
            f"rca_depth={result.get('rca_depth', 'N/A')}/3 "
            f"(mode={result.get('mode', '?')})"
        )
    logger.info(f"Batch {batch_num}/{n_batches}: scored={n_scored} skipped={n_skipped}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the LLM judge on every trial of the given Hub jobs."""
    parser = get_base_parser()
    parser.description = "Run LLM-as-a-judge evaluation on Harbor Hub trials."
    # Match the in-container verifier, which is what produced every score
    # these are compared against: tests/test.sh runs check_prediction.py with
    # no arguments, so it judges at its own defaults (openai-gpt-5.4, effort
    # high). The shared base parser defaults --effort to None, which would
    # silently judge this split at a different reasoning effort.
    parser.set_defaults(model=DEFAULT_MODEL, effort="high")
    parser.add_argument(
        "--job",
        "-j",
        action="append",
        required=True,
        metavar="UUID",
        help="Harbor Hub job to score (link or bare UUID). Repeatable.",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=None,
        help="Number of trials per batch. If not set, process all trials at once.",
    )
    parser.add_argument(
        "--batch-number",
        "-bn",
        type=int,
        default=None,
        help="1-indexed batch number to run. If not set, run all batches sequentially.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score even if output JSON already exists.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent LLM judge API calls (default: 10).",
    )
    parser.add_argument(
        "--download-concurrency",
        type=int,
        default=4,
        help=(
            "Maximum number of concurrent trial-archive downloads (default: 4). "
            "Each holds one whole archive on disk while its report is read."
        ),
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bare UUIDs or hub links; the trailing path segment is the id either way.
    job_ids = [link.rstrip("/").split("/")[-1] for link in args.job]

    await hub_source.check_hub_auth()

    rows = await hub_source.list_job_trials(job_ids)
    if not rows:
        logger.error(f"No trials found for job(s) {', '.join(job_ids)}")
        sys.exit(1)
    rows.sort(key=lambda r: r["id"])

    # Ground truth is fetched for every task up front: it is deduplicated and
    # cached by harbor, and a missing rubric should fail before any judging
    # spend rather than silently scoring a trial against nothing.
    task_names = [r["task_name"] for r in rows]
    await hub_source.check_task_access(hub_source.oracle_package(task_names[0]))
    truth = await hub_source.load_ground_truth(task_names)
    missing = sorted({n for n in task_names if not truth.get(n, {}).get("expected")})
    if missing:
        logger.error(
            f"{len(missing)} task(s) resolved no ground truth, e.g. {missing[:3]}"
        )
        sys.exit(1)

    # Batching
    if args.batch_size is not None:
        n_batches = math.ceil(len(rows) / args.batch_size)
        batches = [
            rows[i : i + args.batch_size]
            for i in range(0, len(rows), args.batch_size)
        ]
    else:
        n_batches = 1
        batches = [rows]

    if args.batch_number is not None:
        if args.batch_number < 1 or args.batch_number > n_batches:
            logger.error(
                f"--batch-number {args.batch_number} out of range [1, {n_batches}]"
            )
            sys.exit(1)
        batches = [batches[args.batch_number - 1]]
        logger.info(
            f"Running batch {args.batch_number}/{n_batches} ({len(batches[0])} trials)"
        )

    # Init async OpenAI client
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # Process batches sequentially (trials within each batch run concurrently)
    bs = args.batch_size if args.batch_size is not None else len(rows)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(exist_ok=True)
    prompts_dir = output_dir / "judge_prompts"
    prompts_dir.mkdir(exist_ok=True)
    judge_sem = asyncio.Semaphore(args.concurrency)
    download_sem = asyncio.Semaphore(args.download_concurrency)

    with tempfile.TemporaryDirectory(prefix="hub-trials-", dir=output_dir) as tmp:
        work_dir = Path(tmp)
        for batch_idx, batch in enumerate(batches):
            batch_num = (
                args.batch_number if args.batch_number is not None else batch_idx + 1
            )
            await process_batch(
                client,
                judge_sem,
                download_sem,
                batch,
                truth,
                batch_num,
                n_batches,
                scores_dir,
                prompts_dir,
                work_dir,
                args.force,
                args.model,
                args.effort,
                bs,
            )
    logger.info("Done!")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
