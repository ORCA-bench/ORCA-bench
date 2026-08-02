#!/usr/bin/env python3
r"""Run LLM-as-a-judge evaluation on harbor-export output.

Reads OUTPUT_DIR/all-predictions.json (produced by ``harbor-export``) and
scores every trial with GPT 5.4.  Results are written back to OUTPUT_DIR.

Usage::

    # Score all trials
    uv run python run_llm_judge.py -od out-0224-curl --rubrics-dir rubrics/

    # Score batch 2 of size 5 (trials 6-10)
    uv run python run_llm_judge.py -od out-0224-curl --rubrics-dir rubrics/ \\
        --batch-size 5 --batch-number 2

Prerequisite: run ``harbor-export`` with ``--registry-url`` to include
task_meta and expected in the JSON::

    uv run harbor-export <jobs_dir> -od <output_dir> \\
        --registry-url https://huggingface.co/datasets/.../registry.json

"""

import asyncio
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

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
    semaphore: asyncio.Semaphore,
    trial_id: str,
    entry: dict[str, Any],
    model: str,
    reasoning_effort: str | None,
    bs: int,
    batch_num: int,
    scores_dir: Path,
    prompts_dir: Path,
    force: bool,
) -> tuple[str, dict[str, Any] | None]:
    """Score a single trial and write its result to a per-trial JSON file.

    Also writes the judge prompt to a sibling ``.md`` under ``prompts_dir``.

    Returns ``(trial_id, None)`` when the trial is skipped because its
    output file already exists (and ``--force`` was not passed). In that
    case, the prompt markdown is backfilled from the existing JSON if
    missing.
    """
    out_path = _trial_score_path(scores_dir, model, reasoning_effort, trial_id)
    prompt_path = _trial_prompt_path(prompts_dir, model, reasoning_effort, trial_id)
    if out_path.is_file() and not force:
        existing = json.loads(out_path.read_text())
        if existing.get("mode") != "llm_judge_error":
            if not prompt_path.is_file():
                prompt_text = existing.get("judge_prompt")
                if prompt_text:
                    _atomic_write_text(prompt_path, prompt_text)
            return trial_id, None
        logger.info(f"Trial {trial_id}: previous run errored, re-scoring")

    task_meta = entry.get("task_meta", {})
    task_name = entry.get("result", {}).get("task_name", "")
    flag = task_meta.get("flag", "")
    rubric_data = entry.get("rubric") or []
    expected = entry.get("expected", {})

    async with semaphore:
        try:
            result = await judge(
                client,
                expected,
                entry.get("predictions", ""),
                rubric_data,
                model=model,
                reasoning_effort=reasoning_effort,
                task_meta=task_meta,
            )
        except Exception as exc:
            logger.error(f"Trial {trial_id} ({task_name}) failed: {exc}")
            payload = {
                "trial_id": trial_id,
                "task_name": task_name,
                "error": f"{type(exc).__name__}: {exc}",
                "raw_response": getattr(exc, "raw_response", None),
                "reward": 0.0,
                "mode": "llm_judge_error",
                "model": model,
                "reasoning_effort": reasoning_effort,
                "batch_size": bs,
                "batch_number": batch_num,
                "rca_depth": 0,
                "rubric_used": bool(rubric_data),
                "flag": flag,
            }
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
    semaphore: asyncio.Semaphore,
    batch: list[str],
    all_data: dict[str, Any],
    batch_num: int,
    n_batches: int,
    scores_dir: Path,
    prompts_dir: Path,
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
            semaphore,
            trial_id,
            all_data[trial_id],
            model,
            reasoning_effort,
            bs,
            batch_num,
            scores_dir,
            prompts_dir,
            force,
        )
        for trial_id in batch
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
    """Run the LLM judge on all trials in all-predictions.json."""
    parser = get_base_parser()
    parser.description = "Run LLM-as-a-judge evaluation on harbor-export output."
    parser.set_defaults(model=DEFAULT_MODEL)
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
        "--sampled-tasks-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file whose top-level keys are task names. "
            "When set, only trials whose task_name is in this set are scored."
        ),
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    output_dir = args.output_dir

    # Load predictions
    predictions_path = output_dir / "all-predictions.json"
    if not predictions_path.is_file():
        logger.error(f"No predictions file found at {predictions_path}")
        logger.error("Run `harbor-export <jobs_dir> -od <output_dir>` first.")
        sys.exit(1)

    logger.info(f"Loading predictions from {predictions_path}...")
    with predictions_path.open() as f:
        all_data = json.load(f)

    # Sort trial IDs deterministically
    trial_ids = sorted(all_data.keys())
    logger.info(f"Found {len(trial_ids)} trials total")

    if args.sampled_tasks_file is not None:
        with args.sampled_tasks_file.open() as f:
            allowed = set(json.load(f).keys())
        before = len(trial_ids)
        trial_ids = [
            tid
            for tid in trial_ids
            if all_data[tid].get("result", {}).get("task_name") in allowed
        ]
        logger.info(
            f"Filtered to {len(trial_ids)}/{before} trials "
            f"({len(allowed)} allowed task names from {args.sampled_tasks_file})"
        )

    # Verify that task_meta and expected are present
    sample_entry = next(iter(all_data.values()), {})
    if "task_meta" not in sample_entry or "expected" not in sample_entry:
        logger.error(
            "all-predictions.json is missing task_meta and/or expected fields. "
            "Re-run `harbor-export` with --registry-url to include them."
        )
        sys.exit(1)

    # Batching
    if args.batch_size is not None:
        n_batches = math.ceil(len(trial_ids) / args.batch_size)
        batches = [
            trial_ids[i : i + args.batch_size]
            for i in range(0, len(trial_ids), args.batch_size)
        ]
    else:
        n_batches = 1
        batches = [trial_ids]

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
    bs = args.batch_size if args.batch_size is not None else len(trial_ids)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(exist_ok=True)
    prompts_dir = output_dir / "judge_prompts"
    prompts_dir.mkdir(exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    for batch_idx, batch in enumerate(batches):
        batch_num = (
            args.batch_number if args.batch_number is not None else batch_idx + 1
        )
        await process_batch(
            client,
            semaphore,
            batch,
            all_data,
            batch_num,
            n_batches,
            scores_dir,
            prompts_dir,
            args.force,
            args.model,
            args.effort,
            bs,
        )
    logger.info("Done!")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
