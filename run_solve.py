#!/usr/bin/env python3
"""Batch-generate solution reports from rubric JSONs via LLM.

Reads rubric JSON files from ``output_dir/json/`` and generates incident
reports into ``output_dir/solutions/`` using the LLM.

Examples::

    # Generate solutions for all rubrics:
    uv run python run_solve.py -od share-8h

    # Single feature flag with custom effort:
    uv run python run_solve.py -od share-8h -f productCatalogFailure -e high

    # With concurrency limit:
    uv run python run_solve.py -od share-8h --concurrency 3
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from check_prediction import default_model
from solve import DEFAULT_EFFORT, generate_report
from utils import get_base_parser, setup_logging

logger = logging.getLogger(__name__)


async def process_rubric(
    client: AsyncOpenAI | None,
    semaphore: asyncio.Semaphore,
    rubric_path: Path,
    solutions_dir: Path,
    responses_dir: Path,
    model: str | None,
    effort: str | None,
    force: bool = False,
) -> Path | None:
    """Generate a solution report for a single rubric JSON file.

    Returns the output path on success, None on failure.
    """
    rubric_data = json.loads(rubric_path.read_text())
    flag = rubric_data.get("feature_flag", "unknown")

    out_path = solutions_dir / f"{rubric_path.stem}.md"
    if out_path.exists() and not force:
        logger.info(f"Skipping {rubric_path.name} (already exists)")
        return out_path

    logger.info(f"Generating solution for {rubric_path.name} (flag: {flag})...")

    async with semaphore:
        try:
            # One rubric file per call here: run_solve.py batch-generates a
            # per-cause reference report, unlike the in-task oracle which
            # covers a task's whole rubric set in one report.
            report_text, details = await generate_report(
                client, [rubric_data], model=model, effort=effort
            )
        except Exception as exc:
            logger.error(f"Generation failed for {rubric_path.name}: {exc}")
            return None

    out_path.write_text(report_text)
    logger.info(f"Wrote {out_path}")

    # Only persist a response record when an LLM was actually called. The
    # record is solve.py's provenance dict verbatim, so it matches what the
    # in-task oracle writes to /logs/agent/solve-details.json.
    if model is not None:
        response_path = responses_dir / f"{rubric_path.stem}-{model}-{effort}.json"
        response_path.write_text(json.dumps(details, indent=2))
        logger.info(f"Wrote {response_path}")
    return out_path


async def async_main() -> None:
    """Process all rubric JSONs and generate solution reports concurrently."""
    parser = get_base_parser()
    parser.description = "Batch-generate solution reports from rubric JSONs via LLM."
    # Resolved here, not at import: load_dotenv() runs in __main__, after
    # this module is imported, and the default follows $OPENAI_BASE_URL.
    parser.set_defaults(model=default_model())
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent LLM calls (default: 5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing solution files instead of skipping them.",
    )
    args = parser.parse_args()
    setup_logging(getattr(logging, args.log_level))

    output_dir: Path = args.output_dir
    model = args.model
    effort = args.effort if args.effort else DEFAULT_EFFORT

    json_dir = output_dir / "json"
    if not json_dir.is_dir():
        logger.error(f"JSON rubrics directory not found: {json_dir}")
        sys.exit(1)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        logger.error(f"No JSON files found in {json_dir}")
        sys.exit(1)

    # Filter by feature flag if specified
    if args.feature_flag:
        filtered = []
        for jf in json_files:
            data = json.loads(jf.read_text())
            if data.get("feature_flag") == args.feature_flag:
                filtered.append(jf)
        json_files = filtered
        if not json_files:
            logger.error(f"No rubrics found matching feature flag: {args.feature_flag}")
            sys.exit(1)

    solutions_dir = output_dir / "solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    client: AsyncOpenAI | None = None
    if model is not None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not set")
            sys.exit(1)

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        process_rubric(
            client,
            semaphore,
            jf,
            solutions_dir,
            responses_dir,
            model,
            effort,
            force=args.force,
        )
        for jf in json_files
    ]

    results = await asyncio.gather(*tasks)
    succeeded = sum(1 for r in results if r is not None)
    failed = sum(1 for r in results if r is None)

    print(f"\nDone. Generated {succeeded} solutions, {failed} failures.")
    print(f"Output: {solutions_dir}")


def main() -> None:
    """Entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    load_dotenv()
    main()
