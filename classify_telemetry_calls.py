"""Classify each telemetry tool call as success, error, or empty using an LLM judge.

For each trial CSV under ``output_dir/tool_calls``, filters to telemetry rows
(those whose ``categories`` intersect with ``TELEM_ALL_CATEGORIES``) and asks
an LLM to classify each call individually as one of ``success``, ``error``,
or ``empty``. Results are written one JSON per call at
``output_dir/telemetry_classifications/{trial_id}/{tool_call_id}.json``.

The judge uses the **recovered** terminal output written by
``recover_outputs.py`` when available — recovered outputs are uncontaminated
by terminus-2's 40-row pane truncation, which can cause a telemetry call to
appear as ``error`` when its JSON response was actually scrolled off-screen
by a parallel sibling command. Falls back to the CSV's ``output`` column
(the truncated trajectory observation) when no recovered file exists.

Re-running the script auto-retries any per-call file that has an ``error``
field set (e.g. from a previous timeout); pass ``--force`` to also re-do
calls that completed successfully.

Examples::

    cd examples/otel-demo
    # 1. Recover full outputs from recording.cast first (optional but recommended)
    uv run python recover_outputs.py -od out-0516 -jd jobs-sub
    # 2. Then classify
    uv run python classify_telemetry_calls.py -od out-0516 --concurrency 800 -e high
    # 3. (Optional) re-classify ignoring recovered outputs, for before/after:
    uv run python classify_telemetry_calls.py -od out-0516 --from-trajectory --force

"""

import ast
import asyncio
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from check_prediction import DEFAULT_MODEL, async_call_llm_judge
from format_tool_calls import SHELL_TOOL_NAMES, _extract_shell_cmd
from utils import (
    TELEM_ALL_CATEGORIES,
    get_base_parser,
    load_invalid_trial_ids,
    setup_logging,
)

logger = logging.getLogger(__name__)

LABELS = ("success", "error", "empty")

CLASSIFIER_PROMPT = """\
You are evaluating a single telemetry tool call made by an AI agent during an
SRE investigation. Classify it into EXACTLY ONE of these labels:

- success: the query returned a non-empty, well-formed telemetry result.
- error: the query failed — HTTP 4xx/5xx, connection refused, malformed query,
  authentication failure, JSON parse error, exception trace, Prometheus
  ``"status": "error"``, ``curl: (`` connection errors, etc.
- empty: the query succeeded but the result set is empty — e.g.
  ``"result": []``, no log streams matched, no spans returned, an HTTP 200 with
  zero data points.

## Categories assigned to this call
{categories}

## Command
{command}

## Raw arguments
{arguments}

## Output (verbatim)
{output}

## Notes
- The output may be a terminus terminal transcript that echoes prompts and
  contains text from sibling parallel commands within the same step.
  Classify based on whether THIS specific command appears to have returned
  data, errored, or returned an empty result.
- A 200 response that legitimately contains no data is ``empty``, not ``error``.
- An ``"status": "error"`` field inside an otherwise-200 JSON body is ``error``.

Respond with ONLY a JSON object (no markdown fences, no extra text):
{{"label": "<success|error|empty>", "reasoning": "<1-2 sentence explanation>"}}
"""


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via tmp + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _parse_categories(value: Any) -> list[str]:
    """Parse a categories cell (Python repr list string) safely."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    s = value.strip()
    if not s or s.lower() in {"none", "nan"}:
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def _telemetry_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the telemetry-tool-call subset of ``df`` with a ``_cats`` column."""
    shell_df = df[df["function_name"].isin(SHELL_TOOL_NAMES)].copy()
    shell_df["_cats"] = shell_df["categories"].apply(_parse_categories)
    is_telem = shell_df["_cats"].apply(
        lambda cs: any(c in TELEM_ALL_CATEGORIES for c in cs)
    )
    return shell_df[is_telem]


def _build_prompt(
    row: pd.Series, output_text: str | None = None
) -> tuple[str, str, dict]:
    """Render the classifier prompt for a single telemetry row.

    If ``output_text`` is provided, it's used in place of ``row['output']`` —
    pass the asciinema-recovered text here. Returns
    ``(prompt_text, command, parsed_arguments)``.
    """
    try:
        args = json.loads(row.get("arguments", "{}") or "{}")
    except (json.JSONDecodeError, TypeError):
        args = {}
    command = _extract_shell_cmd(row["function_name"], args)
    out = output_text if output_text is not None else str(row.get("output", "") or "")
    prompt = CLASSIFIER_PROMPT.format(
        categories=", ".join(row["_cats"]) or "(none)",
        command=command or "(empty command)",
        arguments=json.dumps(args, indent=2, ensure_ascii=False),
        output=out,
    )
    return prompt, command, args


def _parse_response(text: str) -> dict:
    """Parse the LLM JSON response and validate the ``label`` field."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    parsed = json.loads(cleaned)
    label = str(parsed.get("label", "")).strip().lower()
    if label not in LABELS:
        raise ValueError(f"Unrecognized label: {label!r}")
    return {"label": label, "reasoning": parsed.get("reasoning", "")}


async def _classify_call(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    row: pd.Series,
    trial_id: str,
    output_text: str,
    source: str,
    model: str,
    reasoning_effort: str | None,
    prompts_dir: Path,
    out_trial_dir: Path,
    force: bool,
) -> tuple[str, dict[str, Any] | None]:
    """Classify one telemetry tool call.

    Returns ``(tc_id, None)`` when an existing per-call JSON has no ``error``
    field and ``--force`` was not passed; otherwise returns ``(tc_id, record)``.
    """
    tc_id = str(row["tool_call_id"])
    out_path = out_trial_dir / f"{tc_id}.json"
    if out_path.is_file() and not force:
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            existing = None
        if existing and "error" not in existing:
            return tc_id, None

    prompt, command, _args = _build_prompt(row, output_text=output_text)
    trial_prompts_dir = prompts_dir / trial_id
    trial_prompts_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(trial_prompts_dir / f"{tc_id}.md", prompt)

    record: dict[str, Any] = {
        "step_id": row["step_id"],
        "function_name": row["function_name"],
        "command": command,
        "categories": list(row["_cats"]),
        "output_source": source,
    }

    async with semaphore:
        try:
            raw, _ = await async_call_llm_judge(
                client, prompt, model=model, reasoning_effort=reasoning_effort
            )
            record.update(_parse_response(raw))
        except Exception as exc:
            logger.error(f"Trial {trial_id} call {tc_id} failed: {exc}")
            record.update(
                {
                    "label": None,
                    "reasoning": f"Classification error: {type(exc).__name__}: {exc}",
                    "error": str(exc),
                }
            )

    _atomic_write_text(out_path, json.dumps(record, indent=2, ensure_ascii=False))
    return tc_id, record


async def _classify_trial(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    trial_id: str,
    df: pd.DataFrame,
    recovered: dict[str, str] | None,
    model: str,
    reasoning_effort: str | None,
    out_dir: Path,
    prompts_dir: Path,
    force: bool,
    from_trajectory: bool,
) -> None:
    """Classify all telemetry calls in a trial; one JSON per call."""
    telem_df = _telemetry_rows(df)
    out_trial_dir = out_dir / trial_id
    out_trial_dir.mkdir(parents=True, exist_ok=True)

    if telem_df.empty:
        logger.info(f"{trial_id}: no telemetry calls")
        return

    tasks = []
    for _, row in telem_df.iterrows():
        tc_id = str(row["tool_call_id"])
        if from_trajectory or recovered is None or tc_id not in recovered:
            output_text = str(row.get("output", "") or "")
            source = "trajectory.json observation"
        else:
            output_text = recovered[tc_id]
            source = "asciinema-recovered"
        tasks.append(
            _classify_call(
                client,
                semaphore,
                row,
                trial_id,
                output_text,
                source,
                model,
                reasoning_effort,
                prompts_dir,
                out_trial_dir,
                force,
            )
        )
    results = await asyncio.gather(*tasks)

    n_skipped = sum(1 for _, rec in results if rec is None)
    counts: dict[str, int] = {lab: 0 for lab in LABELS}
    n_errors = 0
    for _, rec in results:
        if rec is None:
            continue
        if rec["label"] in counts:
            counts[rec["label"]] += 1
        else:
            n_errors += 1
    n_processed = len(results) - n_skipped
    logger.info(
        f"{trial_id}: {len(results)} calls — processed {n_processed}, "
        f"skipped {n_skipped} (already done) — "
        f"success={counts['success']}, error={counts['error']}, "
        f"empty={counts['empty']}"
        + (f" (judge errors: {n_errors})" if n_errors else "")
    )


async def main() -> None:
    """Classify telemetry tool calls for all trials in ``output_dir``."""
    parser = get_base_parser()
    parser.description = (
        "Classify each telemetry tool call as success/error/empty using an LLM judge."
    )
    parser.set_defaults(model=DEFAULT_MODEL)
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=None,
        help="Number of trials per batch.",
    )
    parser.add_argument(
        "--batch-number",
        "-bn",
        type=int,
        default=None,
        help="1-indexed batch number to run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-classify every call, including ones with successful prior "
            "results. Without --force, only calls with an error field are "
            "redone (other completed calls are skipped)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent LLM API calls (default: 10).",
    )
    parser.add_argument(
        "--from-trajectory",
        action="store_true",
        help=(
            "Force using the trajectory.json observation text (the truncated "
            "one) instead of the asciinema-recovered output. Useful for "
            "before/after comparisons."
        ),
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    output_dir = args.output_dir
    tool_calls_dir = output_dir / "tool_calls"
    if not tool_calls_dir.is_dir():
        logger.error(f"No tool_calls directory at {tool_calls_dir}")
        sys.exit(1)

    out_dir = output_dir / "telemetry_classifications"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "telemetry_classifier_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    recovered_dir = output_dir / "recovered_outputs"
    have_recovered = recovered_dir.is_dir()
    if not have_recovered and not args.from_trajectory:
        logger.warning(
            f"No recovered_outputs/ at {recovered_dir} — falling back to "
            "trajectory.json observations. Run recover_outputs.py first for "
            "the cleaner signal."
        )

    invalid_ids: set[str] = (
        load_invalid_trial_ids(args.filter_xlsx) if args.filter_xlsx else set()
    )

    csv_paths = sorted(tool_calls_dir.glob("*.csv"))
    trial_ids = [p.stem for p in csv_paths if p.stem not in invalid_ids]
    logger.info(f"Found {len(trial_ids)} trial CSVs in {tool_calls_dir}")

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

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    for batch_idx, batch in enumerate(batches):
        batch_num = (
            args.batch_number if args.batch_number is not None else batch_idx + 1
        )
        logger.info(
            f"=== Batch {batch_num}/{n_batches}: scanning {len(batch)} trials "
            f"(per-call skip handles already-done) ==="
        )
        trial_tasks = []
        for tid in batch:
            df = pd.read_csv(tool_calls_dir / f"{tid}.csv")
            recovered = None
            if have_recovered and not args.from_trajectory:
                rp = recovered_dir / f"{tid}.json"
                if rp.is_file():
                    try:
                        recovered = json.loads(rp.read_text())
                    except Exception as exc:
                        logger.warning(f"Could not parse {rp}: {exc}")
                        recovered = None
            trial_tasks.append(
                _classify_trial(
                    client,
                    semaphore,
                    tid,
                    df,
                    recovered,
                    args.model,
                    args.effort,
                    out_dir,
                    prompts_dir,
                    args.force,
                    args.from_trajectory,
                )
            )
        await asyncio.gather(*trial_tasks)

    done = list(out_dir.glob("*/*.json"))
    logger.info(f"Done! {len(done)} per-call JSONs in {out_dir}")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
