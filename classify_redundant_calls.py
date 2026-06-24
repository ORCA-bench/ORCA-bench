"""Cluster redundant telemetry tool calls per ``(trial × type)`` using an LLM judge.

For each trial CSV under ``output_dir/tool_calls``, filters to telemetry rows,
buckets each row into ``metrics / logs / traces`` (rows whose categories
don't intersect those buckets are skipped), and asks an LLM to cluster
commands that repeat the same query — allowing minor variations like
different time ranges, ``head -N``, or rephrasings.

Results are written one JSON per ``(trial, type)`` at
``output_dir/redundant_telemetry_calls/{trial_id}/{type}.json``, keyed by
``tool_call_id``, with ``cluster_id``, ``is_redundant``, and
``peer_call_ids``. Re-running auto-retries any per-(trial, type) JSON that
has an ``error`` field; pass ``--force`` to also re-cluster groups that
completed successfully.

Examples::

    cd examples/otel-demo
    uv run python classify_redundant_calls.py -od out-0514
    uv run python classify_redundant_calls.py -od out-0514 --concurrency 200 -e high
    uv run python classify_redundant_calls.py -od out-0514 --force

"""

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
from classify_telemetry_calls import _atomic_write_text, _telemetry_rows
from format_telemetry_calls import TELEM_TYPES, classify_type
from format_tool_calls import _extract_shell_cmd
from utils import get_base_parser, load_invalid_trial_ids, setup_logging

logger = logging.getLogger(__name__)

CLUSTERING_PROMPT = """\
You are evaluating telemetry tool calls made by an SRE agent during a single
investigation. Some of the {n} {type} commands below may be redundant — the
agent ran the same or a near-identical query more than once.

Cluster the commands. Two commands belong to the SAME cluster if they query
essentially the same data, allowing minor variations like:
- different time ranges (different ``start=`` / ``end=`` / ``[5m]`` etc.),
- different ``head -N`` / ``tail -N`` / ``limit`` / pagination,
- different output formatting (``| python3 -m json.tool``, ``| jq ...``),
- the same query expressed slightly differently.

But NOT the same cluster if they query different metrics, services, log
streams, span operations, or are fundamentally different operations.

Number clusters 0, 1, 2, ... in order of first occurrence (the cluster of
the first command is 0; the cluster of the next command that doesn't match
any earlier one is 1; etc.).

Commands (in order):
{commands}

Respond with ONLY a JSON object with two top-level keys, no markdown fences:

  - ``assignments``: object mapping every tool_call_id to its cluster id (int).
  - ``reasonings``: object mapping every cluster id (string form, e.g. "0")
    to a 1-2 sentence justification — name the shared query/data and call
    out the variations among members, or note that the cluster has only one
    member.

Example:
{{
  "assignments": {{"call_0_1": 0, "call_2_1": 0, "call_3_1": 1}},
  "reasonings": {{
    "0": "Both query app_ads_ad_requests_total, only the time range differs.",
    "1": "Distinct query for rpc_server_duration_milliseconds_count with no peers."
  }}
}}
"""


def _build_prompt(rows: list[dict], ttype: str) -> str:
    """Render the clustering prompt for one ``(trial, type)`` group."""
    cmds = "\n".join(
        f"[{i}] {r['tool_call_id']} -- {r['command']}" for i, r in enumerate(rows)
    )
    return CLUSTERING_PROMPT.format(n=len(rows), type=ttype, commands=cmds)


def _parse_response(
    text: str, expected_ids: list[str]
) -> tuple[dict[str, int], dict[int, str]]:
    """Parse the LLM JSON response.

    Returns ``(cluster_map, reasonings)`` where ``cluster_map`` is
    ``{tool_call_id: cluster_id}`` and ``reasonings`` is
    ``{cluster_id: reasoning_text}``. Raises if any ``expected_ids`` is
    missing from ``assignments``.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object")
    assignments = parsed.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("missing or non-object 'assignments' in response")

    cluster_map: dict[str, int] = {}
    for tc_id in expected_ids:
        if tc_id not in assignments:
            raise ValueError(f"missing tool_call_id in assignments: {tc_id}")
        v = assignments[tc_id]
        if isinstance(v, bool) or not isinstance(v, int):
            try:
                v = int(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-int cluster_id for {tc_id}: {v!r}") from exc
        cluster_map[tc_id] = v

    reasonings: dict[int, str] = {}
    raw_reasonings = parsed.get("reasonings")
    if isinstance(raw_reasonings, dict):
        for k, v in raw_reasonings.items():
            try:
                reasonings[int(k)] = str(v) if v is not None else ""
            except (TypeError, ValueError):
                continue
    return cluster_map, reasonings


def _build_payload(
    rows: list[dict],
    ttype: str,
    cluster_map: dict[str, int],
    reasonings: dict[int, str],
) -> dict:
    """Compose the per-call output dict from rows, cluster_map, and reasonings."""
    cluster_to_first: dict[int, str] = {}
    cluster_to_members: dict[int, list[str]] = {}
    for r in rows:  # rows preserve original order
        cid = cluster_map[r["tool_call_id"]]
        if cid not in cluster_to_first:
            cluster_to_first[cid] = r["tool_call_id"]
        cluster_to_members.setdefault(cid, []).append(r["tool_call_id"])

    payload: dict[str, Any] = {}
    for r in rows:
        tc_id = r["tool_call_id"]
        cid = cluster_map[tc_id]
        members = cluster_to_members[cid]
        payload[tc_id] = {
            "step_id": r["step_id"],
            "function_name": r["function_name"],
            "command": r["command"],
            "telem_type": ttype,
            "cluster_id": cid,
            "is_redundant": tc_id != cluster_to_first[cid],
            "peer_call_ids": [m for m in members if m != tc_id],
            "cluster_reasoning": reasonings.get(cid, ""),
        }
    return payload


async def _classify_group(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    trial_id: str,
    ttype: str,
    rows: list[dict],
    model: str,
    reasoning_effort: str | None,
    prompts_dir: Path,
    out_path: Path,
    force: bool,
) -> tuple[str, str]:
    """Classify one ``(trial, type)`` group; returns ``(ttype, status)``."""
    if out_path.is_file() and not force:
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            existing = None
        if existing and "error" not in existing:
            return ttype, "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Singleton fast-path: no LLM call needed.
    if len(rows) == 1:
        payload = _build_payload(
            rows,
            ttype,
            {rows[0]["tool_call_id"]: 0},
            {0: "Only one telemetry call of this type in the trial."},
        )
        _atomic_write_text(out_path, json.dumps(payload, indent=2, ensure_ascii=False))
        return ttype, "singleton"

    prompt = _build_prompt(rows, ttype)
    trial_prompts_dir = prompts_dir / trial_id
    trial_prompts_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(trial_prompts_dir / f"{ttype}.md", prompt)

    expected = [r["tool_call_id"] for r in rows]

    async with semaphore:
        try:
            raw, _ = await async_call_llm_judge(
                client, prompt, model=model, reasoning_effort=reasoning_effort
            )
            cluster_map, reasonings = _parse_response(raw, expected)
        except Exception as exc:
            logger.error(f"Trial {trial_id} type {ttype} failed: {exc}")
            _atomic_write_text(
                out_path,
                json.dumps(
                    {
                        "error": str(exc),
                        "_error_type": type(exc).__name__,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            return ttype, "error"

    payload = _build_payload(rows, ttype, cluster_map, reasonings)
    _atomic_write_text(out_path, json.dumps(payload, indent=2, ensure_ascii=False))
    return ttype, "ok"


async def _classify_trial(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    trial_id: str,
    df: pd.DataFrame,
    model: str,
    reasoning_effort: str | None,
    out_dir: Path,
    prompts_dir: Path,
    force: bool,
) -> None:
    """Cluster all telemetry calls in a trial, one task per ``(trial, type)``."""
    telem_df = _telemetry_rows(df)
    if telem_df.empty:
        return

    by_type: dict[str, list[dict]] = {ttype: [] for ttype in TELEM_TYPES}
    for _, row in telem_df.iterrows():
        ttype = classify_type(row["_cats"])
        if ttype is None:
            continue
        try:
            args = json.loads(row.get("arguments", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        cmd = _extract_shell_cmd(row["function_name"], args)
        by_type[ttype].append(
            {
                "tool_call_id": str(row["tool_call_id"]),
                "step_id": row["step_id"],
                "function_name": row["function_name"],
                "command": cmd,
            }
        )

    out_trial_dir = out_dir / trial_id
    out_trial_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for ttype, rows in by_type.items():
        if not rows:
            continue
        out_path = out_trial_dir / f"{ttype}.json"
        tasks.append(
            _classify_group(
                client,
                semaphore,
                trial_id,
                ttype,
                rows,
                model,
                reasoning_effort,
                prompts_dir,
                out_path,
                force,
            )
        )

    if not tasks:
        return

    results = await asyncio.gather(*tasks)
    parts = ", ".join(f"{t}={status}" for t, status in results)
    logger.info(f"{trial_id}: {parts}")


async def main() -> None:
    """Cluster redundant telemetry tool calls for all trials in ``output_dir``."""
    parser = get_base_parser()
    parser.description = (
        "Cluster redundant telemetry tool calls per (trial x type) using an LLM judge."
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
            "Re-cluster every (trial, type) group, including ones that "
            "completed successfully. Without --force, only groups whose JSON "
            "has an error field are redone."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent LLM API calls (default: 10).",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    output_dir = args.output_dir
    tool_calls_dir = output_dir / "tool_calls"
    if not tool_calls_dir.is_dir():
        logger.error(f"No tool_calls directory at {tool_calls_dir}")
        sys.exit(1)

    out_dir = output_dir / "redundant_telemetry_calls"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "redundant_telemetry_classifier_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

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
            f"(per-(trial,type) skip handles already-done) ==="
        )
        trial_tasks = []
        for tid in batch:
            df = pd.read_csv(tool_calls_dir / f"{tid}.csv")
            trial_tasks.append(
                _classify_trial(
                    client,
                    semaphore,
                    tid,
                    df,
                    args.model,
                    args.effort,
                    out_dir,
                    prompts_dir,
                    args.force,
                )
            )
        await asyncio.gather(*trial_tasks)

    done = list(out_dir.glob("*/*.json"))
    logger.info(f"Done! {len(done)} (trial, type) JSONs in {out_dir}")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
