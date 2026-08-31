r"""Generate plausible-root-cause answers for v2 Harbor tasks.

Step 2 of the three-script pipeline (``generate_task_specs.py`` ->
``generate_answers.py`` -> ``build_harbor_tasks.py``).

For each task spec under ``output_dir/task_specs/``, this script:

  * Skips control tasks → ``{"events": []}``
  * With ``--no-llm`` (or for any granularity listed in
    ``TRUSTED_GRANULARITIES``, currently empty), trusts the authored answer
    → ``{"events": [{"root_cause": authored_flag, "event_time": incident_dt}]}``
  * Otherwise (every non-control granularity by default):
      - Enumerates candidate events from ``data_dir/events/`` whose
        ``wall_clock_utc`` is in
        ``[start_of_local_day(reported_dt), min(current_dt, end_of_local_day(reported_dt)))``
        (window spans the reported local day the agent sees in its prompt; for
        broad sections without ``reported_dt`` we fall back to
        ``reported_period[0]``) and whose ``value != "off"``. The reported day
        bounds candidates: misspecification of the reported time is tolerated,
        but by at most one day, so events from a later day never leak in.
      - Loads each candidate's rubric from ``output_dir/json/<event_id>.json``
        (rendered via ``check_prediction.format_rubric(..., include_frontend=True)``).
      - Sends a single Structured-Outputs LLM call per task, asking which
        candidates' user-facing effects could plausibly explain the
        ``user_facing_issue``.
      - Projects plausible events to ``{root_cause: flag, event_time: ts}``.
      - Drops events whose flag is in ``EXCLUDED_FEATURE_FLAGS`` (e.g.
        ``imageSlowLoad``, too generic to stand as a ground-truth answer) when
        the task has at least one other answer; keeps such a flag when it is the
        sole answer.

Examples::

    uv run python generate_answers.py -od OUTPUT_DIR -dd DATA_DIR
    uv run python generate_answers.py -od OUTPUT_DIR -dd DATA_DIR --no-llm
    uv run python generate_answers.py -od OUTPUT_DIR -dd DATA_DIR --limit 5

"""

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from check_prediction import async_call_llm_judge, format_rubric
from task_spec import (
    TaskSpec,
    candidate_events_in_window,
    end_of_local_day,
    load_event_timestamps,
    start_of_local_day,
)
from utils import get_base_parser, setup_logging

logger = logging.getLogger(__name__)

TRUSTED_GRANULARITIES: tuple[str, ...] = ()

# Feature flags that are too generic to stand as a ground-truth answer when a
# more specific incident is also present. Dropped from a task's answer set only
# when at least one non-excluded event remains; kept when sole.
EXCLUDED_FEATURE_FLAGS: frozenset[str] = frozenset({"imageSlowLoad"})


# ───────────────────────────── Prompt ─────────────────────────────
ANSWER_PROMPT_TEMPLATE = """\
You are an expert SRE deciding which of several recent feature-flag-driven \
incidents could plausibly explain a user-reported issue. The reported issue \
is short and may be flag-agnostic; multiple incidents may match.

## Reported issue

{issue}

## Candidate incidents

For each candidate below, decide whether its user-visible effect could \
plausibly explain the reported issue. Base your decision on the rubric's \
description, mechanism, and (where present) the user-facing frontend \
symptoms. Be inclusive: if the symptoms could reasonably be confused with \
the reported issue, mark the candidate plausible.

You must return one verdict per candidate, in the same order as listed. \
Each verdict has:
  - ``event_id``: copy verbatim from the candidate.
  - ``plausible``: true if the candidate could explain the reported issue.
  - ``reason``: one short sentence justifying your verdict.

The order of verdicts must match the order of candidates. Do not omit any.

{candidate_blocks}
"""


def build_answer_prompt(issue: str, candidates: list[dict]) -> str:
    """Render the per-task LLM prompt.

    ``candidates`` is a list of ``{"event_id": str, "rubric_md": str}``.
    """
    blocks: list[str] = []
    for i, c in enumerate(candidates):
        blocks.append(f"### Candidate {i + 1}: `{c['event_id']}`\n\n{c['rubric_md']}")
    return ANSWER_PROMPT_TEMPLATE.format(
        issue=issue,
        candidate_blocks="\n\n".join(blocks),
    )


def build_output_schema(n_candidates: int) -> dict:
    """Strict JSON schema requiring exactly ``n_candidates`` verdicts."""
    return {
        "type": "object",
        "properties": {
            "plausible_events": {
                "type": "array",
                "minItems": n_candidates,
                "maxItems": n_candidates,
                "items": {
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string"},
                        "plausible": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["event_id", "plausible", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["plausible_events"],
        "additionalProperties": False,
    }


# ───────────────────────────── Fingerprint / cache ─────────────────────────────
def compute_fingerprint(
    *,
    authored_event_id: str | None,
    candidate_event_ids: list[str],
    user_facing_issue: str,
    model: str,
    effort: str | None,
) -> str:
    """Stable hash of the inputs that determine the LLM verdict for a task."""
    payload = json.dumps(
        {
            "authored_event_id": authored_event_id,
            "candidate_event_ids": sorted(candidate_event_ids),
            "user_facing_issue": user_facing_issue,
            "model": model,
            "effort": effort,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


# ───────────────────────────── Per-task work ─────────────────────────────
@dataclass
class _Ctx:
    answers_dir: Path
    prompts_dir: Path
    json_dir: Path
    event_timestamps: dict[str, dict]
    model: str
    effort: str | None
    no_llm: bool
    dry_run: bool
    force: bool
    semaphore: asyncio.Semaphore
    client: Any  # AsyncOpenAI; None when --no-llm


def _prompt_path(
    prompts_dir: Path, model: str, effort: str | None, task_id: str
) -> Path:
    """Return the per-task prompt markdown path under ``prompts_dir``."""
    return prompts_dir / f"answer-{model}-{effort or 'none'}-{task_id}.md"


def _trivial_answer(spec: TaskSpec) -> dict:
    """Build the answer for a control / easy / medium / --no-llm task."""
    if spec.section == "control" or spec.no_incident or spec.flag is None:
        return {"events": []}
    return {
        "events": [
            {
                "event_id": spec.event_id,
                "root_cause": spec.flag,
                "event_time": spec.incident_dt.isoformat(),
            }
        ]
    }


def _load_rubric(json_dir: Path, event_id: str) -> dict | None:
    path = json_dir / f"{event_id}.json"
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def _project_to_events(
    plausible_event_ids: list[str],
    by_id: dict[str, dict],
    authored_event_id: str | None,
) -> list[dict]:
    """Project plausible events to ``{event_id, root_cause, event_time}`` triples.

    All plausible events are preserved verbatim — even when two share the
    same flag (e.g. ``paymentFailure-10%`` and ``paymentFailure-50%``), they
    are materially different incidents (different times, different
    severities) and the agent might correctly identify either. The LLM
    judge sees per-event rubrics and credits matching any of them; the
    string-match fallback any-of-matches across all entries.

    The authored event (if plausible) is placed first so downstream scripts
    can pick it as the canonical rubric for the oracle solver. Other events
    follow in chronological order.
    """
    triples: list[dict] = []
    for eid in plausible_event_ids:
        ev = by_id.get(eid)
        if ev is None:
            continue
        flag = ev.get("flag")
        if not flag:
            continue
        triples.append(
            {"event_id": eid, "root_cause": flag, "event_time": ev["wall_clock_utc"]}
        )
    others = [t for t in triples if t["event_id"] != authored_event_id]
    others.sort(key=lambda e: e["event_time"])
    authored = next((t for t in triples if t["event_id"] == authored_event_id), None)
    return [authored, *others] if authored is not None else others


def _drop_excluded_flag_events(events: list[dict]) -> list[dict]:
    """Drop events whose ``root_cause`` is in ``EXCLUDED_FEATURE_FLAGS`` when at
    least one non-excluded event remains; keep them if an excluded flag is the
    sole answer. Order is preserved (authored-first from ``_project_to_events``).
    """
    non_excluded = [e for e in events if e["root_cause"] not in EXCLUDED_FEATURE_FLAGS]
    if non_excluded and len(non_excluded) < len(events):
        return non_excluded
    return events


async def process_spec(spec_path: Path, ctx: _Ctx) -> None:
    """Compute and write the answer JSON for one task spec."""
    spec = TaskSpec.read_json(spec_path)
    logger.info(f"Reading spec from {spec_path}")
    answer_path = ctx.answers_dir / f"{spec.task_id}.json"

    # Trivial paths: control / no_incident / --no-llm. Granularities listed
    # in ``TRUSTED_GRANULARITIES`` (currently empty) also short-circuit; set
    # that tuple to e.g. ("easy", "medium") to skip the LLM for those.
    if (
        spec.section == "control"
        or spec.no_incident
        or spec.flag is None
        or spec.granularity in TRUSTED_GRANULARITIES
        or ctx.no_llm
    ):
        answer = {**_trivial_answer(spec), "task_spec": spec.to_dict()}
        if ctx.dry_run:
            logger.info(f"[dry-run] {spec.task_id}: {answer}")
            return
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        answer_path.write_text(json.dumps(answer, indent=2))
        return

    # LLM path: hard / medium / universal.
    # Window spans the local day of the *reported* time (what the agent sees in
    # its prompt), not of ``current_dt``. For broad sections without a
    # point-in-time reported_dt, fall back to the start of the reported period.
    # The upper bound closes at the end of that reported local day (capped at
    # ``current_dt``): we tolerate misspecification of the reported time, but by
    # at most one day, so candidates may not spill into a later day even when
    # the agent is asked well after the reported time.
    reported_anchor = (
        spec.reported_dt
        if spec.reported_dt is not None
        else (spec.reported_period[0] if spec.reported_period else spec.current_dt)
    )
    window_lo = start_of_local_day(reported_anchor)
    window_hi = min(spec.current_dt, end_of_local_day(reported_anchor))
    candidates = candidate_events_in_window(ctx.event_timestamps, window_lo, window_hi)

    # NOTE: force-including the authored event is disabled for now. For
    # ``hard``/``medium``/``universal`` questions the authored event may
    # not actually be the most plausible match for the reported issue, and
    # we want the LLM's judgement to stand. Re-enable later if we want the
    # answer to be a superset of the authored event.
    # authored = ctx.event_timestamps.get(spec.event_id) if spec.event_id else None
    # if authored is not None and authored not in candidates:
    #     candidates = [authored, *candidates]

    # Drop candidates without a matching rubric, or with no frontend
    # symptoms documented. The reported issue is always user-facing, so an
    # event whose rubric has no frontend section can't be the cause.
    rubrics: list[dict] = []
    valid_candidates: list[dict] = []
    for ev in candidates:
        rubric = _load_rubric(ctx.json_dir, ev["id"])
        if rubric is None:
            logger.warning(
                f"{spec.task_id}: dropping candidate {ev['id']!r} — no rubric "
                f"at {ctx.json_dir / (ev['id'] + '.json')}"
            )
            continue
        frontend = (rubric.get("symptoms") or {}).get("frontend") or []
        if not frontend:
            logger.info(
                f"{spec.task_id}: dropping candidate {ev['id']!r} — no frontend "
                f"symptoms (can't explain a user-reported issue)"
            )
            continue
        rubrics.append(rubric)
        valid_candidates.append(ev)

    if not valid_candidates:
        logger.warning(
            f"{spec.task_id}: no candidate events with rubrics in window "
            f"[{window_lo}, {window_hi}); writing empty answer"
        )
        answer = {"events": [], "task_spec": spec.to_dict()}
        if not ctx.dry_run:
            answer_path.parent.mkdir(parents=True, exist_ok=True)
            answer_path.write_text(json.dumps(answer, indent=2))
        return

    by_id = {ev["id"]: ev for ev in valid_candidates}
    candidate_event_ids = [ev["id"] for ev in valid_candidates]
    fingerprint = compute_fingerprint(
        authored_event_id=spec.event_id,
        candidate_event_ids=candidate_event_ids,
        user_facing_issue=spec.user_facing_issue or "",
        model=ctx.model,
        effort=ctx.effort,
    )

    # Cache: skip when an existing answer has the same fingerprint.
    # ``--force`` bypasses the cache so the LLM is re-called and the answer
    # JSON is overwritten (combine with ``--filter``/``--limit`` to scope a
    # partial recompute without clobbering other answers).
    if answer_path.is_file() and not ctx.force:
        try:
            existing = json.loads(answer_path.read_text())
            if existing.get("fingerprint") == fingerprint:
                logger.info(f"{spec.task_id}: cached at {answer_path}, skipping")
                return
        except Exception:
            pass

    # Build prompt + schema.
    rubric_blocks = [
        {"event_id": ev["id"], "rubric_md": format_rubric(r, include_frontend=True)}
        for ev, r in zip(valid_candidates, rubrics)
    ]
    prompt = build_answer_prompt(spec.user_facing_issue or "", rubric_blocks)
    schema = build_output_schema(len(valid_candidates))

    if ctx.dry_run:
        logger.info(
            f"[dry-run] {spec.task_id}: would call LLM for {len(valid_candidates)} "
            f"candidate(s): {candidate_event_ids}"
        )
        return

    # Persist the prompt as a sibling markdown file for browsing/debugging,
    # mirroring the ``judge_prompts/`` directory written by run_llm_judge.py.
    # The full prompt is also stored in the answer JSON, but a flat .md is
    # easier to read with grep / less / an editor.
    prompt_path = _prompt_path(ctx.prompts_dir, ctx.model, ctx.effort, spec.task_id)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt)
    logger.info(f"{spec.task_id}: prompt at {prompt_path}")

    # Single LLM call per task (semaphore-bounded).
    async with ctx.semaphore:
        try:
            raw, reasoning_summary_raw, _usage = await async_call_llm_judge(
                ctx.client,
                prompt,
                model=ctx.model,
                reasoning_effort=ctx.effort,
                output_schema=schema,
            )
        except Exception as exc:
            # Don't write an answer file on LLM error: the absence of the file
            # is the cache miss that lets a plain re-run (without --force)
            # retry just the failed tasks while skipping every previously
            # successful one. The prompt is still on disk under
            # answer_prompts/ for offline inspection.
            logger.error(
                f"{spec.task_id}: LLM call failed ({type(exc).__name__}: {exc}); "
                f"no answer file written — re-run to retry"
            )
            return

    # Flatten reasoning summaries to ``list[str]`` so the on-disk shape
    # matches ``generate_questions.py``. async_call_llm_judge returns
    # ``list[list[{type, text}]]``; collapse into a flat list of texts.
    reasoning_summary: list[str] = [
        item["text"]
        for sublist in (reasoning_summary_raw or [])
        for item in sublist
        if isinstance(item, dict) and item.get("text")
    ]

    parsed = json.loads(raw)
    verdicts = parsed.get("plausible_events") or []
    verdict_by_id = {v["event_id"]: v for v in verdicts}

    # NOTE: force-including the authored event when the LLM voted false is
    # disabled for now. See the matching note above the candidate-set
    # construction. Re-enable later if we want the answer to be a superset
    # of the authored event.
    # if spec.event_id and spec.event_id in by_id:
    #     v = verdict_by_id.get(spec.event_id)
    #     if v is None:
    #         logger.warning(
    #             f"{spec.task_id}: LLM dropped authored event {spec.event_id!r} "
    #             f"from output; force-including"
    #         )
    #         verdict_by_id[spec.event_id] = {
    #             "event_id": spec.event_id,
    #             "plausible": True,
    #             "reason": "Authored event force-included (LLM omitted it).",
    #         }
    #     elif not v.get("plausible"):
    #         logger.warning(
    #             f"{spec.task_id}: LLM voted authored event {spec.event_id!r} as "
    #             f"not plausible; force-including"
    #         )
    #         v["plausible"] = True
    #         v["reason"] = "[force-included by generate_answers.py] " + v.get(
    #             "reason", ""
    #         )

    plausible_event_ids = [
        eid for eid, v in verdict_by_id.items() if v.get("plausible")
    ]
    events = _project_to_events(plausible_event_ids, by_id, spec.event_id)
    before = len(events)
    events = _drop_excluded_flag_events(events)
    if len(events) < before:
        logger.info(
            f"{spec.task_id}: dropped {before - len(events)} excluded-flag "
            f"event(s) {sorted(EXCLUDED_FEATURE_FLAGS)} (other answers present)"
        )

    candidate_audit = [
        {
            "event_id": ev["id"],
            "flag": ev.get("flag"),
            "wall_clock_utc": ev["wall_clock_utc"],
            "value": ev.get("value"),
            "plausible": bool(verdict_by_id.get(ev["id"], {}).get("plausible")),
            "reason": verdict_by_id.get(ev["id"], {}).get("reason", ""),
        }
        for ev in valid_candidates
    ]

    answer = {
        "events": events,
        "candidate_events": candidate_audit,
        "fingerprint": fingerprint,
        "model": ctx.model,
        "effort": ctx.effort,
        "prompt": prompt,
        "response": raw,
        "reasoning_summary": reasoning_summary,
        "task_spec": spec.to_dict(),
    }
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(json.dumps(answer, indent=2))
    logger.info(
        f"{spec.task_id}: {len(events)} plausible event(s) "
        f"from {len(valid_candidates)} candidate(s) "
        f"save to {answer_path}"
    )


# ───────────────────────────── Main ─────────────────────────────
async def _amain(args: Any) -> None:
    output_dir: Path = args.output_dir
    specs_dir = output_dir / "task_specs"
    answers_dir = output_dir / "answers"
    prompts_dir = output_dir / "answer_prompts"
    json_dir = output_dir / "json"

    if not specs_dir.is_dir():
        raise FileNotFoundError(f"specs directory missing: {specs_dir}")
    if not json_dir.is_dir():
        raise FileNotFoundError(
            f"rubric directory missing: {json_dir} "
            f"(generate_answers.py needs per-event rubrics there)"
        )

    answers_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    spec_paths = sorted(specs_dir.glob("*.json"))
    if args.filter:
        pattern = re.compile(args.filter)
        spec_paths = [p for p in spec_paths if pattern.search(p.stem)]
    if args.limit is not None:
        spec_paths = spec_paths[: args.limit]
    logger.info(f"Found {len(spec_paths)} spec(s) under {specs_dir}")

    event_timestamps = load_event_timestamps(args.data_dir)
    if not event_timestamps:
        raise RuntimeError(
            f"no event timestamps found in {args.data_dir / 'events'}; "
            f"generate_answers.py needs them to enumerate candidates"
        )

    client = None
    if not args.no_llm:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    ctx = _Ctx(
        answers_dir=answers_dir,
        prompts_dir=prompts_dir,
        json_dir=json_dir,
        event_timestamps=event_timestamps,
        model=args.model,
        effort=args.effort,
        no_llm=args.no_llm,
        dry_run=args.dry_run,
        force=args.force,
        semaphore=asyncio.Semaphore(args.concurrency),
        client=client,
    )

    await asyncio.gather(*(process_spec(p, ctx) for p in spec_paths))
    logger.info(f"Wrote answers to {answers_dir}")


def main() -> None:
    """CLI entry point — see module docstring for usage."""
    parser = get_base_parser()
    parser.description = (
        "Generate plausible-root-cause answers for v2 Harbor task specs."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent LLM calls (default: 8).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM filter; emit authored-only answers.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Regex to subset task_ids (matched against the spec filename stem).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N specs (after filter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print decisions without writing answer files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bypass the per-task fingerprint cache and re-call the LLM, "
            "overwriting existing answer JSONs. Combine with --filter / --limit "
            "to scope a partial recompute without clobbering other answers."
        ),
    )
    args = parser.parse_args()
    setup_logging(args.log_level)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    load_dotenv()
    main()
