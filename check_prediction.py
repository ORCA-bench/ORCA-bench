"""LLM-as-a-judge verifier for Harbor incident-RCA-report tasks.

Supports two modes:

1. **Harbor verifier** (default):
       python /tests/check_prediction.py

2. **Post-hoc single-trial**:
       python check_prediction.py --expected expected.json --predictions report.md [--rubric rubric.json]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tabulate import tabulate

logger = logging.getLogger(__name__)

JUDGE_PROMPT_MULTI_HEADER = """\
You are an expert SRE tasked with judging the quality of an AI-generated incident RCA report.

You are given:
1. **One or more** ground-truth rubrics. Each describes a plausible root cause, its incident time, mechanism, and symptoms. Several flags may have been active in the same window; the agent is credited if it correctly identifies **any one** of these rubrics.
2. The SRE agent's incident RCA report.

## SRE Agent's Incident RCA Report

{predictions}

## Ground-Truth Rubrics

{rubrics}

## Evaluation Questions

For **each rubric** in order, answer every question with true or false. Base your answer only on evidence in the agent's report; if the report does not cite the required evidence, answer false. The output must contain one entry per rubric, in the same order, and must preserve cluster order within each rubric exactly as listed.

For each rubric:

- `incident_time_within_10min`: Did the agent's timeline place the incident start within ±10 minutes of this rubric's `incident_time`?
- `feature_flag_match`: Did the agent identify this rubric's feature flag as a root cause? (The agent may name multiple flags; mark true if this flag is among them.)
- `mechanism_match`: Did the agent correctly explain this rubric's mechanism — how the flag propagates to the user-visible failure?
- For each metric/log/trace cluster listed for this rubric: per-cluster match flags as before.
- `score`: An integer 0-3 reflecting how well the report matches **this rubric specifically** (see the Per-Rubric Scoring Rubric at the end of this prompt).
"""

JUDGE_PROMPT_SCORING_FOOTER = """\

## Per-Rubric Scoring Rubric

Assign each rubric a `score` from 0 to 3 reflecting how well the agent's report matches **that rubric specifically**:

- 0: AI findings are misaligned with this rubric's mechanism and completely miss this root cause
- 1: AI verifies incident description — the report confirms symptoms described in the task prompt but does not investigate further toward this root cause
- 2: AI makes correct progress beyond symptoms — the report identifies some of this rubric's ground-truth metrics, logs, or traces and moves toward this root cause, but does not fully identify it
- 3: AI nails this root cause — the report correctly identifies it, this rubric's mechanism, and the supporting signals

Scoring is based on content correctness only; do not penalize or reward for formatting, prose style, or structural polish. Materially incorrect or misleading claims count against the per-rubric score: if the report would otherwise score N for a rubric but also adds a materially wrong claim (e.g., invents an unrelated additional cause), drop that rubric's score to N-1 (floor at 0). The same fabricated claim therefore lowers every rubric's score uniformly. When applying this penalty, explicitly identify in your reasoning which specific claim is materially wrong — quote or paraphrase the offending text. Do not apply the penalty without naming the offending claim; missing detail or incomplete explanation does not qualify as a materially wrong claim. Naming several of the listed plausible root causes is **not** a materially wrong claim.

Respond with ONLY a JSON object matching the provided output schema.
"""

JUDGE_PROMPT_CONTROL = """\
You are an expert SRE judging whether an AI-generated incident RCA report \
makes a false-positive claim about an incident at the queried time.

The task is a **control task**: the user queried the system at `current_time` \
during a verified quiet period in which no problem-pattern feature flag is \
active. The agent's snapshot includes telemetry from before `current_time`, \
so it may legitimately describe past, resolved incidents. Past incidents are \
**not** false positives — only claims placing an incident inside the quiet \
window are.

## Queried time

`current_time = {current_time}`

## Quiet window (no flag active inside)

- `quiet_window_start = {quiet_window_start}` (inclusive lower bound; \
"null" means open-ended, i.e. no neighboring incident before `current_time`)
- `quiet_window_end   = {quiet_window_end}` (exclusive upper bound; \
"null" means open-ended, i.e. no neighboring incident after `current_time`)

## SRE Agent's Incident RCA Report

{predictions}

## Evaluation

Answer the single boolean question `claims_incident_in_quiet_window`:

- **true** iff the report places **any** incident — its start, ongoing period, \
or any reported timestamp of active impact — at a time **strictly inside** \
`(quiet_window_start, quiet_window_end)`. Examples that count as true:
  - Claiming the incident is currently ongoing at `current_time`.
  - Reporting a timeline entry with a UTC timestamp inside the quiet window.
  - Stating that the incident started or is still active at any time inside the window.
- **false** otherwise. In particular, the following are NOT false positives:
  - Describing a past, resolved incident whose start, end, and timeline \
entries all fall at or before `quiet_window_start`.
  - Describing a future incident whose entries all fall at or after \
`quiet_window_end`.
  - Saying "no incident occurred at `current_time`" while describing past \
context.
  - **Timeline entries that merely document when the user reported the issue** \
(e.g. "02:04 UTC — Incident reported by user", "22:00 ET — issue reported"). \
These describe the moment the complaint arrived, not active system impact. \
Ignore such entries when applying the rule above; only entries that describe \
the system actively failing or degraded count.

Treat the bounds as half-open: a reported time exactly equal to \
`quiet_window_start` or exactly equal to `quiet_window_end` is **outside** the \
quiet window (no false positive). An open-ended bound (null) is equivalent to \
`-inf` / `+inf` respectively.

In `reasoning`, briefly state the verdict; if `claims_incident_in_quiet_window` \
is true, quote or paraphrase the specific offending claim from the report.

Respond with ONLY a JSON object matching the provided output schema.
"""

# Shrink the schedule-derived quiet window by this much on each defined side
# before showing it to the judge. The raw window is the maximal no-flag-active
# interval around current_time; the margin absorbs minor agent imprecision
# near the boundary (e.g. claiming the incident ended at 01:07 when it
# actually ended at 01:04:55 — that's ~2-3 min off, within the margin).
CONTROL_QUIET_WINDOW_MARGIN_MIN = 10


CONTROL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims_incident_in_quiet_window": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["claims_incident_in_quiet_window", "reasoning"],
    "additionalProperties": False,
}


DEFAULT_MODEL = "openai-gpt-5.4"


# ---------------------------------------------------------------------------
# Rubric rendering (JSON -> markdown)
# ---------------------------------------------------------------------------


def _format_log_clusters(clusters: list[dict], lines: list[str]) -> None:
    """Render log clusters as markdown subsections."""
    for cluster in clusters:
        name = cluster.get("name", "Unknown")
        count = cluster.get("count")
        suffix = f" ({count} entries)" if count is not None else ""
        lines.append(f"#### {name}{suffix}")
        lines.append("")

        description = cluster.get("description", "")
        if description:
            lines.append(description)
            lines.append("")

        cause = cluster.get("cause", "")
        if cause:
            lines.append(f"**Cause:** {cause}")
            lines.append("")

        source = cluster.get("source", "")
        if source:
            lines.append(f"**Source:** {source}")
            lines.append("")

        attrs = cluster.get("representative_attributes", {})
        if attrs:
            lines.append("**Representative attributes:**")
            lines.append("")
            rows = [[f"`{k}`", f"`{v}`"] for k, v in attrs.items()]
            lines.append(tabulate(rows, headers=["Field", "Value"], tablefmt="github"))
            lines.append("")

        docs = cluster.get("log_documents", [])
        if docs:
            lines.append("**Log documents:**")
            lines.append("")
            rows = [
                [f"`{d['_index']}`", f"`{d['_id']}`", d["@timestamp"]] for d in docs
            ]
            lines.append(
                tabulate(
                    rows, headers=["_index", "_id", "@timestamp"], tablefmt="github"
                )
            )
            lines.append("")


def _format_trace_clusters(clusters: list[dict], lines: list[str]) -> None:
    """Render trace clusters as markdown subsections."""
    for cluster in clusters:
        name = cluster.get("name", "Unknown")
        count = cluster.get("count")
        suffix = f" ({count} traces)" if count is not None else ""
        lines.append(f"#### {name}{suffix}")
        lines.append("")

        description = cluster.get("description", "")
        if description:
            lines.append(description)
            lines.append("")

        cause = cluster.get("cause", "")
        if cause:
            lines.append(f"**Cause:** {cause}")
            lines.append("")

        call_chain = cluster.get("call_chain", "")
        if call_chain:
            lines.append("**Call chain:**")
            lines.append("")
            lines.append("```")
            lines.append(call_chain)
            lines.append("```")
            lines.append("")

        smoking_gun = cluster.get("smoking_gun", [])
        if smoking_gun:
            lines.append("**Smoking-gun error messages:**")
            lines.append("")
            for msg in smoking_gun:
                lines.append(f"- {msg}")
            lines.append("")

        docs = cluster.get("trace_documents", [])
        if docs:
            lines.append("**Trace documents:**")
            lines.append("")
            headers = ["traceID", "@timestamp"]
            has_action = any("user_action" in d for d in docs)
            if has_action:
                headers.append("user_action")
            rows = []
            for d in docs:
                row = [f"`{d['traceID']}`", d["@timestamp"]]
                if has_action:
                    row.append(d.get("user_action", ""))
                rows.append(row)
            lines.append(tabulate(rows, headers=headers, tablefmt="github"))
            lines.append("")


_LAYER_DESCRIPTIONS = {
    "root_cause": "metric closest to the fault injection point",
    "propagation": "intermediate services reflecting the error through the call chain",
    "symptom": "user-facing impact visible to end users or clients",
    "meta": "feature flag counters or internal plumbing",
    "unknown": "unclassified",
}

_LAYER_ORDER = ["root_cause", "propagation", "symptom", "meta", "unknown"]


def _format_onset(offset: int | None) -> str:
    """Format an onset offset as '+Xs' or 'N/A'."""
    if offset is None:
        return "N/A"
    return f"+{offset}s"


def _format_label_values(labels: dict[str, str]) -> str:
    """Format label values as a compact comma-separated string."""
    if not labels:
        return ""
    return ", ".join(labels.values())


def _earliest_onset(variants: list[dict]) -> int | None:
    """Return the earliest onset_offset_seconds across variants, or None."""
    offsets = [
        v["onset_offset_seconds"]
        for v in variants
        if v.get("onset_offset_seconds") is not None
    ]
    return min(offsets) if offsets else None


def _format_metrics_list(metrics_list: list[dict], lines: list[str]) -> None:
    """Render metric families grouped by signal layer."""
    by_layer: dict[str, list[dict]] = {}
    for fam in metrics_list:
        by_layer.setdefault(fam.get("signal_layer", "unknown"), []).append(fam)

    for layer in _LAYER_ORDER:
        layer_families = by_layer.get(layer, [])
        if not layer_families:
            continue

        layer_onsets = [_earliest_onset(f["variants"]) for f in layer_families]
        layer_onsets = [o for o in layer_onsets if o is not None]
        layer_onset_str = (
            f" (earliest onset: {_format_onset(min(layer_onsets))})"
            if layer_onsets
            else ""
        )

        display_layer = layer.replace("_", " ").title()
        desc = _LAYER_DESCRIPTIONS.get(layer, "")
        lines.append(f"**{display_layer}** — {desc}{layer_onset_str}")
        lines.append("")

        for fam in layer_families:
            fam_onset = _earliest_onset(fam["variants"])
            label_vals = _format_label_values(fam.get("defining_labels", {}))
            label_part = f" ({label_vals})" if label_vals else ""
            lines.append(
                f"#### {fam['metric_family']} family{label_part}"
                f" — onset: {_format_onset(fam_onset)}"
            )
            lines.append("")

            first_desc = (
                fam["variants"][0].get("description", "") if fam["variants"] else ""
            )
            if first_desc:
                lines.append(first_desc)
                lines.append("")

            for v in fam["variants"]:
                v_onset = _format_onset(v.get("onset_offset_seconds"))
                lines.append(f"- **{v['type']}** (onset: {v_onset})")
                lines.append(f"  `{v['query']}`")
                lines.append("")


def _format_frontend_issues(issues: list[dict], lines: list[str]) -> None:
    """Render frontend issues as markdown subsections."""
    for issue in issues:
        route = issue.get("route", "Unknown route")
        lines.append(f"#### {route}")
        lines.append("")

        description = issue.get("description", "")
        if description:
            lines.append(description)
            lines.append("")

        rows = [
            ["Deterministic", str(issue.get("deterministic", ""))],
            ["Trigger condition", issue.get("trigger_condition", "")],
            ["Calling service", issue.get("calling_service", "")],
            [
                "Source",
                f"`{issue.get('source_file', '')}:{issue.get('source_line', '')}`",
            ],
        ]
        lines.append(tabulate(rows, headers=["Field", "Value"], tablefmt="github"))
        lines.append("")

        call_pattern = issue.get("call_pattern", "")
        if call_pattern:
            lines.append(f"**Call pattern:** {call_pattern}")
            lines.append("")

        cause = issue.get("cause", "")
        if cause:
            lines.append(f"**Cause:** {cause}")
            lines.append("")


def format_rubric(data: dict, include_frontend: bool = False) -> str:
    """Convert a rubric JSON dict into a consistently formatted markdown string.

    Args:
        data: The rubric JSON dict.
        include_frontend: If True, include the ``### Frontend`` section.
            Defaults to False since frontend symptoms may be noisy for
            downstream consumers (LLM judge, oracle solutions).
            TODO(Albert): systematically test the impact of including vs.
            excluding frontend symptoms on downstream consumers.

    """
    lines: list[str] = []

    lines.append(f"# {data['feature_flag']}")
    lines.append("")
    lines.append(f"**Feature flag:** `{data['feature_flag']}`")
    lines.append("")
    lines.append(f"**Description:** {data['description']}")
    lines.append("")
    lines.append(f"**Incident time:** {data['incident_time']}")
    lines.append("")

    lines.append("## Mechanism")
    lines.append("")
    lines.append(data["mechanism"])
    lines.append("")

    interaction = data.get("interaction", "")
    if interaction:
        lines.append(f"**Interaction with other feature flags:** {interaction}")
        lines.append("")

    lines.append("## Evidence")
    lines.append("")

    gs = data["symptoms"]

    metrics = gs.get("metrics", [])
    lines.append("### Metrics")
    lines.append("")
    if metrics:
        _format_metrics_list(metrics, lines)

    logs = gs.get("logs", [])
    lines.append("### Logs")
    lines.append("")
    if logs:
        _format_log_clusters(logs, lines)

    if include_frontend:
        frontend = gs.get("frontend", [])
        lines.append("### Frontend")
        lines.append("")
        if frontend:
            _format_frontend_issues(frontend, lines)

    traces = gs.get("traces", [])
    lines.append("### Traces")
    lines.append("")
    if traces:
        _format_trace_clusters(traces, lines)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge prompt + output-schema builders
# ---------------------------------------------------------------------------


def _render_checklist(rubric_data: dict) -> str:
    """Build per-cluster checklist blocks for metrics/logs/traces."""
    symptoms = rubric_data.get("symptoms", {}) or {}
    metric_clusters = symptoms.get("metrics", []) or []
    log_clusters = symptoms.get("logs", []) or []
    trace_clusters = symptoms.get("traces", []) or []

    blocks: list[str] = []

    if metric_clusters:
        lines = ["#### Metric clusters"]
        for i, m in enumerate(metric_clusters):
            family = m.get("metric_family", "")
            lines.append(
                f'- Cluster {i}: `metric_family = "{family}"`'
                f" — family_match: did the agent cite a query targeting"
                f" metric_family `{family}` under a Why-step whose causal chain"
                f" aligns with the rubric?"
            )
        blocks.append("\n".join(lines))

    if log_clusters:
        lines = ["#### Log clusters"]
        for i, log in enumerate(log_clusters):
            attrs = log.get("representative_attributes") or {}
            body = attrs.get("body", "")
            service_name = attrs.get("resource.service.name", "")
            lines.append(
                f"- Cluster {i}:\n"
                f"  - `resource.service.name`: `{service_name}`\n"
                f"  - `body`: `{body}`\n"
                f"  - body_match: did the agent cite log evidence whose body"
                f" matches the rubric's `body` under a Why-step whose causal"
                f" chain aligns with the rubric?\n"
                f"  - service_name_match: did the agent attribute the cited"
                f" log evidence to the OTel service `{service_name}`"
                f" (`resource.service.name`)?"
            )
        blocks.append("\n".join(lines))

    if trace_clusters:
        lines = ["#### Trace clusters"]
        for i, trace in enumerate(trace_clusters):
            call_chain = trace.get("call_chain", "")
            smoking_gun = trace.get("smoking_gun", []) or []
            sg_rendered = "\n".join(f"    - {s}" for s in smoking_gun)
            lines.append(
                f"- Cluster {i}:\n"
                f"  - call_chain: `{call_chain}`\n"
                f"  - smoking_gun:\n{sg_rendered}\n"
                f"  - call_chain_match: did the agent cite evidence matching"
                f" this service-level call chain under a Why-step whose causal"
                f" chain aligns with the rubric?\n"
                f"  - smoking_gun_match: did the agent cite at least one of"
                f" the listed smoking-gun signals?"
            )
        blocks.append("\n".join(lines))

    return (
        "\n\n".join(blocks)
        if blocks
        else "(No metric, log, or trace clusters in rubric.)"
    )


def build_judge_prompt(
    rubrics_data: list[dict],
    predictions: str,
) -> str:
    """Render the judge prompt for one or more ground-truth rubrics.

    The prompt asks the LLM to evaluate the agent's report against **each**
    rubric independently. The agent is credited if it correctly identifies
    any one rubric (the overall score reflects the best match).
    """
    if not rubrics_data:
        raise ValueError("build_judge_prompt called with no rubrics")

    rubric_blocks: list[str] = []
    checklist_blocks: list[str] = []
    for i, rubric in enumerate(rubrics_data):
        feature_flag = rubric.get("feature_flag", "")
        rubric_blocks.append(
            f"### Rubric {i + 1} of {len(rubrics_data)} — `{feature_flag}`\n\n"
            + format_rubric(rubric)
        )
        checklist_blocks.append(
            f"### Rubric {i + 1} of {len(rubrics_data)} — `{feature_flag}`\n\n"
            + _render_checklist(rubric)
        )

    rubrics_text = "\n\n---\n\n".join(rubric_blocks)
    checklist_text = "\n\n---\n\n".join(checklist_blocks)

    header = JUDGE_PROMPT_MULTI_HEADER.format(
        rubrics=rubrics_text,
        predictions=predictions,
    )
    return header + "\n" + checklist_text + "\n" + JUDGE_PROMPT_SCORING_FOOTER


_METRIC_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "metric_family": {"type": "string"},
        "family_match": {"type": "boolean"},
    },
    "required": ["metric_family", "family_match"],
    "additionalProperties": False,
}

_LOG_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "body_match": {"type": "boolean"},
        "service_name_match": {"type": "boolean"},
    },
    "required": ["body_match", "service_name_match"],
    "additionalProperties": False,
}

_TRACE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "call_chain_match": {"type": "boolean"},
        "smoking_gun_match": {"type": "boolean"},
    },
    "required": ["call_chain_match", "smoking_gun_match"],
    "additionalProperties": False,
}


_PER_RUBRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "feature_flag": {"type": "string"},
        "incident_time_within_10min": {"type": "boolean"},
        "feature_flag_match": {"type": "boolean"},
        "mechanism_match": {"type": "boolean"},
        "symptoms": {
            "type": "object",
            "properties": {
                "metrics": {"type": "array", "items": _METRIC_ITEM_SCHEMA},
                "logs": {"type": "array", "items": _LOG_ITEM_SCHEMA},
                "traces": {"type": "array", "items": _TRACE_ITEM_SCHEMA},
            },
            "required": ["metrics", "logs", "traces"],
            "additionalProperties": False,
        },
        "score": {"type": "integer"},
    },
    "required": [
        "feature_flag",
        "incident_time_within_10min",
        "feature_flag_match",
        "mechanism_match",
        "symptoms",
        "score",
    ],
    "additionalProperties": False,
}


def build_judge_output_schema(rubrics_data: list[dict]) -> dict:
    """Build a strict JSON Schema for the judge output: one verdict per rubric,
    each carrying its own ``score`` integer. The trial-level ``rca_depth`` is
    derived post-hoc as the mean of the per-rubric scores by
    :func:`aggregate_judge_response`.

    The rubrics array length is unconstrained in the schema (Anthropic's
    structured-output endpoint rejects ``minItems``/``maxItems`` on arrays);
    the prompt tells the model how many entries to return, and the
    per-rubric cluster order is enforced via prompt instructions.
    """
    del rubrics_data  # Schema is independent of rubric count; kept for API stability.
    return {
        "type": "object",
        "properties": {"rubrics": {"type": "array", "items": _PER_RUBRIC_SCHEMA}},
        "required": ["rubrics"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


async def async_call_llm_judge(
    client: Any,
    prompt: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str | None = None,
    output_schema: dict | None = None,
) -> tuple[str, list[dict] | None]:
    """Call the LLM judge via the OpenAI Responses API (falling back to Chat).

    When ``output_schema`` is provided, uses Structured Outputs (strict JSON
    Schema) so the returned text is guaranteed to conform to the schema.

    Returns:
        A tuple of (output_text, reasoning_summaries). reasoning_summaries is
        None when reasoning_effort is not set.

    """
    text_format: dict[str, Any] = (
        {
            "type": "json_schema",
            "name": "judge_response",
            "schema": output_schema,
            "strict": True,
        }
        if output_schema is not None
        else {"type": "text"}
    )
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "text": {"format": text_format},
            "tools": [],
            "store": True,
            "max_output_tokens": 16384,
        }
        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
        response = await client.responses.create(**kwargs)
        reasoning_summaries = None
        if reasoning_effort is not None:
            reasoning_summaries = [
                [
                    s.model_dump() if hasattr(s, "model_dump") else s
                    for s in item.summary
                ]
                for item in response.output
                if getattr(item, "type", None) == "reasoning"
                and getattr(item, "summary", None) is not None
            ]
        return response.output_text, reasoning_summaries
    except Exception as exc:
        if "404" not in str(exc):
            raise
        logger.info(
            f"Responses API returned 404 for {model}, falling back to Chat Completions"
        )
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16384,
        }
        if output_schema is not None:
            chat_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_response",
                    "schema": output_schema,
                    "strict": True,
                },
            }
        if reasoning_effort is not None:
            chat_kwargs["reasoning_effort"] = reasoning_effort
        response = await client.chat.completions.create(**chat_kwargs)
        reasoning_summaries = None
        if reasoning_effort is not None:
            reasoning_content = getattr(
                response.choices[0].message, "reasoning_content", None
            )
            if reasoning_content is not None:
                reasoning_summaries = [{"type": "text", "text": reasoning_content}]
            else:
                reasoning_summaries = []
        return response.choices[0].message.content, reasoning_summaries


# ---------------------------------------------------------------------------
# Parsing + aggregation
# ---------------------------------------------------------------------------


def parse_judge_response(response_text: str) -> dict:
    """Parse the judge response JSON and range-check every per-rubric ``score``.

    When Structured Outputs are used upstream, the response is already
    schema-validated; this function only decodes the JSON and enforces the
    0-3 range for each rubric's ``score``.
    """
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        # Attach the raw text so callers can surface it for debugging models
        # that don't actually honor Structured Outputs (e.g. markdown-fenced JSON).
        exc.raw_response = response_text
        raise
    for r in parsed.get("rubrics") or []:
        score = int(r["score"])
        if score < 0 or score > 3:
            raise ValueError(f"Per-rubric score {score} out of range [0, 3]")
        r["score"] = score
    return parsed


def _aggregate_one_rubric(verdict: dict) -> dict:
    """Roll up cluster verdicts for a single per-rubric judge entry."""
    symptoms = verdict.get("symptoms", {}) or {}
    metrics = symptoms.get("metrics", []) or []
    logs = symptoms.get("logs", []) or []
    traces = symptoms.get("traces", []) or []

    metric_hits = [bool(m.get("family_match")) for m in metrics]
    log_hits = [
        bool(l.get("body_match")) and bool(l.get("service_name_match")) for l in logs
    ]
    trace_hits = [
        bool(t.get("call_chain_match")) and bool(t.get("smoking_gun_match"))
        for t in traces
    ]

    def _all(hits: list[bool]) -> bool | None:
        return all(hits) if hits else None

    def _any(hits: list[bool]) -> bool | None:
        return any(hits) if hits else None

    return {
        "feature_flag": verdict.get("feature_flag", ""),
        "incident_time_within_10min": bool(verdict.get("incident_time_within_10min")),
        "feature_flag_match": bool(verdict.get("feature_flag_match")),
        "mechanism_match": bool(verdict.get("mechanism_match")),
        "metrics_all_match": _all(metric_hits),
        "metrics_any_match": _any(metric_hits),
        "logs_all_match": _all(log_hits),
        "logs_any_match": _any(log_hits),
        "traces_all_match": _all(trace_hits),
        "traces_any_match": _any(trace_hits),
        "score": int(verdict["score"]),
    }


def aggregate_judge_response(parsed: dict, *, mode: str | None = None) -> dict:
    """Aggregate per-rubric, per-cluster verdicts into flat task-level rollups.

    Per-rubric rollups are computed via ``_aggregate_one_rubric``. The
    task-level rollup is ``any`` over rubrics for most booleans (the agent
    is credited if it matches any one of the listed rubrics), with ``None``
    skipped — sections without clusters in any rubric stay ``None``.

    For ``feature_flag`` we expose ``rca_accuracy`` (did the agent name every
    listed root cause?) and ``hallucinate_any`` (did the agent name a root
    cause that doesn't match **any** of the listed plausible root causes?).
    Hallucination requires the report to be non-empty — pass
    ``mode="empty_report"`` from the caller so this function can force
    ``hallucinate_any = False`` for the empty-report short-circuit (where
    the synthesized nested otherwise looks identical to a fully wrong LLM
    judgment).

    Overall scoring is derived from per-rubric integer scores: ``rca_depth``
    is the float mean across rubrics (the canonical number consumed by
    formatters), and is ``None`` when there are no rubrics.

    The per-rubric rollups are also returned under ``per_rubric`` for
    downstream analysis that needs to know which specific rubric matched.
    """
    rubrics = parsed.get("rubrics") or []
    per_rubric = [_aggregate_one_rubric(r) for r in rubrics]

    def _any_skip_none(values: list[bool | None]) -> bool | None:
        truthy = [v for v in values if v is not None]
        return any(truthy) if truthy else None

    def _all_skip_none(values: list[bool | None]) -> bool | None:
        defined = [v for v in values if v is not None]
        return all(defined) if defined else None

    per_rubric_scores = [r["score"] for r in per_rubric]
    rca_depth = (
        sum(per_rubric_scores) / len(per_rubric_scores) if per_rubric_scores else None
    )

    # No-incident control tasks synthesize a single "(no_incident)" rubric in
    # judge() / string_match_fallback. Hallucination is undefined there
    # (no plausible root cause to match against), so leave it as None — that
    # renders as "—" in tables rather than the misleading 100% you'd get
    # from naively inverting the any-match across controls. Likewise an
    # empty incident report isn't a hallucination, just an absence of
    # output — callers signal that via mode="empty_report".
    is_no_incident = (
        len(per_rubric) == 1 and per_rubric[0].get("feature_flag") == "(no_incident)"
    )
    feature_flag_any_match = (
        _any_skip_none([r["feature_flag_match"] for r in per_rubric]) or False
    )
    if is_no_incident:
        hallucinate_any: bool | None = None
    elif mode == "empty_report":
        hallucinate_any = False
    else:
        hallucinate_any = not feature_flag_any_match

    return {
        "incident_time_within_10min": _any_skip_none(
            [r["incident_time_within_10min"] for r in per_rubric]
        )
        or False,
        "hallucinate_any": hallucinate_any,
        "rca_accuracy": _all_skip_none([r["feature_flag_match"] for r in per_rubric]),
        "mechanism_match": _any_skip_none([r["mechanism_match"] for r in per_rubric])
        or False,
        "metrics_all_match": _any_skip_none(
            [r["metrics_all_match"] for r in per_rubric]
        ),
        "metrics_any_match": _any_skip_none(
            [r["metrics_any_match"] for r in per_rubric]
        ),
        "logs_all_match": _any_skip_none([r["logs_all_match"] for r in per_rubric]),
        "logs_any_match": _any_skip_none([r["logs_any_match"] for r in per_rubric]),
        "traces_all_match": _any_skip_none([r["traces_all_match"] for r in per_rubric]),
        "traces_any_match": _any_skip_none([r["traces_any_match"] for r in per_rubric]),
        "per_rubric": per_rubric,
        "per_rubric_scores": per_rubric_scores,
        "rca_depth": rca_depth,
    }


_REWARD_METRIC_KEYS = ("rca_accuracy", "hallucinate_any")


def build_rewards(agg: dict, reward: float) -> dict[str, float | int]:
    """Flat {name: number} payload for ``/logs/verifier/reward.json``.

    Harbor requires a flat dict of numbers (``harbor.models.verifier.result
    .VerifierResult``) and hard-codes its primary-score lookup to the
    ``reward`` key, so that key is always present. Rollup values that are
    ``None`` — e.g. ``hallucinate_any`` on control tasks, where hallucination
    is undefined — are omitted, because a JSON ``null`` fails harbor's
    validation outright. Note that harbor aggregates a missing key as 0
    rather than excluding the trial.
    """
    rewards: dict[str, float | int] = {"reward": reward}
    for key in _REWARD_METRIC_KEYS:
        value = agg.get(key)
        if value is not None:
            rewards[key] = int(value)
    return rewards


def _synth_rubric_verdict(feature_flag: str, score: int) -> dict:
    """Build a synthetic per-rubric verdict shaped like the LLM judge output.

    All match booleans default to False and the symptom cluster lists are
    empty — callers that need to convey "no investigation" (string-match
    fallback, no-incident, empty-report) use this so that
    ``aggregate_judge_response`` produces well-typed rollups.
    """
    return {
        "feature_flag": feature_flag,
        "incident_time_within_10min": False,
        "feature_flag_match": False,
        "mechanism_match": False,
        "symptoms": {"metrics": [], "logs": [], "traces": []},
        "score": int(score),
    }


def string_match_fallback(expected: dict, predictions: str) -> dict:
    """Fall back to case-insensitive string search for plausible root causes.

    ``expected["events"]`` is a list of ``{"root_cause": <flag>, "event_time": ...}``.
    Each event becomes its own synthetic rubric: the rubric scores 3 if its
    ``root_cause`` substring is found in the report (case-insensitive),
    otherwise 0. For a no-incident task we emit a single synthetic rubric
    scoring 3 iff the report is empty.

    Returns ``{"rubrics": [...]}`` — same shape as the parsed LLM judge
    response — so that ``aggregate_judge_response`` can derive ``rca_depth``
    (the mean across rubrics) uniformly.
    """
    events: list[dict] = expected.get("events") or []
    if not events:
        score = 3 if len(predictions.strip()) == 0 else 0
        return {"rubrics": [_synth_rubric_verdict("(no_incident)", score)]}
    text = predictions.lower()
    rubrics = [
        _synth_rubric_verdict(
            e["root_cause"],
            3 if e["root_cause"].strip().lower() in text else 0,
        )
        for e in events
    ]
    return {"rubrics": rubrics}


# ---------------------------------------------------------------------------
# Judge entry point (used by both modes)
# ---------------------------------------------------------------------------


async def _judge_control(
    client: Any,
    predictions: str,
    expected: dict,
    *,
    model: str,
    reasoning_effort: str | None,
) -> dict:
    """LLM judge for control (no-incident) tasks.

    The agent passes iff its report does not place any incident inside the
    quiet window ``(quiet_window_start, quiet_window_end)`` around
    ``current_time``. An empty report is a trivial pass (no LLM call).

    ``expected["current"]`` is required — a control task built by
    ``build_harbor_tasks.py`` always carries it, so a ``KeyError`` here means
    the task was built by an older pipeline and cannot be scored.
    ``quiet_window_start`` / ``quiet_window_end`` are optional: either bound is
    absent when the quiet window is open-ended on that side, which the prompt
    renders as ``"null"`` (i.e. -inf / +inf).
    """
    if not predictions.strip():
        return {
            "mode": "no_incident_llm_judge",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "nested": {"rubrics": [_synth_rubric_verdict("(no_incident)", 3)]},
        }

    current = expected["current"]
    qstart = expected.get("quiet_window_start")
    qend = expected.get("quiet_window_end")
    margin = timedelta(minutes=CONTROL_QUIET_WINDOW_MARGIN_MIN)
    qstart_adj = (
        (datetime.fromisoformat(qstart) + margin).isoformat()
        if qstart is not None
        else "null"
    )
    qend_adj = (
        (datetime.fromisoformat(qend) - margin).isoformat()
        if qend is not None
        else "null"
    )
    prompt = JUDGE_PROMPT_CONTROL.format(
        current_time=current,
        quiet_window_start=qstart_adj,
        quiet_window_end=qend_adj,
        predictions=predictions,
    )

    raw_response, reasoning_summary = await async_call_llm_judge(
        client,
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        output_schema=CONTROL_OUTPUT_SCHEMA,
    )
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        exc.raw_response = raw_response
        raise
    claims = bool(parsed.get("claims_incident_in_quiet_window"))
    score = 0 if claims else 3

    return {
        "mode": "no_incident_llm_judge",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "reasoning_summary": reasoning_summary,
        "judge_prompt": prompt,
        "judge_response_raw": raw_response,
        "claims_incident_in_quiet_window": claims,
        "judge_reasoning": parsed.get("reasoning", ""),
        "nested": {"rubrics": [_synth_rubric_verdict("(no_incident)", score)]},
    }


async def judge(
    client: Any,
    expected: dict,
    predictions: str,
    rubrics_data: list[dict],
    model: str = DEFAULT_MODEL,
    reasoning_effort: str | None = None,
) -> dict:
    """Run the LLM judge over one or more ground-truth rubrics, or
    short-circuit for no-incident and empty-report tasks.

    For control tasks (``expected["events"] == []``) the queried time
    ``expected["current"]`` and the optional quiet-window bounds drive
    :func:`_judge_control`.

    Returns:
        A dict with keys ``mode``, ``model``, ``nested``, and (for the LLM
        judge mode) ``reasoning_summary``, ``rubric_used``, ``judge_prompt``,
        ``judge_response_raw``. All scoring (the ``rca_depth`` mean and the
        per-section rollups) is derived post-hoc from ``nested`` by
        :func:`aggregate_judge_response`.

    """
    events: list[dict] = expected.get("events") or []
    if not events:
        return await _judge_control(
            client,
            predictions,
            expected,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    # Short-circuit when the agent emitted an empty report despite an incident.
    # No LLM call can rescue this; every rubric scores 0. We synthesize the
    # nested response shape (per-rubric all False, matching cluster counts)
    # so that ``aggregate_judge_response`` produces the expected all-False
    # rollups at load time.
    if not predictions.strip():
        synthetic_per_rubric = []
        for rubric in rubrics_data:
            symptoms = rubric.get("symptoms", {}) or {}
            synthetic_per_rubric.append(
                {
                    "feature_flag": rubric.get("feature_flag", ""),
                    "incident_time_within_10min": False,
                    "feature_flag_match": False,
                    "mechanism_match": False,
                    "symptoms": {
                        "metrics": [
                            {
                                "metric_family": m.get("metric_family", ""),
                                "family_match": False,
                            }
                            for m in (symptoms.get("metrics") or [])
                        ],
                        "logs": [
                            {"body_match": False, "service_name_match": False}
                            for _ in (symptoms.get("logs") or [])
                        ],
                        "traces": [
                            {"call_chain_match": False, "smoking_gun_match": False}
                            for _ in (symptoms.get("traces") or [])
                        ],
                    },
                    "score": 0,
                }
            )
        return {
            "mode": "empty_report",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "rubric_used": bool(rubrics_data),
            "nested": {"rubrics": synthetic_per_rubric},
        }

    prompt = build_judge_prompt(rubrics_data, predictions)
    output_schema = build_judge_output_schema(rubrics_data)

    raw_response, reasoning_summary = await async_call_llm_judge(
        client,
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        output_schema=output_schema,
    )
    parsed = parse_judge_response(raw_response)

    # Per-section rollups and the ``rca_depth`` mean are NOT spread into the
    # result — they're pure post-hoc derivations from ``nested`` via
    # ``aggregate_judge_response``, applied by formatters at load time so
    # schema changes don't require rewriting saved JSONs.
    return {
        "mode": "llm_judge",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "reasoning_summary": reasoning_summary,
        "rubric_used": bool(rubrics_data),
        "judge_prompt": prompt,
        "judge_response_raw": raw_response,
        "nested": parsed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main() -> None:
    """Entry point: load expected + predictions, score, write reward/details."""
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge verifier for Harbor incident-RCA-report tasks."
    )
    parser.add_argument("--expected", type=str, default="/tests/expected.json")
    parser.add_argument("--predictions", type=str, default="/app/report.md")
    parser.add_argument(
        "--rubrics-dir",
        type=str,
        default="/tests/rubrics",
        help=(
            "Directory containing one rubric JSON per plausible root cause "
            "(e.g. /tests/rubrics/<event_id>.json). The judge scores the "
            "agent against any one of them."
        ),
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Judge LLM model name."
    )
    parser.add_argument(
        "--effort",
        type=str,
        choices=["low", "medium", "high"],
        default="high",
        help="Reasoning effort level for the judge LLM (default: high).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["string_match", "llm_judge"],
        default="llm_judge",
        help="Scoring mode: string_match or llm_judge (default: llm_judge).",
    )
    parser.add_argument("--reward", type=str, default="/logs/verifier/reward.json")
    parser.add_argument(
        "--details", type=str, default="/logs/verifier/reward-details.json"
    )
    args = parser.parse_args()

    expected_path = Path(args.expected)
    predictions_path = Path(args.predictions)
    rubrics_dir = Path(args.rubrics_dir)
    reward_path = Path(args.reward)
    details_path = Path(args.details)

    try:
        # Load expected
        print(f"Loading expected results from {expected_path}...")
        with expected_path.open() as f:
            expected = json.load(f)

        # Load predictions (markdown report)
        print(f"Loading predictions from {predictions_path}...")
        predictions = predictions_path.read_text()

        # Load rubric JSONs — one per plausible root cause. Order is the
        # filesystem sort order, which matches the build_harbor_tasks.py
        # naming convention (event_id stems sort lexicographically).
        rubrics_data: list[dict] = []
        if expected.get("events"):
            if rubrics_dir.is_dir():
                rubric_paths = sorted(rubrics_dir.glob("*.json"))
                for p in rubric_paths:
                    with p.open() as f:
                        rubrics_data.append(json.load(f))
                print(f"Loaded {len(rubrics_data)} rubric(s) from {rubrics_dir}")
            else:
                print(f"Rubric directory missing: {rubrics_dir}")
        else:
            print("No incident events in expected results; skipping rubric load.")

        # Run judge
        if args.mode == "string_match":
            result = {
                "mode": "string_match",
                "nested": string_match_fallback(expected, predictions),
            }
        else:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
            result = await judge(
                client,
                expected,
                predictions,
                rubrics_data,
                model=args.model,
                reasoning_effort=args.effort,
            )

        # Derive ``rca_depth`` (mean) and ``reward`` from ``nested`` so all
        # modes share one scoring path.
        agg = aggregate_judge_response(result["nested"], mode=result.get("mode"))
        rca_depth = agg["rca_depth"] or 0.0
        reward = rca_depth / 3.0

        # Write rewards
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text(json.dumps(build_rewards(agg, reward)))

        # Write details
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(json.dumps(result, indent=2))

        print(f"RCA depth: {rca_depth:.2f}/3 (reward: {reward})")
        print(f"Mode: {result['mode']}")
        if result.get("reasoning_summary"):
            print(f"Reasoning summary: {result['reasoning_summary']}")

        if reward < 1.0:
            sys.exit(1)

    except Exception as exc:
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text(json.dumps({"reward": 0.0}))
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(
            json.dumps(
                {
                    "reward": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
        )
        print(f"Verifier error: {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
