"""Recover per-tool_call terminal output from each trial's asciinema recording.

For every trial under ``jobs_dir/<batch>/<trial_id>/agent``, parse
``recording.cast`` and ``trajectory.json`` to align each ``tool_call_id``
with the ``"i"`` (input) events in the cast file. The output bytes between
each command's ``"i"`` event and the next one are the "true" output of that
command — uncontaminated by terminus-2's 40-row visible-pane fallback that
truncates ``trajectory.json``'s ``observation.results`` when multiple
parallel tool calls share a single capture buffer.

Results are written one JSON per trial at
``output_dir/recovered_outputs/{trial_id}.json``::

    {"<tool_call_id>": "<recovered output text (ANSI escapes stripped)>"}

Idempotent: skips trials whose output JSON already exists unless ``--force``
is passed. No LLM calls — this is a pure parser, so it's fast (~0.5s for
100 trials).

Examples::

    cd examples/otel-demo
    uv run python recover_outputs.py -od out-0516 -jd jobs-sub
    uv run python recover_outputs.py -od out-0516 -jd jobs-sub --force
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

from utils import get_base_parser, setup_logging

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _parse_cast(path: Path) -> list[tuple[float, str, str]]:
    """Return ``[(ts, "i"|"o"|..., data), ...]`` from an asciinema v2 recording."""
    events: list[tuple[float, str, str]] = []
    with path.open() as f:
        f.readline()  # discard header
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(e, list) and len(e) >= 3:
                events.append((float(e[0]), str(e[1]), str(e[2])))
    return events


def _trajectory_tool_calls(trajectory_path: Path) -> list[tuple[str, str]]:
    """Return ``[(tool_call_id, keystrokes), ...]`` in trajectory order."""
    d = json.loads(trajectory_path.read_text())
    pairs: list[tuple[str, str]] = []
    for step in d.get("steps", []):
        if step.get("source") != "agent":
            continue
        for tc in step.get("tool_calls", []) or []:
            tc_id = tc.get("tool_call_id")
            args = tc.get("arguments") or {}
            ks = args.get("keystrokes") or args.get("cmd") or ""
            if tc_id:
                pairs.append((str(tc_id), str(ks)))
    return pairs


def recover_outputs(agent_dir: Path) -> dict[str, str]:
    """Return ``{tool_call_id: recovered_output}`` for one trial.

    The alignment is greedy left-to-right: for each tool_call in trajectory
    order, find the next unconsumed ``"i"`` event whose data contains the
    first 30 chars of the tool_call's first keystroke line. Slice all
    ``"o"`` events strictly between this ``"i"`` event and the next one.
    Returns ``{}`` if either source file is missing or empty.
    """
    cast_path = agent_dir / "recording.cast"
    traj_path = agent_dir / "trajectory.json"
    if not (cast_path.is_file() and traj_path.is_file()):
        return {}
    events = _parse_cast(cast_path)
    if not events:
        return {}
    tcs = _trajectory_tool_calls(traj_path)
    i_events = [(ts, data) for ts, t, data in events if t == "i"]
    out: dict[str, str] = {}
    cursor = 0
    for tc_id, ks in tcs:
        first_line = ks.split("\n", 1)[0].strip()[:40]
        if not first_line:
            continue
        needle = first_line[:30]
        matched = None
        for k in range(cursor, len(i_events)):
            if needle in i_events[k][1]:
                matched = k
                break
        if matched is None:
            continue
        start_ts = i_events[matched][0]
        end_ts = (
            i_events[matched + 1][0]
            if matched + 1 < len(i_events)
            else events[-1][0] + 1
        )
        chunks = [
            data for ts, t, data in events if t == "o" and start_ts < ts <= end_ts
        ]
        out[tc_id] = _ANSI_RE.sub("", "".join(chunks))
        cursor = matched + 1
    return out


def main() -> None:
    """Recover per-tool_call output for every trial under ``jobs_dir``."""
    parser = get_base_parser()
    parser.description = (
        "Recover per-tool_call terminal output from each trial's asciinema "
        "recording.cast, sidestepping terminus-2's 40-row pane truncation."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing per-trial recovered_outputs/{trial_id}.json.",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    jobs_dir = args.jobs_dir.resolve()
    if not jobs_dir.is_dir():
        logger.error(f"No jobs dir at {jobs_dir}")
        sys.exit(1)

    out_dir = args.output_dir / "recovered_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_trials = n_recovered = n_skipped = n_missing = 0
    n_outputs = 0
    for batch_dir in sorted(jobs_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        for trial_dir in sorted(batch_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            trial_id = trial_dir.name
            n_trials += 1
            agent_dir = trial_dir / "agent"
            out_path = out_dir / f"{trial_id}.json"
            if out_path.is_file() and not args.force:
                n_skipped += 1
                continue
            recovered = recover_outputs(agent_dir)
            if not recovered:
                n_missing += 1
                continue
            out_path.write_text(json.dumps(recovered, indent=2, ensure_ascii=False))
            n_recovered += 1
            n_outputs += len(recovered)
    elapsed = time.time() - t0
    logger.info(
        f"Done in {elapsed:.1f}s — {n_trials} trials scanned, "
        f"recovered {n_outputs} tool_call outputs from {n_recovered} trials "
        f"({n_skipped} already done, {n_missing} missing recording.cast/trajectory.json)"
    )


if __name__ == "__main__":
    main()
