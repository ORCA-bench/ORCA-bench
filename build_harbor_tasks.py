r"""Render Harbor task directories from per-task specs and answers.

Step 3 of the three-script pipeline (``generate_task_specs.py`` ->
``generate_answers.py`` -> ``build_harbor_tasks.py``).

Reads:
  * ``output_dir/task_specs/<task_id>.json``  (from generate_task_specs.py)
  * ``output_dir/answers/<task_id>.json``     (from generate_answers.py)
  * ``output_dir/json/<event_id>.json``       (per-event rubrics)

Writes the full Harbor task tree under ``output_dir/harbor/tasks/<task_id>/``
(instruction.md, task_meta.json, task.toml, environment/, tests/, solution/),
plus the top-level ``tasks.csv``, ``task_snapshot_map.json``, and
``used_snapshots.json`` under ``output_dir/harbor/``.

Each ``task.toml`` gets a ``[task]`` section whose package name is a hash of the
task id (``<org>/<hash>``), so ``harbor add`` accepts the tasks directly with no
separate ``harbor task update`` pass, and the hub identity does not leak the
feature flag. ``tasks.csv`` maps the real ``task_id`` to its hashed
``package_name``.

``tests/expected.json`` schema:

    {
      "events": [
        {"root_cause": "<flag>", "event_time": "<ISO 8601>"},
        ...
      ]
    }

A control task: ``{"events": []}``.

Examples::

    uv run python build_harbor_tasks.py -od OUTPUT_DIR -dd DATA_DIR \
      --templates-dir harbor-template-telemetry-only --force

"""

import csv
import hashlib
import json
import logging
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv
from harbor.models.task.config import PackageInfo, TaskConfig

from task_spec import LOCAL_TZ, LOCAL_TZ_LABEL, TaskSpec
from utils import get_base_parser, setup_logging

logger = logging.getLogger(__name__)


# ───────────────────────────── Template helpers ─────────────────────────────
def read_template(templates_dir: Path, src_path: str) -> str:
    """Read a template file relative to ``templates_dir``."""
    return (templates_dir / src_path).read_text()


def copy_template(templates_dir: Path, src_path: str, dest_path: Path) -> None:
    """Copy a template file relative to ``templates_dir`` to ``dest_path``."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(templates_dir / src_path, dest_path)


_LBRACE_SENTINEL = "\0LBRACE\0"
_RBRACE_SENTINEL = "\0RBRACE\0"


def _format_template(template: str, values: Mapping[str, Any]) -> str:
    """Render ``{name}`` placeholders without disturbing JSON/LaTeX braces."""
    protected = template.replace("{{", _LBRACE_SENTINEL).replace("}}", _RBRACE_SENTINEL)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            return match.group(0)
        value = values[key]
        if value is None:
            return ""
        return str(value)

    rendered = re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, protected)
    return rendered.replace(_LBRACE_SENTINEL, "{").replace(_RBRACE_SENTINEL, "}")


# ────────────────────────── Task package helpers ──────────────────────────
def task_name_hash(task_name: str) -> str:
    """Stable, GT-opaque id for a task.

    The raw task_id (e.g. ``d1-c1-admanualgc-on-00-hard_control_post60m_at``)
    leaks the feature flag / ground truth, so the published harbor-hub package
    name is a hash instead. Deterministic so the same task always maps to the
    same name across runs.
    """
    return hashlib.sha256(task_name.encode()).hexdigest()[:16]


def add_task_package(
    toml_path: Path, *, org: str, task_name: str, description: str = ""
) -> None:
    """Inject a ``[task]`` section so ``harbor add`` accepts the task directly.

    Mirrors ``harbor task update`` (set ``config.task`` + ``schema_version`` and
    re-serialize, preserving the other sections) but names the package by a hash
    of ``task_name`` so the identity published to harbor hub does not leak the
    feature flag.
    """
    config = TaskConfig.model_validate_toml(toml_path.read_text())
    config.task = PackageInfo(
        name=f"{org}/{task_name_hash(task_name)}", description=description
    )
    config.schema_version = "1.3"
    toml_path.write_text(config.model_dump_toml())


# ───────────────────────────── Task building ─────────────────────────────
_CSV_COLUMNS: list[str] = [
    "task_id",
    "package_name",
    "qid",
    "granularity",
    "section",
    "phrasing",
    "ttd_minutes",
    "offset_minutes",
    "range_window",
    "broad_category",
    "reported_period_start",
    "reported_period_end",
    "reported_correctly_specified",
    "user_facing_issue",
    "reported_styled",
    "reported",
    "current",
    "incident_time",
    "quiet_window_start",
    "quiet_window_end",
    "flag",
    "root_causes",
    "snapshot_name",
    "no_incident",
]


def build_task(
    task_dir: Path,
    spec: TaskSpec,
    events: list[dict],
    *,
    templates_dir: Path,
    docker_repo: str | None,
    data_dir_name: str | None,
    code_tag: str | None,
    code_only: bool,
    json_dir: Path | None,
    org: str,
) -> None:
    """Render a single Harbor task directory.

    ``events`` is the answer list ``[{"root_cause": <flag>, "event_time": <iso>}, ...]``
    from ``output_dir/answers/<task_id>.json``. ``events == []`` indicates a
    control / no-incident task.
    """
    incident_iso = spec.incident_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    current_local = spec.current_dt.astimezone(LOCAL_TZ)
    current_str = current_local.strftime("%b %d, %Y at %H:%M") + f" {LOCAL_TZ_LABEL}"

    # -- Add instruction.md --
    instruction_template = read_template(templates_dir, "instruction.md.template")
    instruction_template = instruction_template.replace(
        "issue was reported at {reported}", "issue was reported {reported}"
    )
    instruction = _format_template(
        instruction_template,
        {
            "user_facing_issue": spec.user_facing_issue or "",
            "current": current_str,
            "reported": spec.reported_styled,
            "flag": spec.flag,
            "flag_description": spec.flag_description,
            "incident_time": incident_iso,
        },
    )
    (task_dir / "instruction.md").write_text(textwrap.dedent(instruction))

    # -- Add task metadata --
    meta: dict[str, Any] = {
        "qid": spec.qid,
        "user_facing_issue": spec.user_facing_issue,
        "incident_time": incident_iso,
        "current": spec.current_dt.isoformat(),
        "ttd_minutes": spec.ttd_minutes,
        "section": spec.section,
        "phrasing": spec.phrasing,
        "reported_styled": spec.reported_styled,
        "reported_correctly_specified": spec.reported_correctly_specified,
        "events": events,
    }
    if spec.offset_minutes is not None:
        meta["offset_minutes"] = spec.offset_minutes
    if spec.reported_dt is not None:
        meta["reported"] = spec.reported_dt.isoformat()
    if spec.range_window is not None:
        meta["range_window"] = spec.range_window
    if spec.broad_category is not None:
        meta["broad_category"] = spec.broad_category
    if spec.reported_period is not None:
        meta["reported_period_start"] = spec.reported_period[0].isoformat()
        meta["reported_period_end"] = spec.reported_period[1].isoformat()
    if spec.flag is not None:
        meta["flag"] = spec.flag
    if spec.flag_description:
        meta["description"] = spec.flag_description
    if spec.granularity is not None:
        meta["granularity"] = spec.granularity
    if spec.quiet_window_start is not None:
        meta["quiet_window_start"] = spec.quiet_window_start.isoformat()
    if spec.quiet_window_end is not None:
        meta["quiet_window_end"] = spec.quiet_window_end.isoformat()
    (task_dir / "task_meta.json").write_text(json.dumps(meta, indent=2))

    # -- Add configuration --
    copy_template(templates_dir, "task.toml.template", task_dir / "task.toml")
    add_task_package(task_dir / "task.toml", org=org, task_name=task_dir.name)

    # -- Add Docker environment files --
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    dockerfile_template = read_template(templates_dir, "environment/Dockerfile")
    if code_only:
        dockerfile = _format_template(
            dockerfile_template,
            {
                "docker_repo": docker_repo,
                "code_tag": code_tag,
                "templates_dir_name": templates_dir.name,
            },
        )
    else:
        dockerfile = _format_template(
            dockerfile_template,
            {
                "snapshot_name": spec.snapshot_name,
                "docker_repo": docker_repo,
                "data_dir_name": data_dir_name,
                "templates_dir_name": templates_dir.name,
            },
        )
    (env_dir / "Dockerfile").write_text(dockerfile)
    copy_template(
        templates_dir,
        "environment/docker-compose.yaml",
        env_dir / "docker-compose.yaml",
    )

    # -- Add solution --
    expected: dict[str, Any] = {"events": events}
    wait_block = """\
# ── Wait for the entrypoint to finish setting up the environment ──
echo "[solve] Waiting for environment to be ready..."
for i in $(seq 1 180); do
    [ -f /tmp/env-ready ] && break
    sleep 1
done
if [ ! -f /tmp/env-ready ]; then
    echo "[solve] ERROR: Environment did not become ready within 180s" >&2
    exit 1
fi
echo "[solve] Environment is ready."
"""
    if not code_only and templates_dir.name == "harbor-template-telemetry-only":
        lockdown_block = """\

# ── Verify the agent cannot read the otel-demo source ──
echo "[solve] Verifying source lockdown..."
if ls /app/opentelemetry-demo/ >/dev/null 2>&1; then
    echo "[solve] ERROR: /app/opentelemetry-demo is readable by the agent user" >&2
    exit 1
fi
if ls "${CONTEXT_DIR}/opentelemetry-demo/" >/dev/null 2>&1; then
    echo "[solve] ERROR: ${CONTEXT_DIR}/opentelemetry-demo is readable by the agent user" >&2
    exit 1
fi
echo "[solve] Source lockdown verified."
"""
    else:
        lockdown_block = ""
    if not code_only:
        health_block = """\

# ── Verify all services are healthy ──
echo "[solve] Running health checks..."
source /tmp/env-ports
/app/check_health.sh || {
    echo "[solve] ERROR: Health checks failed" >&2
    exit 1
}
"""
    else:
        health_block = ""
    solution_script = f"""\
#!/bin/bash
set -euo pipefail

{wait_block}{lockdown_block}{health_block}
# ── Generate solution report via LLM ──
echo "[solve] Installing dependencies..."
pip install tabulate
echo "[solve] Generating report..."
python /solution/solve.py --rubric /solution/rubric.json
"""
    solution_dir = task_dir / "solution"
    solution_dir.mkdir(parents=True, exist_ok=True)
    (solution_dir / "solve.sh").write_text(solution_script)
    (solution_dir / "solve.sh").chmod(0o755)
    shutil.copy2(
        Path(__file__).resolve().parent / "solve.py", solution_dir / "solve.py"
    )
    copy_template(
        templates_dir,
        "tests/check_prediction.py",
        solution_dir / "check_prediction.py",
    )

    # -- Add tests --
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "expected.json").write_text(json.dumps(expected, indent=2))
    copy_template(
        templates_dir, "tests/check_prediction.py", tests_dir / "check_prediction.py"
    )
    copy_template(templates_dir, "tests/test.sh", tests_dir / "test.sh")
    (tests_dir / "test.sh").chmod(0o755)

    # -- Add rubrics for LLM judge --
    # ``tests/rubrics/<event_id>.json`` carries one rubric per plausible event
    # in ``expected.events``. The verifier (check_prediction.py) loads all of
    # them and scores the agent against any one of them. The oracle solver
    # (solve.py) is given just ``solution/rubric.json`` — the *first* event's
    # rubric (= the authored event when present), so the oracle still produces
    # a single coherent report.
    if json_dir is not None and events:
        tests_rubrics_dir = tests_dir / "rubrics"
        tests_rubrics_dir.mkdir(parents=True, exist_ok=True)
        solution_rubrics_dir = solution_dir / "rubrics"
        solution_rubrics_dir.mkdir(parents=True, exist_ok=True)
        for i, ev in enumerate(events):
            event_id = ev.get("event_id")
            flag = ev.get("root_cause")
            if not event_id:
                continue
            candidates = [
                json_dir / f"{event_id}-{flag}.json",
                json_dir / f"{event_id}.json",
            ]
            rubric_path = next((p for p in candidates if p.is_file()), None)
            if rubric_path is None:
                logger.warning(
                    f"No rubric found at any of: {[str(c) for c in candidates]}"
                )
                continue
            shutil.copy2(rubric_path, tests_rubrics_dir / f"{event_id}.json")
            shutil.copy2(rubric_path, solution_rubrics_dir / f"{event_id}.json")
            if i == 0:
                # Canonical rubric for the oracle solver. solve.sh reads this
                # path; only the first (authored) event's rubric is used.
                shutil.copy2(rubric_path, solution_dir / "rubric.json")


def _csv_row_for(spec: TaskSpec, events: list[dict]) -> dict[str, Any]:
    period_start, period_end = spec.reported_period or (None, None)
    return {
        "task_id": spec.task_id,
        "package_name": task_name_hash(spec.task_id),
        "qid": spec.qid,
        "granularity": spec.granularity,
        "section": spec.section,
        "phrasing": spec.phrasing,
        "ttd_minutes": spec.ttd_minutes,
        "offset_minutes": spec.offset_minutes,
        "range_window": spec.range_window,
        "broad_category": spec.broad_category,
        "reported_period_start": period_start.isoformat() if period_start else None,
        "reported_period_end": period_end.isoformat() if period_end else None,
        "reported_correctly_specified": spec.reported_correctly_specified,
        "user_facing_issue": spec.user_facing_issue,
        "reported_styled": spec.reported_styled,
        "reported": spec.reported_dt.isoformat() if spec.reported_dt else None,
        "current": spec.current_dt.isoformat(),
        "incident_time": spec.incident_dt.isoformat(),
        "quiet_window_start": (
            spec.quiet_window_start.isoformat()
            if spec.quiet_window_start is not None
            else None
        ),
        "quiet_window_end": (
            spec.quiet_window_end.isoformat()
            if spec.quiet_window_end is not None
            else None
        ),
        "flag": spec.flag,
        "root_causes": ";".join(e["root_cause"] for e in events),
        "snapshot_name": spec.snapshot_name,
        "no_incident": spec.no_incident,
    }


# ───────────────────────────── Build orchestration ─────────────────────────────
@dataclass
class _BuildOutputs:
    task_dirs: list[Path]
    used_snapshots: set[str]
    task_snapshot_map: dict[str, str | None]
    csv_rows: list[dict[str, Any]]


def _build_all(
    *,
    specs_dir: Path,
    answers_dir: Path,
    json_dir: Path,
    tasks_dir: Path,
    templates_dir: Path,
    docker_repo: str,
    code_tag: str,
    data_dir_name: str,
    code_only: bool,
    org: str,
) -> _BuildOutputs:
    out = _BuildOutputs(
        task_dirs=[],
        used_snapshots=set(),
        task_snapshot_map={},
        csv_rows=[],
    )
    spec_paths = sorted(specs_dir.glob("*.json"))
    if not spec_paths:
        logger.error(f"No specs found under {specs_dir}")
        return out

    for spec_path in spec_paths:
        spec = TaskSpec.read_json(spec_path)
        answer_path = answers_dir / f"{spec.task_id}.json"
        if not answer_path.is_file():
            logger.warning(
                f"Missing answer for task {spec.task_id} at {answer_path}; skipping"
            )
            continue
        answer = json.loads(answer_path.read_text())
        events = answer.get("events") or []

        # Skip non-control tasks with no plausible root causes. Control tasks
        # are expected to have ``events == []`` (they assert no incident
        # occurred); other tasks with empty events are degenerate (the LLM
        # rejected every candidate, or every candidate was filtered out
        # upstream) and shouldn't be rendered into the Harbor task tree.
        is_control = spec.section == "control" or spec.no_incident
        if not events and not is_control:
            logger.warning(
                f"Skipping {spec.task_id}: non-control task with empty "
                f"`events` (no plausible root causes from generate_answers.py)"
            )
            continue

        task_dir = tasks_dir / spec.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        build_task(
            task_dir,
            spec,
            events,
            templates_dir=templates_dir,
            docker_repo=docker_repo if (spec.snapshot_name or code_only) else None,
            data_dir_name=data_dir_name if spec.snapshot_name else None,
            code_tag=code_tag if code_only else None,
            code_only=code_only,
            json_dir=json_dir,
            org=org,
        )
        logger.info(f"Wrote task {spec.task_id} -> {task_dir}")
        out.task_dirs.append(task_dir)
        out.task_snapshot_map[spec.task_id] = spec.snapshot_name
        if spec.snapshot_name:
            out.used_snapshots.add(spec.snapshot_name)
        out.csv_rows.append(_csv_row_for(spec, events))

    return out


# ───────────────────────────── Main ─────────────────────────────
def main() -> None:
    """CLI entry point — see module docstring for usage."""
    parser = get_base_parser()
    parser.description = (
        "Render Harbor task directories from per-task specs and answers."
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=Path("harbor-template"),
        help="Path to the Harbor template directory (default: harbor-template).",
    )
    parser.add_argument(
        "--docker-repo",
        default="orcabench/sre-otel-snapshot",
        help="Docker Hub repository for pre-built snapshot images.",
    )
    parser.add_argument(
        "--code-tag",
        default="code-only",
        help="Docker image tag for the pre-built code-only image (default: code-only).",
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Build tasks with source code only (no telemetry snapshots/services).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output task directory if it already exists.",
    )
    parser.add_argument(
        "--org",
        default="orca-bench",
        help="Org for the [task] package name (default: orca-bench).",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    output_dir: Path = args.output_dir
    specs_dir = output_dir / "task_specs"
    answers_dir = output_dir / "answers"
    json_dir = output_dir / "json"
    if not specs_dir.is_dir():
        raise FileNotFoundError(f"specs directory missing: {specs_dir}")
    if not answers_dir.is_dir():
        raise FileNotFoundError(f"answers directory missing: {answers_dir}")
    if not json_dir.is_dir():
        logger.warning(
            f"rubric directory missing: {json_dir} — tasks will not have tests/rubric.json"
        )

    task_root = output_dir / "harbor"
    tasks_dir = task_root / "tasks"
    if task_root.exists():
        if args.force:
            logger.info(f"Removing existing task root at {task_root}...")
            shutil.rmtree(task_root)
        elif any(task_root.iterdir()):
            raise FileExistsError(
                f"{task_root} already exists. Re-run with --force to rebuild tasks."
            )
    tasks_dir.mkdir(parents=True, exist_ok=True)

    out = _build_all(
        specs_dir=specs_dir,
        answers_dir=answers_dir,
        json_dir=json_dir if json_dir.is_dir() else None,  # type: ignore[arg-type]
        tasks_dir=tasks_dir,
        templates_dir=args.templates_dir,
        docker_repo=args.docker_repo,
        code_tag=args.code_tag,
        data_dir_name=args.data_dir.name,
        code_only=args.code_only,
        org=args.org,
    )

    # -- Write used_snapshots.json and task_snapshot_map.json --
    if not args.code_only:
        if out.used_snapshots:
            used_snapshots_path = task_root / "used_snapshots.json"
            used_snapshots_path.write_text(
                json.dumps(sorted(out.used_snapshots), indent=2)
            )
            logger.info(
                f"Wrote {len(out.used_snapshots)} used snapshot(s) to {used_snapshots_path}"
            )
        task_snapshot_map_path = task_root / "task_snapshot_map.json"
        task_snapshot_map_path.write_text(json.dumps(out.task_snapshot_map, indent=2))
        logger.info(
            f"Wrote {len(out.task_snapshot_map)} task-snapshot mapping(s) to {task_snapshot_map_path}"
        )

    # -- Write per-task CSV index --
    csv_path = task_root / "tasks.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in out.csv_rows:
            writer.writerow(
                {k: ("" if row.get(k) is None else row[k]) for k in _CSV_COLUMNS}
            )
    logger.info(f"Wrote {len(out.csv_rows)} row(s) to {csv_path}")
    print(f"Wrote {len(out.task_dirs)} task(s) -> {tasks_dir}")


if __name__ == "__main__":
    load_dotenv()
    main()
