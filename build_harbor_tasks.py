r"""Render Harbor task directories from per-task specs and answers.

Step 3 of the three-script pipeline (``generate_task_specs.py`` ->
``generate_answers.py`` -> ``build_harbor_tasks.py``).

Reads:
  * ``output_dir/task_specs/<task_id>.json``  (from generate_task_specs.py)
  * ``output_dir/answers/<task_id>.json``     (from generate_answers.py)
  * ``output_dir/json/<event_id>.json``       (per-event rubrics)

Writes the full Harbor task tree under ``output_dir/harbor/tasks/<task_id>/``
(instruction.md, task.toml, environment/, tests/, solution/), plus the
top-level ``tasks.csv``, ``task_snapshot_map.json`` and ``used_snapshots.json``
under ``output_dir/harbor/``.

No dataset manifest is written by default -- tasks are first-class registry
packages, so ``harbor publish <task dirs>`` works standalone. ``--split``
additionally partitions the build at the incident level and writes four
dataset manifests (names derived from ``--org``, so the full build and the
public split can never claim the same name)::

    harbor/split.json                            # auditable record
    harbor/datasets/public/          <org>/<org>
    harbor/datasets/private-oracle/  <org>/<org>-private-oracle
    harbor/datasets/private/         <org>/<org>-private
    harbor/datasets/verified/        <org>/<org>-verified   # --verified-json only

``--split`` also renders an answer-free duplicate of every private-split task
at ``harbor/tasks/<task_id>-hidden/``, published as ``<hash>-hidden``. The
duplicate drops ``tests/expected.json``, ``tests/rubrics/`` and ``solution/``,
and keeps only ``category``, ``tags``, ``user_facing_issue``, ``current``,
``reported_styled`` and ``difficulty`` in ``task.toml`` ``[metadata]``. Those
are the answer-bearing files: a submitter who runs the oracle private tasks
holds the root cause for every one of them, so the held-out split ships as the
hidden variant and is scored out-of-band instead.

The hidden tasks build and run exactly like their oracle twins -- same
instruction, same telemetry -- but their in-container verifier cannot score:
``tests/check_prediction.py`` fails to open the missing ``expected.json``, and
its except branch writes ``reward.json`` = ``{"reward": 0.0}``. That is a real
0, not an absent verdict, so anything aggregating these trials must read the
score from the out-of-band judge rather than from the Hub's ``evals``.

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
      ],
      "current": "<ISO 8601>",
      "quiet_window_start": "<ISO 8601>",   # optional
      "quiet_window_end": "<ISO 8601>"      # optional
    }

``current`` (the queried time) is always written. The quiet-window bounds are
written only for control tasks that have them; an omitted bound means the
window is open-ended on that side. The verifier's control judge
(``check_prediction._judge_control``) requires ``current`` and treats a missing
bound as -inf / +inf.

A control task: ``{"events": [], "current": ..., "quiet_window_end": ...}``.

Examples::

    uv run python build_harbor_tasks.py -od OUTPUT_DIR -dd DATA_DIR \
      --templates-dir harbor-template-telemetry-only --force

    # Build and split into public / private / verified dataset manifests
    uv run python build_harbor_tasks.py -od OUTPUT_DIR -dd DATA_DIR \
      --templates-dir harbor-template --force --split \
      --verified-json human-eval-sample/output/sampled_tasks.json \
      --dataset-author "Albert Gong <ag2435@cornell.edu>"

"""

import csv
import hashlib
import json
import logging
import random
import re
import shutil
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv
from harbor.models.dataset.manifest import DatasetInfo, DatasetManifest, DatasetTaskRef
from harbor.models.task.config import Author, PackageInfo, TaskConfig
from harbor.publisher.packager import Packager

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
    toml_path: Path,
    *,
    org: str,
    task_name: str,
    description: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Inject a ``[task]`` section so ``harbor add`` accepts the task directly.

    Mirrors ``harbor task update`` (set ``config.task`` + ``schema_version`` and
    re-serialize, preserving the other sections) but names the package by a hash
    of ``task_name`` so the identity published to harbor hub does not leak the
    feature flag. When ``metadata`` is given, its keys are merged into the
    ``[metadata]`` section (on top of the template's category/tags).
    """
    config = TaskConfig.model_validate_toml(toml_path.read_text())
    config.task = PackageInfo(
        name=f"{org}/{task_name_hash(task_name)}", description=description
    )
    if metadata:
        config.metadata = {**config.metadata, **dict(metadata)}
    config.schema_version = "1.3"
    toml_path.write_text(config.model_dump_toml())


def _parse_authors(authors: list[str] | None) -> list[Author]:
    """Parse 'Name <email>' / 'Name' strings into Author objects (matches harbor)."""
    out: list[Author] = []
    for a in authors or []:
        m = re.match(r"^(.+?)\s*<(.+?)>\s*$", a)
        out.append(
            Author(name=m.group(1).strip(), email=m.group(2))
            if m
            else Author(name=a.strip())
        )
    return out


# ───────────────────────────── Task building ─────────────────────────────
# Public difficulty label derived from the question granularity. ``medium``
# granularity tasks (broad feature-area complaints) are filtered out
# downstream, so they get no difficulty label.
_GRANULARITY_TO_DIFFICULTY: dict[str, str | None] = {
    "easy": "easy",
    "hard": "medium",
    "universal": "hard",
    "medium": None,
}

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

    # -- Build task metadata (merged into task.toml [metadata] below) --
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
        if spec.granularity not in _GRANULARITY_TO_DIFFICULTY:
            logger.warning(
                f"{spec.task_id}: unknown granularity {spec.granularity!r}; "
                "leaving difficulty unset"
            )
        difficulty = _GRANULARITY_TO_DIFFICULTY.get(spec.granularity)
        if difficulty is not None:
            meta["difficulty"] = difficulty
    if spec.quiet_window_start is not None:
        meta["quiet_window_start"] = spec.quiet_window_start.isoformat()
    if spec.quiet_window_end is not None:
        meta["quiet_window_end"] = spec.quiet_window_end.isoformat()

    # -- Add configuration --
    copy_template(templates_dir, "task.toml.template", task_dir / "task.toml")
    add_task_package(
        task_dir / "task.toml", org=org, task_name=task_dir.name, metadata=meta
    )

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
    # ``current`` is required by the verifier's control-task judge
    # (check_prediction._judge_control); the quiet-window bounds are written
    # only when defined — an omitted bound means the window is open-ended on
    # that side, which the judge prompt renders as "null" (-inf / +inf).
    expected: dict[str, Any] = {
        "events": events,
        "current": spec.current_dt.isoformat(),
    }
    if spec.quiet_window_start is not None:
        expected["quiet_window_start"] = spec.quiet_window_start.isoformat()
    if spec.quiet_window_end is not None:
        expected["quiet_window_end"] = spec.quiet_window_end.isoformat()
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
python /solution/solve.py --rubrics-dir /solution/rubrics
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
    # them, scores the report against each independently, and averages the
    # per-rubric scores. A report covering only one cause therefore caps at
    # ``1/len(events)`` of full credit, so ``solution/rubrics/`` mirrors the
    # full set and the oracle solver (solve.py, via ``--rubrics-dir``) writes
    # one report covering every plausible cause. This matches what the
    # agent-facing prompt asks for: "There may be multiple root causes or no
    # root cause at all" (instruction.md.template).
    if json_dir is not None and events:
        tests_rubrics_dir = tests_dir / "rubrics"
        tests_rubrics_dir.mkdir(parents=True, exist_ok=True)
        solution_rubrics_dir = solution_dir / "rubrics"
        solution_rubrics_dir.mkdir(parents=True, exist_ok=True)
        for ev in events:
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
    dataset_refs: list[DatasetTaskRef]
    # Everything --split needs, collected during the build so the split
    # never has to re-read and re-join tasks/, tasks.csv and a manifest.
    split_tasks: list["Task"]


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
        dataset_refs=[],
        split_tasks=[],
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

        # Pin the task by name + content digest (same digest harbor add computes).
        content_hash, _ = Packager.compute_content_hash(task_dir)
        package_name = task_name_hash(spec.task_id)
        digest = f"sha256:{content_hash}"
        out.dataset_refs.append(
            DatasetTaskRef(name=f"{org}/{package_name}", digest=digest)
        )
        out.split_tasks.append(
            Task(
                task_id=spec.task_id,
                qid=spec.qid,
                granularity=spec.granularity,
                package_name=package_name,
                digest=digest,
            )
        )

    return out


# ───────────────────────────── Dataset splitting ─────────────────────────────
# Only reached under --split. The partition is incident-level: ~10 tasks share
# each qid (same root cause, same telemetry snapshot, different reported-time
# phrasing), so splitting by task would put the same incident on both sides and
# the held-out set would not actually be held out.

# Difficulty strata, in ascending-difficulty order. `medium` tasks exist in
# tasks.csv specs but were never rendered into the task tree, so they never reach
# this script.
GRANULARITIES = ("easy", "hard", "universal")

# Public-facing tier names. The internal `granularity` values predate the
# published naming: what the pipeline calls `hard` ships as `medium`, and
# `universal` ships as `hard`. Applies to dataset descriptions and READMEs only —
# split.json and the log table keep the internal values so they stay joinable
# against tasks.csv.
GRANULARITY_DISPLAY = {"easy": "easy", "hard": "medium", "universal": "hard"}

DEFAULT_AUTHORS = ("Albert Gong <ag2435@cornell.edu>",)

README_TEMPLATE = """\
# {name}

An agent benchmark for root cause analysis — {blurb}

## Quick Start

> [!NOTE]
> These instructions use GradientAI serverless inference from DigitalOcean, for both the agent
> LLM (`gradient_ai/...`) and the LLM judge in the task verifier. To run against another
> provider, swap the `-m` model name and export that provider's credentials instead.

> [!IMPORTANT]
> Harbor needs a one-time patch before the first run, and GradientAI needs a LiteLLM patch for
> `reasoning_effort` passthrough — see
> [Additional setup instructions for Harbor](https://github.com/ORCA-bench/ORCA-bench#additional-setup-instructions-for-harbor).

```bash
# Point the agent LLM and the verifier's judge at GradientAI serverless inference.
# Model access key: https://cloud.digitalocean.com/gen-ai/model-access-keys
export GRADIENT_AI_API_KEY=$MODEL_ACCESS_KEY
export OPENAI_API_KEY=$MODEL_ACCESS_KEY
export OPENAI_BASE_URL=https://inference.do-ai.run/v1

# Stage the snapshot's /app/ to a host-side cache keyed by image id.
SNAPSHOT_IMAGE=orcabench/sre-otel-snapshot:data-0418-harbor-template
docker pull "$SNAPSHOT_IMAGE"
CACHE_DIR="$HOME/.cache/sre-snapshot-cache/$(docker image inspect "$SNAPSHOT_IMAGE" -f '{{{{.Id}}}}' | cut -d: -f2 | cut -c1-12)"
[ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ] && {{ mkdir -p "$CACHE_DIR"; CID=$(docker create "$SNAPSHOT_IMAGE"); docker cp "$CID:/app/." "$CACHE_DIR/"; docker rm "$CID"; }}

# Run the Harbor trials with the cache (read-only) bind-mounted
SNAPSHOT_CACHE_HOST_DIR="$CACHE_DIR" harbor run \\
    --mounts-json "[{{\\"type\\":\\"bind\\",\\"source\\":\\"$CACHE_DIR\\",\\"target\\":\\"$CACHE_DIR\\",\\"read_only\\":true}}]" \\
    -d {name} \\
    -a terminus-2 \\
    -m gradient_ai/openai-gpt-5.5 \\
    --ak temperature=1 \\
    --ak reasoning_effort=medium \\
    --ak 'llm_kwargs={{"max_tokens": 16384}}'
```

Pass `-a` / `-m` explicitly — without them `harbor run` falls back to the built-in oracle agent.
`--ak` values are parsed as JSON, so nested settings like `llm_kwargs` can be given inline (mind
the single quotes). To sweep several models in one job, list them under `agents:` in a YAML job
config and use `-c` instead.

The bind mount uses `target` = `source` so the path is identical inside and outside the container,
letting the task entrypoint `cp -al` from the cache without translating paths.
`SNAPSHOT_CACHE_HOST_DIR` must be set — the entrypoint fails fast without it.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
"""


@dataclass
class Task:
    """One built Harbor task, joined from the task tree and ``tasks.csv``."""

    task_id: str
    qid: str
    granularity: str
    package_name: str
    digest: str


@dataclass
class Side:
    """One side of the split (public or private)."""

    tasks: list[Task] = field(default_factory=list)

    @property
    def qids(self) -> set[str]:
        return {t.qid for t in self.tasks}

    def by_granularity(self, granularity: str) -> list[Task]:
        return [t for t in self.tasks if t.granularity == granularity]


def load_pinned_qids(verified_json: Path, tasks: list[Task]) -> set[str]:
    """Resolve the verified-subset task ids to the incidents backing them.

    ``sampled_tasks.json`` includes ``medium`` entries that were never built;
    those are dropped rather than treated as missing.
    """
    if not verified_json.exists():
        logger.warning(f"{verified_json} not found — no incidents will be pinned")
        return set()

    sampled = json.loads(verified_json.read_text())
    wanted = {k for k, v in sampled.items() if v.get("difficulty") != "medium"}
    dropped = len(sampled) - len(wanted)
    if dropped:
        logger.info(f"Dropped {dropped} `medium` entry(ies) from {verified_json}")

    qid_by_task = {t.task_id: t.qid for t in tasks}
    missing = sorted(wanted - qid_by_task.keys())
    if missing:
        raise KeyError(
            f"{len(missing)} verified task(s) have no built task directory: "
            f"{missing[:3]}{'...' if len(missing) > 3 else ''}"
        )

    pinned = {qid_by_task[t] for t in wanted}
    logger.info(f"Pinned {len(pinned)} incident(s) from {len(wanted)} verified task(s)")
    return pinned


# ───────────────────────────── Splitting ─────────────────────────────
def split_by_incident(
    tasks: list[Task], pinned_qids: set[str], frac: float, seed: int
) -> tuple[Side, Side]:
    """Partition tasks into (public, private) at the incident level.

    Stratified by granularity. Within a stratum, pinned incidents go public
    first, then shuffled incidents are added while each one moves the running
    task count closer to ``frac`` of the stratum total.
    """
    strata: dict[str, dict[str, list[Task]]] = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        strata[task.granularity][task.qid].append(task)

    public, private = Side(), Side()
    rng = random.Random(seed)

    for granularity in GRANULARITIES:
        incidents = strata[granularity]
        total = sum(len(v) for v in incidents.values())
        target = frac * total

        # Sort before shuffling so the RNG draw doesn't depend on dict order.
        pinned = sorted(q for q in incidents if q in pinned_qids)
        rest = sorted(q for q in incidents if q not in pinned_qids)
        rng.shuffle(rest)

        selected = list(pinned)
        count = sum(len(incidents[q]) for q in selected)
        if count > target:
            logger.warning(
                f"{granularity}: pinned incidents already hold {count} task(s), "
                f"over the {target:.0f}-task target"
            )
        for qid in rest:
            if abs(count + len(incidents[qid]) - target) < abs(count - target):
                selected.append(qid)
                count += len(incidents[qid])

        chosen = set(selected)
        for qid, group in incidents.items():
            (public if qid in chosen else private).tasks.extend(group)

    return public, private


def check_split(
    public: Side, private: Side, tasks: list[Task], pinned_qids: set[str]
) -> None:
    """Fail loudly if the partition is not a clean, incident-disjoint cover."""
    pub_ids = {t.task_id for t in public.tasks}
    priv_ids = {t.task_id for t in private.tasks}

    if overlap := pub_ids & priv_ids:
        raise AssertionError(f"{len(overlap)} task(s) on both sides: {sorted(overlap)[:3]}")
    if (pub_ids | priv_ids) != {t.task_id for t in tasks}:
        raise AssertionError("split does not cover every built task")
    if shared := public.qids & private.qids:
        raise AssertionError(
            f"{len(shared)} incident(s) span both sides: {sorted(shared)[:3]}"
        )
    if leaked := pinned_qids - public.qids:
        raise AssertionError(f"pinned incident(s) not in public set: {sorted(leaked)}")

    logger.info("Split checks passed: disjoint, complete, incident-clean, pins honored")


def log_counts(public: Side, private: Side) -> list[dict[str, object]]:
    """Log the per-stratum counts table and return it for split.json."""
    table: list[dict[str, object]] = []
    header = (
        f"{'granularity':<12}{'pub_inc':>8}{'pub_tasks':>11}{'pub_pct':>9}"
        f"{'priv_inc':>10}{'priv_tasks':>12}"
    )
    logger.info(header)
    for granularity in GRANULARITIES:
        pub = public.by_granularity(granularity)
        priv = private.by_granularity(granularity)
        total = len(pub) + len(priv)
        pct = len(pub) / total if total else 0.0
        row = {
            "granularity": granularity,
            "public_incidents": len({t.qid for t in pub}),
            "public_tasks": len(pub),
            "public_frac": round(pct, 4),
            "private_incidents": len({t.qid for t in priv}),
            "private_tasks": len(priv),
            "total_tasks": total,
        }
        table.append(row)
        logger.info(
            f"{granularity:<12}{row['public_incidents']:>8}{row['public_tasks']:>11}"
            f"{pct:>8.1%}{row['private_incidents']:>10}{row['private_tasks']:>12}"
        )

    total_tasks = len(public.tasks) + len(private.tasks)
    logger.info(
        f"{'TOTAL':<12}{len(public.qids):>8}{len(public.tasks):>11}"
        f"{len(public.tasks) / total_tasks:>8.1%}"
        f"{len(private.qids):>10}{len(private.tasks):>12}"
    )
    return table


def write_dataset_dir(
    dataset_dir: Path,
    name: str,
    description: str,
    blurb: str,
    authors: list[Author],
    side: Side,
    *,
    org: str,
) -> None:
    """Write a publishable dataset directory (dataset.toml + README.md)."""
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest = DatasetManifest(
        dataset=DatasetInfo(name=name, description=description, authors=authors),
        tasks=[
            DatasetTaskRef(name=f"{org}/{t.package_name}", digest=t.digest)
            # Sort by package name so the manifest is stable across runs.
            for t in sorted(side.tasks, key=lambda t: t.package_name)
        ],
    )
    manifest._header = (
        f"# Dataset manifest for {name}\n"
        "# Add tasks using: harbor add <org>/<name>\n"
        "# Publish using: harbor publish\n\n"
    )
    (dataset_dir / "dataset.toml").write_text(manifest.to_toml())
    (dataset_dir / "README.md").write_text(
        README_TEMPLATE.format(name=name, blurb=blurb)
    )
    logger.info(f"Wrote {len(manifest.tasks)} task ref(s) -> {dataset_dir}")


def side_blurb(side: Side, label: str) -> str:
    """One-line composition summary used in the description and README.

    Uses the public-facing tier names from ``GRANULARITY_DISPLAY``.
    """
    counts = ", ".join(
        f"{len(side.by_granularity(g))} {GRANULARITY_DISPLAY[g]}" for g in GRANULARITIES
    )
    return (
        f"{label} split of {len(side.tasks)} tasks "
        f"across {len(side.qids)} incidents ({counts})."
    )


def verified_side(verified_json: Path | None, public: Side) -> Side | None:
    """The verified subset as a **view over the public side**, not a third split.

    ``--verified-json`` pins these incidents public (see ``load_pinned_qids``),
    so every verified task is also a public task. It is recorded under
    ``views`` in split.json rather than ``splits`` precisely because it
    overlaps: ``convert_job.build_routing`` assigns ``routing[task_id]`` while
    looping over ``splits``, so an overlapping entry there would silently
    reroute those trials to whichever dataset iterated last.

    ``sampled_tasks.json`` includes ``medium`` entries that were never built;
    those are dropped, matching ``load_pinned_qids``.
    """
    if verified_json is None:
        return None
    if not verified_json.exists():
        logger.warning(f"{verified_json} not found -- writing no verified view")
        return None

    sampled = json.loads(verified_json.read_text())
    wanted = sorted(k for k, v in sampled.items() if v.get("difficulty") != "medium")
    by_id = {t.task_id: t for t in public.tasks}
    missing = [t for t in wanted if t not in by_id]
    if missing:
        raise KeyError(
            f"{len(missing)} verified task(s) are not on the public side: "
            f"{missing[:3]}{'...' if len(missing) > 3 else ''}"
        )
    logger.info(f"Verified view: {len(wanted)} task(s) selected from the public side")
    return Side(tasks=[by_id[t] for t in wanted])


# Package-name suffix and content policy for the answer-free private variant.
# The oracle task carries its own ground truth -- tests/expected.json names the
# root causes and event times, tests/rubrics/<event>.json spells out the flag
# and mechanism, and solution/ holds a copy of both plus the oracle solver. A
# split is only held out if none of that ships, so all three go.
#
# instruction.md is safe to keep: its template substitutes only `current`,
# `{reported}` (bound to spec.reported_styled in build_task, NOT to the ISO
# `reported` metadata field) and `user_facing_issue` -- the feature flag
# never reaches it.
HIDDEN_SUFFIX = "-hidden"
HIDDEN_STRIP = ("tests/expected.json", "tests/rubrics", "solution")
# task.toml [metadata] is rebuilt from this allowlist, dropping qid,
# incident_time, events, flag, description, granularity and the rest.
#
# `reported_styled` -- not `reported` -- is the timing field kept here. Both
# say when the issue was reported, but `reported` is the ISO instant
# `incident_dt + offset_minutes` (task_spec.py), and offsets are only ever 10
# or 60 minutes, so it narrows the incident time to two candidates. The
# styled string is the exact text instruction.md already renders into the
# agent's prompt, so keeping it adds nothing the agent cannot read.
HIDDEN_METADATA_KEYS = (
    "category",
    "tags",
    "user_facing_issue",
    "current",
    "reported_styled",
    "difficulty",
)


def build_hidden_task(
    src_dir: Path, dst_dir: Path, *, org: str, package_name: str
) -> str:
    """Copy ``src_dir`` to ``dst_dir`` with every answer stripped; return its digest.

    The copy keeps instruction.md, environment/ and the verifier scripts, so it
    builds and runs exactly like the oracle task -- an agent sees the same
    prompt and the same telemetry. It just cannot be scored in-container.
    """
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    for rel in HIDDEN_STRIP:
        target = dst_dir / rel
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    toml_path = dst_dir / "task.toml"
    config = TaskConfig.model_validate_toml(toml_path.read_text())
    config.task = PackageInfo(name=f"{org}/{package_name}", description="")
    config.metadata = {
        k: v for k, v in config.metadata.items() if k in HIDDEN_METADATA_KEYS
    }
    config.schema_version = "1.3"
    toml_path.write_text(config.model_dump_toml())

    content_hash, _ = Packager.compute_content_hash(dst_dir)
    return f"sha256:{content_hash}"


def hidden_side(tasks_dir: Path, private: Side, *, org: str) -> Side:
    """Build one answer-free duplicate per private-split task.

    Same task_id and incident as its oracle twin -- only the package name
    (``<hash>-hidden``) and the content differ -- so this is recorded as a view,
    never as a split: ``convert_job.build_routing`` keys on task_id, and two
    entries for one task_id under ``splits`` would silently reroute those trials
    to whichever dataset iterated last.
    """
    side = Side()
    for task in sorted(private.tasks, key=lambda t: t.task_id):
        src = tasks_dir / task.task_id
        if not src.is_dir():
            raise FileNotFoundError(f"no built task dir for {task.task_id} at {src}")
        package_name = f"{task.package_name}{HIDDEN_SUFFIX}"
        digest = build_hidden_task(
            src,
            tasks_dir / f"{task.task_id}{HIDDEN_SUFFIX}",
            org=org,
            package_name=package_name,
        )
        side.tasks.append(
            Task(
                task_id=task.task_id,
                qid=task.qid,
                granularity=task.granularity,
                package_name=package_name,
                digest=digest,
            )
        )
    logger.info(
        f"Built {len(side.tasks)} answer-free task(s) -> "
        f"{tasks_dir}/*{HIDDEN_SUFFIX}"
    )
    return side


def _side_record(name: str, side: Side) -> dict[str, Any]:
    """One split.json entry: dataset name, counts, incident ids, task rows."""
    return {
        "dataset_name": name,
        "n_tasks": len(side.tasks),
        "n_incidents": len(side.qids),
        "incidents": sorted(side.qids),
        "tasks": [
            {
                "task_id": t.task_id,
                "qid": t.qid,
                "granularity": t.granularity,
                "package_name": t.package_name,
            }
            for t in sorted(side.tasks, key=lambda t: t.task_id)
        ],
    }


def write_splits(
    task_root: Path,
    tasks: list[Task],
    *,
    org: str,
    frac: float,
    seed: int,
    verified_json: Path | None,
    authors: list[Author],
) -> None:
    """Partition ``tasks`` and write datasets/ + split.json under ``task_root``.

    Dataset names derive from ``org`` rather than a flag: a hand-passed name let
    the full build and the public split both claim ``orca-bench/orca-bench``,
    where publishing the wrong manifest would overwrite the public dataset with
    the private tasks included. Deriving them makes that unrepresentable, and
    keeps the slug lowercase (the registry validates package selectors against
    a lowercase-only pattern).
    """
    # Only the published strata are split. A build can legitimately contain
    # other granularities -- `medium` tasks are rendered but deliberately never
    # published -- and they belong to no dataset. Dropping them here (loudly)
    # rather than in the caller keeps check_split's "covers every task"
    # assertion meaningful over the tasks that ARE being split. This guard
    # replaces the granularity validation that used to live in the standalone
    # splitter's load_tasks.
    skipped = [t for t in tasks if t.granularity not in GRANULARITIES]
    tasks = [t for t in tasks if t.granularity in GRANULARITIES]
    if skipped:
        by_gran = Counter(t.granularity for t in skipped)
        logger.warning(
            f"Excluding {len(skipped)} task(s) from the split -- granularity not "
            f"in {GRANULARITIES}: {dict(sorted(by_gran.items()))}"
        )
    if not tasks:
        raise ValueError(f"no tasks left to split (granularities: {GRANULARITIES})")

    pinned_qids = load_pinned_qids(verified_json, tasks) if verified_json else set()
    public, private = split_by_incident(tasks, pinned_qids, frac, seed)
    check_split(public, private, tasks, pinned_qids)
    table = log_counts(public, private)
    verified = verified_side(verified_json, public)
    hidden = hidden_side(task_root / "tasks", private, org=org)

    names = {
        "public": f"{org}/{org}",
        "private-oracle": f"{org}/{org}-private-oracle",
        "private": f"{org}/{org}-private",
        "verified": f"{org}/{org}-verified",
    }
    labels = {
        "public": "public",
        "private-oracle": "held-out private (with answers)",
        "private": "held-out private (answers removed)",
        "verified": "verified",
    }
    sides = [("public", public), ("private-oracle", private), ("private", hidden)]
    if verified is not None:
        sides.append(("verified", verified))

    datasets_dir = task_root / "datasets"
    for key, side in sides:
        blurb = side_blurb(side, labels[key])
        write_dataset_dir(
            datasets_dir / key,
            names[key],
            f"An agent benchmark for root cause analysis - {blurb}",
            blurb,
            authors,
            side,
            org=org,
        )

    split_path = task_root / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "harbor_dir": str(task_root),
                "frac": frac,
                "seed": seed,
                "counts": table,
                "excluded_task_ids": sorted(t.task_id for t in skipped),
                # `splits` is the disjoint cover, and only that -- convert_job.py
                # iterates it assuming each task_id appears once.
                "splits": {
                    key: _side_record(names[key], side)
                    for key, side in (("public", public), ("private-oracle", private))
                },
                # Views share task_ids with a split (verified is a subset of
                # public; private-hidden is a repackaging of private-oracle),
                # so they must stay out of `splits`.
                "views": {
                    key: _side_record(names[name_key], side)
                    for key, name_key, side in (
                        ("verified", "verified", verified),
                        ("private-hidden", "private", hidden),
                    )
                    if side is not None
                },
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(f"Wrote split record -> {split_path}")


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
    parser.add_argument(
        "--split",
        action="store_true",
        help=(
            "Partition the built tasks into public/private dataset manifests "
            "under harbor/datasets/ and write harbor/split.json. Without this, "
            "no dataset manifest is written and the tasks are published "
            "standalone."
        ),
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.70,
        help="--split: target fraction of tasks on the public side (default: 0.70).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="--split: RNG seed for incident shuffling (default: 0).",
    )
    parser.add_argument(
        "--verified-json",
        type=Path,
        default=None,
        help=(
            "--split: verified-subset tasks (JSON keyed by task id). Their "
            "incidents are pinned to the public side and they are written as "
            "the `verified` view."
        ),
    )
    parser.add_argument(
        "--dataset-author",
        action="append",
        default=None,
        help="--split: dataset author 'Name <email>' (repeatable).",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.split and not 0.0 < args.frac < 1.0:
        parser.error(f"--frac must be in (0, 1), got {args.frac}")

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

    # -- Optionally split into dataset manifests --
    # No manifest is written by default: tasks are first-class registry
    # packages, so `harbor publish <task dirs>` works standalone and a dataset
    # is only needed to group them.
    if args.split:
        write_splits(
            task_root,
            out.split_tasks,
            org=args.org,
            frac=args.frac,
            seed=args.seed,
            verified_json=args.verified_json,
            authors=_parse_authors(args.dataset_author or list(DEFAULT_AUTHORS)),
        )

    print(f"Wrote {len(out.task_dirs)} task(s) -> {tasks_dir}")
    print("\nNext steps:")
    print(f"  harbor add {tasks_dir} --scan       # upload the task content")
    if args.split:
        for key in ("public", "private-oracle", "private", "verified"):
            path = task_root / "datasets" / key
            if path.is_dir():
                print(f"  harbor publish {path} --private")
    else:
        print(f"  harbor publish {tasks_dir}       # publish the tasks standalone")


if __name__ == "__main__":
    load_dotenv()
    main()
