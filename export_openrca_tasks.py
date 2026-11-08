#!/usr/bin/env python3
"""Generate OpenRCA ``query.csv`` / ``record.csv`` for an ORCA-bench split.

Companion to ``export_openrca.py``, which converts the telemetry. This converts
the task layer: given a Harbor dataset manifest, it writes the two CSVs
OpenRCA's ``rca.run_agent_standard`` reads beside ``telemetry/``.

    {SYSTEM}/
    |-- query.csv       task_index,instruction
    |-- record.csv      level,component,timestamp,datetime,reason,root_causes,event_count
    |-- task_map.csv    task_index,package_name,task_id,granularity,section,no_incident
    `-- telemetry/      (written by export_openrca.py)

``task_map.csv`` is not an OpenRCA file. OpenRCA identifies tasks as
``task_<int>`` and keeps no reference to where they came from, so without this
the predictions cannot be traced back to the Harbor task packages that hold
their rubrics -- which is exactly what scoring them with ORCA-bench's judge
needs.

No ``scoring_points`` column is written. OpenRCA's ``evaluate()`` matches three
fixed regex templates and compares components and reasons by string equality,
which cannot express a task with no root cause (8 of the verified 40) nor one
whose rubric lists several plausible root causes. Run the agent with
``--no-score`` and score the predictions with ORCA-bench's LLM judge instead.

Examples::

    uv run python export_openrca_tasks.py \\
        -d out-0919/harbor/datasets/verified/dataset.toml \\
        -t out-0919/harbor

    # Somewhere other than the default destination
    uv run python export_openrca_tasks.py -d <manifest> -t <harbor-dir> -od /tmp/out
"""

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from utils import get_base_parser, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("/mnt/volume_nyc3_1777562302334/OpenRCA/dataset")

# The timezone ORCA-bench phrases its reported times in. The telemetry is UTC;
# converting is part of the task, so the instruction keeps ET as ORCA-bench's
# own instruction.md does.
REPORT_TZ = ZoneInfo("America/New_York")

# Which service each feature flag's defect lives in, read off the `description`
# field of the task rubrics rather than inferred from the flag name -- they do
# not always agree. paymentUnreachable is read in checkout, which then cannot
# reach payment, and imageSlowLoad is a frontend React component adding a fault
# header rather than anything wrong in image-provider.
FLAG_COMPONENT = {
    "adFailure": "ad",
    "adManualGc": "ad",
    "cartFailure": "cart",
    "imageSlowLoad": "frontend",
    "paymentFailure": "payment",
    "paymentUnreachable": "checkout",
    "productCatalogFailure": "product-catalog",
    "recommendationCacheFailure": "recommendation",
}

# OpenRCA's record.csv has a `level` column naming the fault granularity. Bank
# is "pod" throughout; every ORCA-bench fault is a feature flag changing one
# service's behaviour.
LEVEL = "service"

# Control tasks have no root cause. Empty cells come back from pandas as NaN
# and render as "nan" in the groundtruth dump, which reads like missing data
# rather than a deliberate absence, so say so explicitly. Anything that tries
# to use a control's timestamp (run_sampling_* does, to locate telemetry) then
# fails loudly instead of coercing a NaN.
NO_INCIDENT = "(no incident)"

QUERY_HEADER = ["task_index", "instruction"]
RECORD_HEADER = [
    "level",
    "component",
    "timestamp",
    "datetime",
    "reason",
    "components",
    "root_causes",
    "event_count",
]
MAP_HEADER = [
    "task_index",
    "package_name",
    "task_id",
    "granularity",
    "section",
    "no_incident",
]


def manifest_package_names(manifest: Path) -> list[str]:
    """Package names listed under ``[[tasks]]`` in a Harbor dataset manifest.

    Parsed with a regex rather than tomllib so the dataset's own top-level
    ``name`` key cannot be mistaken for a task.
    """
    blocks = manifest.read_text().split("[[tasks]]")[1:]
    names = []
    for block in blocks:
        m = re.search(r'name\s*=\s*"([^"]+)"', block)
        if m:
            names.append(m.group(1))
    return names


def compose_instruction(row: dict) -> str:
    """Build the natural-language task prompt for one task.

    ``reported_styled`` already covers all five of ORCA-bench's phrasings ("at
    7:49 PM today ET", "sometime in the morning today", "sometime yesterday",
    "between 4:28 PM and 5:28 PM today ET"), and the relative ones only resolve
    against the current time, so that is stated first.

    The multi-root-cause and no-root-cause caveats are carried over from
    ORCA-bench's instruction.md: 8 of the verified 40 tasks have no incident at
    all and 22 have between 2 and 7 root causes, so an instruction promising a
    single failure -- as Bank's does -- would be wrong for most of the split.
    """
    current = datetime.fromisoformat(row["current"]).astimezone(REPORT_TZ)
    current_str = current.strftime("%b %-d, %Y at %H:%M ET")

    return (
        f"You are an expert site reliability engineer investigating an incident "
        f"in an e-commerce microservices system. The current time is {current_str}.\n"
        f"\n"
        f"The following issue was reported {row['reported_styled']}:\n"
        f"\n"
        f"{row['user_facing_issue']}\n"
        f"\n"
        f"Please analyse the telemetry and identify the root cause of this issue: "
        f"for each root cause, the time it occurred, the component responsible, "
        f"and the reason.\n"
        f"\n"
        f"NOTE:\n"
        f"* The time an issue was reported is not necessarily the time the "
        f"incident actually began.\n"
        f"* There may be multiple root causes, or no root cause at all. Report "
        f"every root cause you find, in chronological order.\n"
        f"* Times quoted above are ET. All telemetry timestamps are UTC."
    )


def task_events(harbor_dir: Path, task_id: str) -> list[dict]:
    """Ground-truth events for a task, from its package's tests/expected.json."""
    path = harbor_dir / "tasks" / task_id / "tests" / "expected.json"
    if not path.is_file():
        logger.warning(f"  no expected.json for {task_id}")
        return []
    return json.loads(path.read_text()).get("events") or []


def record_row(events: list[dict]) -> dict:
    """One record.csv row summarising a task's ground truth.

    OpenRCA indexes record.csv positionally against query.csv and only ever
    stringifies the row into a `groundtruth` column, so a task with several
    events still gets one row: the earliest event fills the canonical Bank
    columns, and `root_causes` / `event_count` carry the rest. A control task
    marks every column NO_INCIDENT.
    """
    if not events:
        return {
            "level": NO_INCIDENT,
            "component": NO_INCIDENT,
            "timestamp": NO_INCIDENT,
            "datetime": NO_INCIDENT,
            "reason": NO_INCIDENT,
            "components": NO_INCIDENT,
            "root_causes": NO_INCIDENT,
            "event_count": 0,
        }

    ordered = sorted(events, key=lambda e: e["event_time"])
    first = ordered[0]
    when = datetime.fromisoformat(first["event_time"])
    flag = first["root_cause"]
    missing = sorted({e["root_cause"] for e in ordered} - set(FLAG_COMPONENT))
    if missing:
        raise KeyError(
            f"No component mapping for flag(s) {missing}. Add them to "
            "FLAG_COMPONENT, taking the service from the flag's rubric description."
        )

    def stamp(event: dict) -> str:
        t = datetime.fromisoformat(event["event_time"]).astimezone(ZoneInfo("UTC"))
        return t.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "level": LEVEL,
        "component": FLAG_COMPONENT[flag],
        "timestamp": when.timestamp(),
        "datetime": when.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S"),
        "reason": flag,
        # The canonical Bank columns above describe only the earliest event, so
        # carry every event here: the same flag can fire several times at
        # different times, which a bare flag list renders as a confusing
        # repeat.
        "components": ";".join(
            sorted({FLAG_COMPONENT[e["root_cause"]] for e in ordered})
        ),
        "root_causes": ";".join(f'{e["root_cause"]}@{stamp(e)}' for e in ordered),
        "event_count": len(ordered),
    }


def main() -> None:
    """Write query.csv, record.csv and task_map.csv for a Harbor split."""
    parser = get_base_parser()
    parser.description = (
        "Generate OpenRCA query.csv / record.csv for an ORCA-bench split"
    )
    parser.set_defaults(output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dataset-manifest",
        "-d",
        type=Path,
        required=True,
        help="Harbor dataset.toml listing the tasks to convert",
    )
    parser.add_argument(
        "--harbor-dir",
        "-t",
        type=Path,
        required=True,
        help="Directory holding tasks/ and tasks.csv (e.g. out-0919/harbor)",
    )
    parser.add_argument(
        "--system",
        default="OtelDemo",
        help="OpenRCA {SYSTEM} directory name (default: OtelDemo)",
    )
    args = parser.parse_args()
    setup_logging(getattr(logging, args.log_level))

    names = manifest_package_names(args.dataset_manifest)
    packages = {n.split("/")[-1] for n in names}
    logger.info(f"{len(packages)} tasks in {args.dataset_manifest}")

    by_package = {
        r["package_name"]: r
        for r in csv.DictReader((args.harbor_dir / "tasks.csv").open())
        if r["package_name"] in packages
    }
    missing = packages - set(by_package)
    if missing:
        raise SystemExit(
            f"{len(missing)} manifest tasks absent from tasks.csv: {sorted(missing)[:5]}"
        )

    # task_<int> is OpenRCA's only task identity, and the report() helper bins
    # difficulty off that integer. Sort by package name so the numbering is
    # stable across runs and reproducible from the manifest alone.
    rows = [by_package[p] for p in sorted(packages)]

    root = Path(args.output_dir) / args.system
    root.mkdir(parents=True, exist_ok=True)

    n_events = 0
    with (root / "query.csv").open("w", newline="") as qf, \
         (root / "record.csv").open("w", newline="") as rf, \
         (root / "task_map.csv").open("w", newline="") as mf:
        qw = csv.DictWriter(qf, QUERY_HEADER, lineterminator="\n")
        rw = csv.DictWriter(rf, RECORD_HEADER, lineterminator="\n")
        mw = csv.DictWriter(mf, MAP_HEADER, lineterminator="\n")
        qw.writeheader()
        rw.writeheader()
        mw.writeheader()

        for i, row in enumerate(rows, 1):
            task_index = f"task_{i}"
            events = task_events(args.harbor_dir, row["task_id"])
            n_events += len(events)
            qw.writerow(
                {"task_index": task_index, "instruction": compose_instruction(row)}
            )
            rw.writerow(record_row(events))
            mw.writerow(
                {
                    "task_index": task_index,
                    "package_name": row["package_name"],
                    "task_id": row["task_id"],
                    "granularity": row["granularity"],
                    "section": row["section"],
                    "no_incident": row["no_incident"],
                }
            )

    controls = sum(1 for r in rows if r["no_incident"] == "True")
    logger.info(
        f"wrote {len(rows)} tasks ({controls} controls, {n_events} ground-truth "
        f"events) -> {root}/"
    )
    logger.info("Run the agent with --no-score; score with run_llm_judge.py.")


if __name__ == "__main__":
    main()
