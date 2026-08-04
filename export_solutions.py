#!/usr/bin/env python3
"""Export locally generated solutions as all-predictions.json for LLM judge scoring.

Reads solution markdown files from ``output_dir/solutions/`` and rubric JSONs
from ``output_dir/json/``, then assembles them into the ``all-predictions.json``
format expected by ``run_llm_judge.py``.

Examples::

    # Export solutions then score them:
    uv run python export_solutions.py -od share-8h
    uv run python run_llm_judge.py -od share-8h --force
"""

import json
import logging
import sys
from pathlib import Path

from utils import get_base_parser, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Assemble all-predictions.json from solution .md and rubric .json files."""
    parser = get_base_parser()
    parser.description = "Export locally generated solutions as all-predictions.json for LLM judge scoring."
    args = parser.parse_args()
    setup_logging(getattr(logging, args.log_level))

    output_dir: Path = args.output_dir
    solutions_dir = output_dir / "solutions"
    json_dir = output_dir / "json"

    if not solutions_dir.is_dir():
        logger.error(f"Solutions directory not found: {solutions_dir}")
        sys.exit(1)
    if not json_dir.is_dir():
        logger.error(f"JSON rubrics directory not found: {json_dir}")
        sys.exit(1)

    # Build a mapping from stem -> rubric JSON path
    rubric_by_stem: dict[str, Path] = {jf.stem: jf for jf in json_dir.glob("*.json")}

    solution_files = sorted(solutions_dir.glob("*.md"))
    if not solution_files:
        logger.error(f"No solution .md files found in {solutions_dir}")
        sys.exit(1)

    all_data: dict[str, dict] = {}
    for sol_path in solution_files:
        stem = sol_path.stem
        rubric_path = rubric_by_stem.get(stem)
        if rubric_path is None:
            logger.warning(f"No matching rubric JSON for {sol_path.name}, skipping")
            continue

        rubric_data = json.loads(rubric_path.read_text())
        flag = rubric_data.get("feature_flag", "")
        incident_time = rubric_data.get("incident_time", "")
        if not flag:
            # Without a feature_flag the entry would be control-shaped
            # (``events: []``), and the control judge needs a ``current`` that
            # an oracle rubric doesn't carry. Every real rubric has one.
            logger.warning(f"Rubric {rubric_path.name} has no feature_flag, skipping")
            continue

        all_data[stem] = {
            "predictions": sol_path.read_text(),
            "result": {"task_name": stem},
            "task_meta": {"flag": flag},
            "expected": {
                "events": [{"root_cause": flag, "event_time": incident_time}],
            },
            "rubric": [rubric_data],
        }
        logger.info(f"Added {stem} (flag: {flag})")

    out_path = output_dir / "all-predictions.json"
    with out_path.open("w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Wrote {len(all_data)} entries to {out_path}")


if __name__ == "__main__":
    main()
