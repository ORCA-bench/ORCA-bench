"""Split a converted jobs tree into per-split trees using symlinked trial dirs.

Dataset names are read from ``split.json`` rather than hardcoded, so this keeps
working when a dataset is republished under a different name.

Each output tree gets real job-level ``config.json``/``result.json`` — they must
differ per split (scoped dataset list, trial count) — while every trial dir is a
*relative* symlink back to the source tree, so no trial payload is duplicated.

Trial membership comes from each trial's ``lock.json`` ``task.source``, which the
conversion step set to the published dataset name.

Examples::

    uv run python split_jobs.py \
        --src .../jobs-sub-backfilled-scores-2 \
        --split .../out-0804/harbor-split/split.json \
        --out public=.../jobs-sub-backfilled-scores-public-2 \
        --out private=.../jobs-sub-backfilled-scores-private-2
"""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="Converted jobs root.")
    p.add_argument("--split", type=Path, required=True, help="Path to split.json.")
    p.add_argument(
        "--out",
        action="append",
        required=True,
        metavar="SPLIT=PATH",
        help="Output root for a split, e.g. public=/path/to/out. Repeatable.",
    )
    p.add_argument(
        "--force", action="store_true", help="Replace existing output trees."
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Report the plan, change nothing."
    )
    return p.parse_args()


def write_job_files(src_job: Path, dst_job: Path, dataset: str, n_trials: int) -> None:
    """Write job-level config/result scoped to a single dataset."""
    cfg = json.loads((src_job / "config.json").read_text())
    cfg["jobs_dir"] = str(dst_job.parent)
    cfg["datasets"] = [
        {
            "path": None,
            "name": dataset,
            "version": None,
            "ref": None,
            "registry_url": None,
            "registry_path": None,
            "overwrite": False,
            "download_dir": None,
            "task_names": ["*"],
            "exclude_task_names": None,
            "n_tasks": None,
        }
    ]
    res = json.loads((src_job / "result.json").read_text())
    res["n_total_trials"] = n_trials
    res["trial_results"] = []

    (dst_job / "config.json").write_text(json.dumps(cfg, indent=4, ensure_ascii=False))
    (dst_job / "result.json").write_text(json.dumps(res, indent=4, ensure_ascii=False))


def main() -> None:
    """Build one symlinked tree per requested split."""
    args = parse_args()
    split = json.loads(args.split.read_text())

    # split name -> output root, resolved through split.json's dataset_name
    targets: dict[str, Path] = {}
    for item in args.out:
        name, _, path = item.partition("=")
        if name not in split["splits"]:
            raise SystemExit(
                f"split {name!r} not in {args.split} ({list(split['splits'])})"
            )
        targets[split["splits"][name]["dataset_name"]] = Path(path)
    for ds, root in targets.items():
        print(f"{ds} -> {root}")

    for root in targets.values():
        if root.exists() and not args.dry_run:
            if not args.force:
                raise SystemExit(f"{root} exists; pass --force to replace it.")
            shutil.rmtree(root)

    grand: Counter = Counter()
    for src_job in sorted(args.src.iterdir()):
        if not (src_job.is_dir() and (src_job / "result.json").exists()):
            continue
        buckets: dict[str, list[Path]] = {ds: [] for ds in targets}
        other = 0
        for child in sorted(src_job.iterdir()):
            lock = child / "lock.json"
            if not (child.is_dir() and lock.exists()):
                if child.is_dir() and (child / "result.json").exists():
                    other += 1
                continue
            ds = json.loads(lock.read_text()).get("task", {}).get("source")
            if ds in buckets:
                buckets[ds].append(child)
            else:
                other += 1

        print(
            f"{src_job.name}: { {k: len(v) for k, v in buckets.items()} } unclassified={other}"
        )
        grand.update({k: len(v) for k, v in buckets.items()})
        grand["unclassified"] += other

        if args.dry_run:
            continue

        for ds, trials in buckets.items():
            if not trials:
                continue
            dst_job = targets[ds] / src_job.name
            dst_job.mkdir(parents=True, exist_ok=True)
            for t in trials:
                link = dst_job / t.name
                link.symlink_to(
                    Path(os.path.relpath(t, dst_job)), target_is_directory=True
                )
            write_job_files(src_job, dst_job, ds, len(trials))

    print(f"\nTOTAL: {dict(grand)}")
    if args.dry_run:
        print("(dry run - nothing written)")


main()
