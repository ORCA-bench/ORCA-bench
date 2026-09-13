"""Copy and convert harbor 0.6.3 job output into the format `harbor upload` (0.20.x) expects.

Does three things in one pass:

1. **Filter** — resolves each trial's task through ``split.json`` to a package
   name, then checks that package is actually published on the Harbor hub.
   Trials whose task is in no split, or whose package is not published, are
   excluded. Nothing unmappable reaches the destination.
2. **Copy** — rsyncs only the selected trial dirs to the destination.
3. **Convert** — rewrites ``config.json``/``result.json`` to point at the
   published package task and synthesizes the per-trial ``lock.json`` that
   harbor 0.6.3 never wrote, then writes the job-level files upload requires.

``--src``/``--dst`` accept either a single job dir or a root containing job
dirs; the layout is detected automatically and mirrored.

Re-running is idempotent: the copy restores pristine source files and the
convert step rewrites them, so the result does not depend on prior runs.

Examples::

    # one job, public split only
    uvx --from harbor python convert_job.py \
        --src .../jobs-sub/2026-05-05__04-10-26 \
        --dst .../jobs-leaderboard/2026-05-05__04-10-26 \
        --split .../split.json --splits public

    # every job under a root, both splits, dropping stale dirs at the dst
    uvx --from harbor python convert_job.py \
        --src .../jobs-sub --dst .../jobs-sub-backfilled-scores \
        --split .../split.json --prune

    # convert an already-copied tree in place
    uvx --from harbor python convert_job.py \
        --src .../jobs-sub --dst .../jobs-copy --split .../split.json --no-copy
"""

import argparse
import asyncio
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_ORG = "orca-bench"
PKG_PREFIX = f"{DATASET_ORG}/"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src", type=Path, required=True, help="Source job dir, or root of job dirs."
    )
    p.add_argument(
        "--dst", type=Path, required=True, help="Destination, mirroring --src layout."
    )
    p.add_argument("--split", type=Path, required=True, help="Path to split.json.")
    p.add_argument(
        "--splits",
        default=None,
        help="Comma-separated split names to include (default: every split in split.json).",
    )
    p.add_argument(
        "--no-copy", action="store_true", help="Skip the copy step; convert in place."
    )
    p.add_argument(
        "--prune",
        action="store_true",
        help="Delete trial dirs at the destination that are not in the selection.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Report the plan, change nothing."
    )
    return p.parse_args()


def is_job_dir(path: Path) -> bool:
    """True if ``path`` looks like a harbor job dir rather than a root of them."""
    return (path / "config.json").exists() and (path / "result.json").exists()


def job_pairs(src: Path, dst: Path) -> list[tuple[Path, Path]]:
    """Return [(src_job, dst_job)] for either a single job dir or a root of them."""
    if is_job_dir(src):
        return [(src, dst)]
    return [
        (c, dst / c.name) for c in sorted(src.iterdir()) if c.is_dir() and is_job_dir(c)
    ]


async def load_published(datasets: list[str]) -> dict[str, dict[str, str]]:
    """Return {dataset: {package_name: 'sha256:...'}} for each published dataset.

    A dataset that is not published on the hub yields an empty mapping rather
    than raising, so its tasks are simply filtered out.
    """
    from harbor.registry.client.package import PackageDatasetClient

    client = PackageDatasetClient()
    out: dict[str, dict[str, str]] = {}
    for ds in datasets:
        try:
            md = await client.get_dataset_metadata(ds)
            out[ds] = {t.name: t.ref for t in md.task_ids}
            print(f"  {ds}: {len(out[ds])} published tasks")
        except Exception as exc:  # noqa: BLE001 - any lookup failure means "not usable"
            out[ds] = {}
            print(
                f"  {ds}: NOT AVAILABLE ({type(exc).__name__}) - its tasks will be excluded"
            )
    return out


def build_routing(
    split_path: Path, wanted: list[str] | None, published: dict[str, dict[str, str]]
) -> dict[str, tuple[str, str, str]]:
    """Map old task id -> (package_name, dataset, digest), keeping only published tasks."""
    split = json.loads(split_path.read_text())
    routing: dict[str, tuple[str, str, str]] = {}
    for name, spec in split["splits"].items():
        if wanted is not None and name not in wanted:
            continue
        ds = spec["dataset_name"]
        for t in spec["tasks"]:
            digest = published.get(ds, {}).get(t["package_name"])
            if digest is not None:
                routing[t["task_id"]] = (t["package_name"], ds, digest)
    return routing


def package_task_config(pkg: str, digest: str, dataset: str) -> dict[str, Any]:
    """TaskConfig dict for a published package task.

    ``path``/``git_url``/``git_commit_id`` must be cleared: TaskConfig rejects
    having both ``path`` and ``name`` set.
    """
    return {
        "path": None,
        "git_url": None,
        "git_commit_id": None,
        "name": f"{PKG_PREFIX}{pkg}",
        "ref": digest,
        "overwrite": False,
        "download_dir": None,
        "source": dataset,
    }


def convert_trial(trial_dir: Path, pkg: str, digest: str, dataset: str) -> None:
    """Rewrite one trial's config/result and write the lock.json upload requires."""
    new_name = f"{PKG_PREFIX}{pkg}"
    task_cfg = package_task_config(pkg, digest, dataset)

    cfg = json.loads((trial_dir / "config.json").read_text())
    cfg["task"] = task_cfg

    res = json.loads((trial_dir / "result.json").read_text())
    res["task_name"] = new_name
    res["task_id"] = {"org": DATASET_ORG, "name": pkg, "ref": digest}
    res["task_checksum"] = digest.removeprefix("sha256:")
    res["source"] = dataset
    res["config"]["task"] = task_cfg

    lock = {
        "schema_version": 1,
        "task": {
            "name": new_name,
            "type": "package",
            "digest": digest,
            "source": dataset,
        },
        "install_only": cfg.get("install_only", False),
        "timeout_multiplier": cfg.get("timeout_multiplier", 1.0),
        "agent_timeout_multiplier": cfg.get("agent_timeout_multiplier"),
        "verifier_timeout_multiplier": cfg.get("verifier_timeout_multiplier"),
        "agent_setup_timeout_multiplier": cfg.get("agent_setup_timeout_multiplier"),
        "environment_build_timeout_multiplier": cfg.get(
            "environment_build_timeout_multiplier"
        ),
        "agent": cfg["agent"],
        "skills": [],
        "environment": cfg["environment"],
        "verifier": cfg["verifier"],
    }

    (trial_dir / "config.json").write_text(
        json.dumps(cfg, indent=4, ensure_ascii=False)
    )
    (trial_dir / "result.json").write_text(
        json.dumps(res, indent=4, ensure_ascii=False)
    )
    (trial_dir / "lock.json").write_text(json.dumps(lock, indent=4, ensure_ascii=False))


def write_job_files(
    src_job: Path, dst_job: Path, datasets: list[str], n_trials: int
) -> None:
    """Write the job-level config.json + result.json that `harbor upload` requires."""
    cfg = json.loads((src_job / "config.json").read_text())
    cfg["job_name"] = dst_job.name
    cfg["jobs_dir"] = str(dst_job.parent)
    cfg["datasets"] = [
        {
            "path": None,
            "name": ds,
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
        for ds in datasets
    ]

    res = json.loads((src_job / "result.json").read_text())
    res["n_total_trials"] = n_trials
    # The uploader enumerates trials by scanning subdirs; the source rollup
    # describes the full unfiltered run and would misreport this subset.
    res["trial_results"] = []

    (dst_job / "config.json").write_text(json.dumps(cfg, indent=4, ensure_ascii=False))
    (dst_job / "result.json").write_text(json.dumps(res, indent=4, ensure_ascii=False))


def select_trials(
    src_job: Path, routing: dict[str, tuple[str, str, str]]
) -> tuple[dict[str, tuple[str, str, str]], Counter]:
    """Return ({trial_dir_name: route}, reason_counts) for one source job."""
    selected: dict[str, tuple[str, str, str]] = {}
    reasons: Counter = Counter()
    for child in sorted(src_job.iterdir()):
        if not (child.is_dir() and (child / "result.json").exists()):
            continue
        task = json.loads((child / "result.json").read_text())["task_name"]
        route = routing.get(task)
        if route is None:
            reasons["excluded (no published task)"] += 1
            continue
        selected[child.name] = route
        reasons[route[1]] += 1
    return selected, reasons


def rsync_trials(src_job: Path, dst_job: Path, names: list[str]) -> None:
    """Copy the named trial dirs from src to dst.

    ``--files-from`` disables recursion even under ``-a``, so ``-r`` is passed
    explicitly; without it rsync silently creates empty directories.
    """
    dst_job.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(names) + "\n")
        list_path = fh.name
    try:
        subprocess.run(
            [
                "rsync",
                "-a",
                "-r",
                f"--files-from={list_path}",
                f"{src_job}/",
                f"{dst_job}/",
            ],
            check=True,
        )
    finally:
        Path(list_path).unlink(missing_ok=True)


async def main() -> None:
    """Filter, copy, and convert every job under ``--src``."""
    args = parse_args()
    wanted = args.splits.split(",") if args.splits else None

    split = json.loads(args.split.read_text())
    datasets = sorted(
        {
            spec["dataset_name"]
            for name, spec in split["splits"].items()
            if wanted is None or name in wanted
        }
    )
    print("Checking published datasets on the hub:")
    published = await load_published(datasets)
    routing = build_routing(args.split, wanted, published)
    print(f"{len(routing)} tasks routable to a published hub task\n")

    grand: Counter = Counter()
    for src_job, dst_job in job_pairs(args.src, args.dst):
        selected, reasons = select_trials(src_job, routing)
        used = sorted({ds for _, ds, _ in selected.values()})
        print(f"{src_job.name}: select {len(selected)} -> {dict(reasons)}")

        pruned = 0
        if dst_job.exists() and args.prune:
            for child in sorted(dst_job.iterdir()):
                if child.is_dir() and child.name not in selected:
                    pruned += 1
                    if not args.dry_run:
                        shutil.rmtree(child)
            if pruned:
                print(f"   pruned {pruned} stale trial dir(s) at destination")

        if args.dry_run:
            grand.update(reasons)
            grand["pruned"] += pruned
            continue

        if not args.no_copy:
            rsync_trials(src_job, dst_job, sorted(selected))

        converted = 0
        for name, (pkg, ds, digest) in selected.items():
            trial_dir = dst_job / name
            if not (trial_dir / "result.json").exists():
                print(f"   ! {name}: missing at destination, skipping")
                continue
            convert_trial(trial_dir, pkg, digest, ds)
            converted += 1

        write_job_files(src_job, dst_job, used or datasets, converted)
        print(f"   copied+converted {converted}")
        grand.update(reasons)
        grand["pruned"] += pruned

    print(f"\nTOTAL: {dict(grand)}")
    if args.dry_run:
        print("(dry run - nothing written)")


asyncio.run(main())
