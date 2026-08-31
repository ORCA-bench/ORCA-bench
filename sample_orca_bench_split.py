r"""Split the full Harbor task build into public and held-out private datasets.

Runs after ``build_harbor_tasks.py``. Reads the built task tree under
``<harbor-dir>/`` (``tasks/``, ``tasks.csv``, ``dataset.toml``) and partitions it
into two ready-to-publish dataset manifests.

The split is **incident-level**, not task-level. Each task id maps to a ``qid``
(one injected incident), and ~10 tasks share each qid — same root cause, same
telemetry snapshot, different reported-time phrasing. Splitting by task would put
the same incident on both sides of the public/private line, so the held-out set
would not actually be held out. Splitting by qid guarantees no incident appears
in both datasets.

Within each ``granularity`` stratum (easy / hard / universal) the incidents
backing ``--verified-json`` are pinned to the public side, the rest are shuffled
with ``--seed``, and incidents are added while they move the running task count
closer to ``--frac`` of the stratum total. That keeps the difficulty mix of the
public set close to the full build while landing near the requested ratio.

Writes under ``--output-dir``::

    split.json          # auditable record: counts table + per-side ids
    public/dataset.toml # + README.md
    public/README.md
    private/dataset.toml
    private/README.md

Task digests are copied from ``<harbor-dir>/dataset.toml`` rather than recomputed,
so the manifests pin exactly the task revisions already in the registry.

Examples::

    uv run python sample_orca_bench_split.py
    uv run python sample_orca_bench_split.py --seed 1 --frac 0.8 --force

Then publish (tasks are already in the registry, so nothing is uploaded)::

    uvx harbor publish out-0623/harbor-split/public --private
    uvx harbor publish out-0623/harbor-split/private --private

"""

import csv
import json
import logging
import random
import shutil
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from harbor.models.dataset.manifest import DatasetInfo, DatasetManifest, DatasetTaskRef
from harbor.models.task.config import Author

from utils import setup_logging

logger = logging.getLogger(__name__)

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


# ───────────────────────────── Loading ─────────────────────────────
def load_tasks(harbor_dir: Path) -> list[Task]:
    """Load every built task, joining the task tree to tasks.csv and dataset.toml.

    The task tree is authoritative: ``tasks.csv`` also holds specs that were
    filtered out during rendering (all ``medium`` tasks, plus non-control tasks
    whose root causes came back empty), so rows without a directory are skipped.
    """
    tasks_dir = harbor_dir / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"task tree missing: {tasks_dir}")

    built = sorted(p.name for p in tasks_dir.iterdir() if p.is_dir())
    logger.info(f"Found {len(built)} built task(s) in {tasks_dir}")

    csv_path = harbor_dir / "tasks.csv"
    with csv_path.open(newline="") as f:
        rows = {row["task_id"]: row for row in csv.DictReader(f)}
    logger.info(f"Read {len(rows)} spec row(s) from {csv_path}")

    manifest = DatasetManifest.from_toml_file(harbor_dir / "dataset.toml")
    digests = {ref.short_name: ref.digest for ref in manifest.tasks}
    logger.info(f"Read {len(digests)} pinned digest(s) from {harbor_dir / 'dataset.toml'}")

    tasks: list[Task] = []
    for task_id in built:
        row = rows.get(task_id)
        if row is None:
            raise KeyError(f"{task_id} has a task directory but no tasks.csv row")
        package_name = row["package_name"]
        if package_name not in digests:
            raise KeyError(f"{task_id} ({package_name}) missing from dataset.toml")
        tasks.append(
            Task(
                task_id=task_id,
                qid=row["qid"],
                granularity=row["granularity"],
                package_name=package_name,
                digest=digests[package_name],
            )
        )

    unexpected = {t.granularity for t in tasks} - set(GRANULARITIES)
    if unexpected:
        raise ValueError(f"unexpected granularity value(s): {sorted(unexpected)}")
    return tasks


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


# ───────────────────────────── Writing ─────────────────────────────
def parse_authors(specs: tuple[str, ...] | list[str]) -> list[Author]:
    """Parse ``'Name <email>'`` (or bare ``'Name'``) author specs."""
    authors = []
    for spec in specs:
        spec = spec.strip()
        if spec.endswith(">") and "<" in spec:
            name, _, email = spec[:-1].partition("<")
            authors.append(Author(name=name.strip(), email=email.strip()))
        else:
            authors.append(Author(name=spec))
    return authors


def write_dataset_dir(
    dataset_dir: Path,
    name: str,
    description: str,
    blurb: str,
    authors: list[Author],
    side: Side,
) -> None:
    """Write a publishable dataset directory (dataset.toml + README.md)."""
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest = DatasetManifest(
        dataset=DatasetInfo(name=name, description=description, authors=authors),
        tasks=[
            DatasetTaskRef(name=f"orca-bench/{t.package_name}", digest=t.digest)
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


# ───────────────────────────── Main ─────────────────────────────
def main() -> None:
    """CLI entry point — see module docstring for usage."""
    parser = ArgumentParser(
        description="Split a built Harbor task tree into public/private datasets."
    )
    parser.add_argument(
        "--harbor-dir",
        type=Path,
        default=Path("out-0623/harbor"),
        help="Built task tree with tasks/, tasks.csv, dataset.toml "
        "(default: out-0623/harbor).",
    )
    parser.add_argument(
        "--verified-json",
        type=Path,
        default=Path("human-eval-sample/output/sampled_tasks.json"),
        help="Verified-subset tasks whose incidents are pinned to the public "
        "side (default: human-eval-sample/output/sampled_tasks.json).",
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        type=Path,
        default=Path("out-0623/harbor-split"),
        help="Where to write split.json and the two dataset dirs "
        "(default: out-0623/harbor-split).",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.70,
        help="Target fraction of tasks in the public split (default: 0.70).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for incident shuffling (default: 0).",
    )
    parser.add_argument(
        "--public-name",
        default="orca-bench/ORCA-bench",
        help="Dataset name for the public split (default: orca-bench/ORCA-bench).",
    )
    parser.add_argument(
        "--private-name",
        default="orca-bench/ORCA-bench-private",
        help="Dataset name for the held-out split "
        "(default: orca-bench/ORCA-bench-private).",
    )
    parser.add_argument(
        "--author",
        action="append",
        default=None,
        help="Dataset author 'Name <email>' (repeatable).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--log_level",
        "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    if not 0.0 < args.frac < 1.0:
        parser.error(f"--frac must be in (0, 1), got {args.frac}")

    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"{output_dir} already exists. Re-run with --force to overwrite."
            )
        logger.info(f"Removing existing output at {output_dir}...")
        shutil.rmtree(output_dir)

    tasks = load_tasks(args.harbor_dir)
    pinned_qids = load_pinned_qids(args.verified_json, tasks)

    public, private = split_by_incident(tasks, pinned_qids, args.frac, args.seed)
    check_split(public, private, tasks, pinned_qids)
    table = log_counts(public, private)

    authors = parse_authors(args.author or DEFAULT_AUTHORS)
    public_blurb = side_blurb(public, "public")
    private_blurb = side_blurb(private, "held-out private")

    write_dataset_dir(
        output_dir / "public",
        args.public_name,
        f"An agent benchmark for root cause analysis — {public_blurb}",
        public_blurb,
        authors,
        public,
    )
    write_dataset_dir(
        output_dir / "private",
        args.private_name,
        f"An agent benchmark for root cause analysis — {private_blurb}",
        private_blurb,
        authors,
        private,
    )

    split_path = output_dir / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "harbor_dir": str(args.harbor_dir),
                "frac": args.frac,
                "seed": args.seed,
                "counts": table,
                "splits": {
                    label: {
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
                    for label, name, side in (
                        ("public", args.public_name, public),
                        ("private", args.private_name, private),
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(f"Wrote split record -> {split_path}")

    print(f"\npublic  {len(public.tasks):>4} tasks / {len(public.qids):>3} incidents")
    print(f"private {len(private.tasks):>4} tasks / {len(private.qids):>3} incidents")
    print("\nNext steps:")
    print(f"  uvx harbor publish {output_dir / 'public'} --private")
    print(f"  uvx harbor publish {output_dir / 'private'} --private")


if __name__ == "__main__":
    main()
