"""Export predictions and results from a Harbor jobs directory.

Usage:
    uv run harbor-export <jobs_dir> -od <output_dir> [--csv] [--registry-url <url>]
"""

import json
import random
import urllib.request
from argparse import ArgumentParser
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars

from .token_utils import get_predictions

disable_progress_bars()

EXPECTED_FIELDS = ["root_cause", "event_time"]
TASK_META_FIELDS = [
    "qid",
    "user_facing_issue",
    "incident_time",
    "current",
    "reported",
    "lag",
    "availability",
    "flag",
    "description",
]


def _load_registry(registry_source: str | Path) -> list[dict]:
    """Load and normalize a registry (HTTP URL or local path) to ``tasks``."""
    if isinstance(registry_source, Path) or not str(registry_source).startswith(
        ("http://", "https://")
    ):
        registry = json.loads(Path(registry_source).read_text())
    else:
        with urllib.request.urlopen(str(registry_source)) as resp:
            registry = json.loads(resp.read())
    return registry[0]["tasks"] if isinstance(registry, list) else registry["tasks"]


_HF_DATASETS_PREFIX = "https://huggingface.co/datasets/"


def _hf_repo_id(git_url: str) -> str | None:
    """Return ``<owner>/<name>`` if ``git_url`` is an HF dataset URL, else ``None``."""
    url = git_url.removesuffix(".git")
    return (
        url[len(_HF_DATASETS_PREFIX) :] if url.startswith(_HF_DATASETS_PREFIX) else None
    )


def _fetch_registry_data(
    registry_source: str | Path,
    task_names: set[str],
    revision: str = "main",
) -> tuple[dict[str, dict], dict[str, dict], dict[str, list[dict]]]:
    """Load task_meta.json, expected.json, and per-event rubrics for each task.

    Tasks built by ``examples/otel-demo/build_harbor_tasks.py`` carry one
    rubric per plausible event under ``tests/rubrics/<event_id>.json``;
    ``task_meta["events"]`` is the authoritative list of event ids to fetch.

    Remote (HuggingFace) tasks are fetched via
    :func:`huggingface_hub.hf_hub_download`, which caches files at
    ``~/.cache/huggingface/hub/`` keyed by ``(repo_id, revision, file)``,
    revalidates via ETag on each call, and retries 429s with backoff.

    Args:
        registry_source: URL to a remote ``registry.json`` or a local
            filesystem :class:`Path` pointing at one. When local, task paths
            are resolved relative to the registry file's parent directory.
        task_names: Set of task names to fetch data for.
        revision: HF dataset revision (branch, tag, or commit SHA). Pin to a
            commit SHA for fully reproducible runs.

    Returns:
        Tuple of (task_meta, expected, rubrics) dicts. ``rubrics`` maps
        task_name to a list of rubric dicts (empty for control tasks).

    """
    tasks = _load_registry(registry_source)

    is_local = isinstance(registry_source, Path) or not str(registry_source).startswith(
        ("http://", "https://")
    )

    # Build task_name -> (repo_id, base_path) for HF or task_name -> local_dir.
    hf_locators: dict[str, tuple[str, str]] = {}
    local_bases: dict[str, str] = {}
    for task in tasks:
        name = task["name"]
        if name not in task_names:
            continue
        if is_local:
            # Harbor registries store ``path`` relative to the cwd at upload
            # time (see ``prepare_harbor_tasks``), not to the registry file.
            local_bases[name] = str(Path(task["path"]))
        else:
            repo_id = _hf_repo_id(task["git_url"])
            if repo_id is None:
                print(f"Warning: skipping {name} — non-HF git_url: {task['git_url']}")
                continue
            hf_locators[name] = (repo_id, task["path"])

    def _fetch(name: str, relpath: str) -> dict:
        if is_local:
            return json.loads(Path(f"{local_bases[name]}/{relpath}").read_text())
        repo_id, base = hf_locators[name]
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{base}/{relpath}",
            repo_type="dataset",
            revision=revision,
        )
        return json.loads(Path(local_path).read_text())

    names = sorted(local_bases.keys() | hf_locators.keys())
    task_meta: dict[str, dict] = {}
    expected: dict[str, dict] = {}
    rubrics: dict[str, list[dict]] = {}
    for name in names:
        for label, relpath, out in (
            ("task_meta.json", "task_meta.json", task_meta),
            ("expected.json", "tests/expected.json", expected),
        ):
            try:
                out[name] = _fetch(name, relpath)
            except Exception as e:
                print(f"Warning: Could not fetch {label} for {name}: {e}")
                out[name] = {}

        event_ids = [
            ev.get("event_id")
            for ev in (task_meta.get(name, {}).get("events") or [])
            if ev.get("event_id")
        ]
        per_task: list[dict] = []
        for event_id in event_ids:
            filename = f"{event_id}.json"
            try:
                per_task.append(_fetch(name, f"tests/rubrics/{filename}"))
            except Exception as e:
                print(
                    f"Warning: Could not fetch tests/rubrics/{filename} for {name}: {e}"
                )
        rubrics[name] = per_task

    return task_meta, expected, rubrics


def _to_csv_rows(
    data: dict[str, dict],
    task_meta: dict[str, dict] | None = None,
    expected: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Convert get_predictions output to a DataFrame with exploded timeline columns."""
    rows = []
    for trial_id, entry in data.items():
        predictions = entry.get("predictions", "")
        result = entry.get("result", {})

        task_name = result.get("task_name", "")
        expected_entry = expected.get(task_name, {}) if expected else {}

        rows.append(
            {
                "trial_id": trial_id,
                "task_name": task_name,
                "result": result,
                "report": predictions,
                "expected_entry": expected_entry,
            }
        )

    # Build flat rows with result fields
    flat_rows = []
    for row in rows:
        flat: dict = {"trial_id": row["trial_id"]}

        # Flatten result.json via json_normalize
        if row["result"]:
            result_flat = pd.json_normalize(row["result"], sep=".").iloc[0].to_dict()
            flat.update(result_flat)

        # Add task metadata fields (qid, lag, availability, etc.)
        if task_meta is not None:
            meta = task_meta.get(row["task_name"], {})
            for field in TASK_META_FIELDS:
                flat[field] = meta.get(field, "")

        flat["report"] = row["report"]

        # Add expected fields (flat schema)
        if expected is not None:
            for field in EXPECTED_FIELDS:
                flat[f"expected_{field}"] = row["expected_entry"].get(field, "")

        flat_rows.append(flat)

    df = pd.DataFrame(flat_rows)
    # sort_cols = [c for c in ["qid", "lag", "availability"] if c in df.columns]
    # if sort_cols:
    #     df = df.sort_values(sort_cols, ignore_index=True)
    return df


def cli_main() -> None:
    """CLI entry point for exporting predictions and results."""
    parser = ArgumentParser(
        description="Export predictions and results from a Harbor jobs directory."
    )
    parser.add_argument(
        "jobs_dir",
        type=Path,
        help="Path to the Harbor jobs directory",
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        type=Path,
        default=Path("out"),
        help="Output directory (default: out)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export as CSV instead of JSON",
    )
    parser.add_argument(
        "--include-task-name",
        "-i",
        action="append",
        default=None,
        help="Task name to include from jobs dir (glob, repeatable)",
    )
    parser.add_argument(
        "--exclude-task-name",
        "-x",
        action="append",
        default=None,
        help="Task name to exclude from jobs dir (glob, repeatable)",
    )
    parser.add_argument(
        "--sample-n-tasks",
        type=int,
        default=None,
        help=(
            "If set, uniformly sample this many distinct task_names and keep "
            "only trials belonging to them. Applied after include/exclude."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for --sample-n-tasks (default: 1).",
    )
    reg_group = parser.add_mutually_exclusive_group()
    reg_group.add_argument(
        "--registry-url",
        type=str,
        default=None,
        help="HTTP URL to registry.json for fetching expected outputs and task metadata",
    )
    reg_group.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Local path to registry.json (task paths resolved relative to its parent directory)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help=(
            "HF dataset revision (branch, tag, or commit SHA) to fetch from. "
            "Pin to a commit SHA for reproducible runs (default: main)."
        ),
    )
    args = parser.parse_args()

    data = get_predictions(args.jobs_dir)

    if args.include_task_name or args.exclude_task_name:
        total = len(data)
        if args.include_task_name:
            data = {
                k: v
                for k, v in data.items()
                if any(
                    fnmatch(v.get("result", {}).get("task_name", ""), p)
                    for p in args.include_task_name
                )
            }
        if args.exclude_task_name:
            data = {
                k: v
                for k, v in data.items()
                if not any(
                    fnmatch(v.get("result", {}).get("task_name", ""), p)
                    for p in args.exclude_task_name
                )
            }
        print(f"Filtered to {len(data)} / {total} trials")

    # Fetch task metadata, expected outputs, and rubrics if a registry was provided
    task_meta = None
    expected = None
    rubric: dict[str, list[dict]] | None = None
    registry_source: str | Path | None = args.registry_url or args.registry_path
    if registry_source is not None:
        task_names = {
            entry.get("result", {}).get("task_name", "") for entry in data.values()
        } - {""}
        print(
            f"Fetching task metadata, expected outputs, and rubrics for {len(task_names)} tasks..."
        )
        task_meta, expected, rubric = _fetch_registry_data(
            registry_source, task_names, revision=args.revision
        )
        total_rubrics = sum(len(rs) for rs in rubric.values())
        print(
            f"Fetched {len(task_meta)} task metadata, {len(expected)} expected outputs, "
            f"and {total_rubrics} rubrics across {len(rubric)} tasks"
        )

        before = len(data)
        data = {
            trial_id: entry
            for trial_id, entry in data.items()
            if task_meta.get(entry.get("result", {}).get("task_name", ""))
        }
        dropped = before - len(data)
        if dropped:
            print(f"Dropped {dropped} trial(s) with empty task_meta")

        if True:
            before = len(data)
            data = {
                trial_id: entry
                for trial_id, entry in data.items()
                if task_meta.get(entry.get("result", {}).get("task_name", ""), {}).get(
                    "granularity"
                )
                != "medium"
            }
            dropped_medium = before - len(data)
            if dropped_medium:
                print(f"Dropped {dropped_medium} trial(s) with granularity == 'medium'")

        # Drop rubric items for feature flags that have no metrics/logs/traces
        # symptoms and are therefore unscoreable by the LLM judge, but only
        # when the task has at least one other rubric to fall back on. If an
        # excluded flag is the task's only rubric, keep it so the trial count
        # stays stable.
        #
        # Disabled: this filtering now lives upstream in
        # examples/otel-demo/generate_answers.py (_drop_excluded_flag_events),
        # so newly generated answers already exclude these flags. Kept here
        # (behind `if False`) for reference / older datasets.
        if False:
            excluded_feature_flags = {"imageSlowLoad"}
            if rubric is not None:
                items_dropped = 0
                tasks_kept_solo = 0
                for task_name, items in rubric.items():
                    non_excluded = [
                        r
                        for r in items
                        if r.get("feature_flag") not in excluded_feature_flags
                    ]
                    if non_excluded and len(non_excluded) < len(items):
                        rubric[task_name] = non_excluded
                        items_dropped += len(items) - len(non_excluded)
                    elif items and not non_excluded:
                        tasks_kept_solo += 1
                if items_dropped:
                    print(
                        f"Dropped {items_dropped} rubric item(s) for excluded "
                        f"feature flags {sorted(excluded_feature_flags)} "
                        f"(only where task had other rubrics)"
                    )
                if tasks_kept_solo:
                    print(
                        f"Kept {tasks_kept_solo} task(s) where an excluded "
                        f"flag was the only rubric"
                    )

    if args.sample_n_tasks is not None:
        task_names = sorted(
            {entry.get("result", {}).get("task_name", "") for entry in data.values()}
            - {""}
        )
        n = min(args.sample_n_tasks, len(task_names))
        rng = random.Random(args.seed)
        sampled = set(rng.sample(task_names, n))
        before = len(data)
        data = {
            trial_id: entry
            for trial_id, entry in data.items()
            if entry.get("result", {}).get("task_name", "") in sampled
        }
        print(
            f"Sampled {len(sampled)}/{len(task_names)} task_name(s) "
            f"(seed={args.seed}); kept {len(data)}/{before} trial(s)"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        output_path = args.output_dir / "all-predictions.csv"
        df = _to_csv_rows(data, task_meta, expected)
        df.to_csv(output_path, index=False)
    else:
        output_path = args.output_dir / "all-predictions.json"
        # Merge task_meta, expected, and rubric into data for JSON output
        if task_meta or expected or rubric:
            for entry in data.values():
                task_name = entry.get("result", {}).get("task_name", "")
                if task_meta:
                    entry["task_meta"] = task_meta.get(task_name, {})
                if expected:
                    entry["expected"] = expected.get(task_name, {})
                if rubric:
                    entry["rubric"] = rubric.get(task_name, [])
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Saved {len(data)} trials to {output_path}")


if __name__ == "__main__":
    cli_main()
