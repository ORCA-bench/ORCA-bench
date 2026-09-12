"""Re-ID a job and all its trials so a re-upload writes to fresh storage paths.

Harbor's storage upload is skip-if-exists (``upload_bytes`` swallows the
"already exists" error), and deleting a Hub job does not delete its blobs. So
re-uploading a job whose ids are unchanged silently serves the OLD archives at
``jobs/{job_id}/job.tar.gz`` and ``trials/{trial_id}/trial.tar.gz``.

Changing both ids is therefore required, not optional, to publish updated
content for a job that was uploaded before.

New ids are UUIDv5 over stable names (job dir name, trial dir name) plus a tag,
so runs are idempotent and reproducible rather than random.

Examples::

    uv run python reid_trials.py --job-dir .../public/2026-05-05__04-10-26 --tag backfill --dry-run
    uv run python reid_trials.py --job-dir .../public/2026-05-05__04-10-26 --tag backfill
"""

import argparse
import json
import uuid
from pathlib import Path

NAMESPACE = uuid.UUID("6f1c2f52-6a3f-5e4a-9a1e-2f0b7c8d4e11")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-dir", type=Path, required=True, help="Job dir to re-id.")
    p.add_argument("--tag", required=True, help="Discriminator mixed into the new UUIDs.")
    p.add_argument("--dry-run", action="store_true", help="Report the plan, change nothing.")
    return p.parse_args()


def derive(*parts: str) -> str:
    """Derive a stable UUID from stable name components."""
    return str(uuid.uuid5(NAMESPACE, ":".join(parts)))


def main() -> None:
    """Rewrite the job id and every trial id under ``--job-dir``."""
    args = parse_args()
    job_dir = args.job_dir
    job_name = job_dir.name

    res_path = job_dir / "result.json"
    res = json.loads(res_path.read_text())
    old_job = res["id"]
    new_job = derive(job_name, args.tag, "job")
    print(f"job  {job_name}: {old_job} -> {new_job}")

    changed = 0
    unchanged = 0
    for child in sorted(job_dir.iterdir()):
        tr_path = child / "result.json"
        if not (child.is_dir() and tr_path.exists()):
            continue
        tr = json.loads(tr_path.read_text())
        new_tr = derive(job_name, child.name, args.tag, "trial")
        if tr["id"] == new_tr:
            unchanged += 1
            continue
        if not args.dry_run:
            tr["id"] = new_tr
            # Keep the embedded job pointer consistent with the new job id.
            if isinstance(tr.get("config"), dict) and "job_id" in tr["config"]:
                tr["config"]["job_id"] = new_job
            tr_path.write_text(json.dumps(tr, indent=4, ensure_ascii=False))
            cfg_path = child / "config.json"
            cfg = json.loads(cfg_path.read_text())
            if "job_id" in cfg:
                cfg["job_id"] = new_job
                cfg_path.write_text(json.dumps(cfg, indent=4, ensure_ascii=False))
        changed += 1

    if not args.dry_run and old_job != new_job:
        res["id"] = new_job
        res_path.write_text(json.dumps(res, indent=4, ensure_ascii=False))

    print(f"trials re-id'd: {changed}, already current: {unchanged}")
    if args.dry_run:
        print("(dry run - nothing written)")
    else:
        print(f"NEW_JOB_ID={new_job}")
        print(f"OLD_JOB_ID={old_job}")


main()
