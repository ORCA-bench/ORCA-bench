#!/usr/bin/env python3
"""Rewrite trials' ``verifier/`` dirs into check_prediction.py's output format.

Older trials carry the pre-``reward.json`` layout: a bare-float ``reward.txt``
plus a three-key ``details.json`` summary (``{reward, mode, score}``) that has
no ``nested``/``rubrics``, so the new files cannot be derived from it. They are
derived instead from the judge score files ``run_llm_judge.py`` wrote for the
same trials, which carry the full ``nested`` verdict.

A score file is ``run_llm_judge``'s wrapper around the exact ``result`` dict
:func:`check_prediction.judge` returns; stripping the wrapper recovers ``result``
and the two verifier files follow with no LLM call. The scoring itself is
imported from ``check_prediction`` rather than reimplemented, so this script
cannot drift from the verifier.

Trials are paired with score files by name: ``--scores-dir`` is searched
recursively for ``judge-<model>-<effort>-<trial_dir_name>.json``. A trial with
no score file is skipped (reported, not an error); a trial matching more than
one is skipped as ambiguous.

For each paired trial this writes ``reward.json`` + ``reward-details.json`` and
deletes the stale ``reward.txt`` + ``details.json``. ``report.md``, ``log.txt``
and ``test-stdout.txt`` are left alone — they are the real container transcript.

Usage::

    uv run python backfill_verifier_outputs.py --dry-run
    uv run python backfill_verifier_outputs.py
    uv run python backfill_verifier_outputs.py --trial d1-c1-admanualgc-on-00-hard_cont__EX7aoRB
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from check_prediction import aggregate_judge_response, build_rewards
from utils import find_trial_dirs, setup_logging

logger = logging.getLogger(__name__)

_OTEL_DEMO = Path("/mnt/volume_nyc2_1777578495585/data/sre-agent/examples/otel-demo")
DEFAULT_JOBS_DIR = (
    _OTEL_DEMO / "jobs-leaderboard-backfilled-scores" / "2026-05-05__04-10-26"
)
DEFAULT_SCORES_DIR = _OTEL_DEMO / "out-leaderboard"

# Keys ``run_llm_judge.process_trial`` adds around the judge ``result``; every
# other key is part of ``result`` itself. ``score`` / ``rca_depth`` are stale
# top-level copies (``score`` predates the rename) — both are recomputed from
# ``nested`` rather than trusted.
#
# ``reasoning_effort`` is deliberately NOT stripped: check_prediction now emits
# it as part of ``result``, and utils.load_trials filters on it, so backfilled
# files must carry it too.
_WRAPPER_KEYS = frozenset(
    {
        "trial_id",
        "task_name",
        "batch_size",
        "batch_number",
        "flag",
        "score",
        "rca_depth",
    }
)

STALE_FILES = ("reward.txt", "details.json")


def build_score_index(
    scores_dir: Path, trial_names: set[str]
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Map each trial name to its ``judge-*.json``, found recursively.

    Score filenames are ``judge-<model>-<effort>-<trial_name>.json``, so the
    match requires the stem to end with ``-<trial_name>`` — a plain suffix test
    could pair a trial with another whose name merely ends the same way.

    Returns ``(index, ambiguous)``; ``ambiguous`` holds any trial matched by
    more than one score file (e.g. two judge models), which the caller skips
    rather than guessing between.
    """
    found: dict[str, list[Path]] = {}
    for path in sorted(scores_dir.rglob("judge-*.json")):
        stem = path.stem
        for name in trial_names:
            if stem.endswith(f"-{name}"):
                found.setdefault(name, []).append(path)
                break

    index = {n: paths[0] for n, paths in found.items() if len(paths) == 1}
    ambiguous = {n: paths for n, paths in found.items() if len(paths) > 1}
    return index, ambiguous


def load_judge_result(score_path: Path, trial_name: str) -> dict:
    """Recover the judge ``result`` dict from a ``judge-*.json`` score file.

    Guards that the score file really belongs to ``trial_name`` — pairing the
    wrong score with a trial would silently write someone else's verdict.
    """
    payload = json.loads(score_path.read_text())

    trial_id = payload.get("trial_id", "")
    if trial_name not in (trial_id, "") and not score_path.stem.endswith(trial_name):
        raise ValueError(
            f"score file {score_path.name} does not belong to trial {trial_name!r} "
            f"(trial_id={trial_id!r})"
        )

    result = {k: v for k, v in payload.items() if k not in _WRAPPER_KEYS}
    if "nested" not in result:
        # run_llm_judge writes error payloads (mode="llm_judge_error") with no
        # nested verdict; there is nothing to derive a reward from.
        raise ValueError(f"{score_path.name} has no 'nested' judge output")
    return result


def render_outputs(result: dict) -> tuple[str, str, float]:
    """Return ``(reward_json, details_json, rca_depth)`` exactly as the verifier writes them.

    Mirrors ``check_prediction.main()``: the reward payload is compact JSON and
    the details are ``indent=2``, neither with a trailing newline.
    """
    agg = aggregate_judge_response(result["nested"], mode=result.get("mode"))
    rca_depth = agg["rca_depth"] or 0.0
    reward = rca_depth / 3.0
    return (
        json.dumps(build_rewards(agg, reward)),
        json.dumps(result, indent=2),
        rca_depth,
    )


def backfill_trial(trial_dir: Path, score_path: Path, *, dry_run: bool) -> str:
    """Rewrite one trial's ``verifier/`` dir. Returns the judge ``mode``."""
    verifier_dir = trial_dir / "verifier"
    if not verifier_dir.is_dir():
        raise FileNotFoundError(f"verifier dir missing: {verifier_dir}")

    result = load_judge_result(score_path, trial_dir.name)
    reward_text, details_text, rca_depth = render_outputs(result)
    mode = result.get("mode", "?")
    logger.debug(f"{trial_dir.name}: {mode} rca_depth={rca_depth:.2f} {reward_text}")

    if dry_run:
        return mode

    (verifier_dir / "reward.json").write_text(reward_text)
    (verifier_dir / "reward-details.json").write_text(details_text)
    for stale in STALE_FILES:
        (verifier_dir / stale).unlink(missing_ok=True)
    return mode


def main() -> None:
    """CLI entry point — see module docstring for usage."""
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite trials' verifier/ dirs into check_prediction.py's "
            "reward.json + reward-details.json format."
        )
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=DEFAULT_JOBS_DIR,
        help="Job directory containing trial subdirectories.",
    )
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=DEFAULT_SCORES_DIR,
        help="Directory searched recursively for judge-*.json score files.",
    )
    parser.add_argument(
        "--trial",
        default=None,
        help="Only process this trial directory name (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without touching disk.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    if not args.jobs_dir.is_dir():
        raise FileNotFoundError(f"jobs dir missing: {args.jobs_dir}")
    if not args.scores_dir.is_dir():
        raise FileNotFoundError(f"scores dir missing: {args.scores_dir}")

    trial_dirs = find_trial_dirs(args.jobs_dir)
    if args.trial is not None:
        trial_dirs = [p for p in trial_dirs if p.name == args.trial]
        if not trial_dirs:
            raise FileNotFoundError(f"no trial {args.trial!r} under {args.jobs_dir}")

    names = {p.name for p in trial_dirs}
    if len(names) != len(trial_dirs):
        # Score files are matched by trial dir name, so a repeated name across
        # job dirs would pair both trials with the same verdict.
        seen: set[str] = set()
        dupes = sorted({p.name for p in trial_dirs if p.name in seen or seen.add(p.name)})
        logger.warning(
            f"{len(dupes)} trial name(s) appear in more than one job dir, "
            f"e.g. {dupes[:3]} — each will take the same score file"
        )
    index, ambiguous = build_score_index(args.scores_dir, names)
    logger.info(
        f"{len(trial_dirs)} trial(s) under {args.jobs_dir.name}; "
        f"{len(index)} paired with a score file from {args.scores_dir}"
    )

    modes: Counter[str] = Counter()
    skipped, failed = [], []
    for trial_dir in trial_dirs:
        if trial_dir.name in ambiguous:
            names_ = [p.name for p in ambiguous[trial_dir.name]]
            logger.warning(f"{trial_dir.name}: ambiguous, {len(names_)} score files")
            failed.append(trial_dir.name)
            continue
        score_path = index.get(trial_dir.name)
        if score_path is None:
            skipped.append(trial_dir.name)
            continue
        try:
            modes[backfill_trial(trial_dir, score_path, dry_run=args.dry_run)] += 1
        except Exception as exc:
            logger.warning(f"{trial_dir.name}: {type(exc).__name__}: {exc}")
            failed.append(trial_dir.name)

    verb = "would convert" if args.dry_run else "converted"
    logger.info(f"{verb}: {sum(modes.values())}  by mode: {dict(modes)}")
    if skipped:
        logger.info(f"skipped (no score file): {len(skipped)} e.g. {skipped[:3]}")
    if failed:
        logger.warning(f"failed: {len(failed)} e.g. {failed[:3]}")

    if args.dry_run or not modes:
        return

    # Authoritative check that harbor will accept every reward.json we wrote.
    from harbor.models.verifier.result import VerifierResult

    checked = 0
    for trial_dir in trial_dirs:
        path = trial_dir / "verifier" / "reward.json"
        if path.is_file():
            VerifierResult(rewards=json.loads(path.read_text()))
            checked += 1
    logger.info(f"harbor VerifierResult accepts all {checked} reward.json file(s)")


if __name__ == "__main__":
    main()
