r"""Generate per-task ``TaskSpec`` JSONs for the v2 Harbor pipeline.

Step 1 of the three-script pipeline (``generate_task_specs.py`` ->
``generate_answers.py`` -> ``build_harbor_tasks.py``).

Reads a YAML config + schedule + per-event question bundles from
``output_dir/questions/`` and writes one spec JSON per future Harbor task to
``output_dir/task_specs/<task_id>.json``. Each spec contains the time-axis
fields, the snapshot binding, and the (event, qid, granularity, flag)
linkage needed by the downstream scripts.

Examples::

    uv run python generate_task_specs.py \
      -od OUTPUT_DIR -dd DATA_DIR \
      --config configs/harbor_tasks_v2_share2w.yaml --force

"""

import bisect
import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv

from task_spec import (
    DAY_CATEGORIES,
    EXACT_PHRASINGS,
    LOCAL_TZ,
    LOCAL_TZ_LABEL,
    PERIOD_CATEGORIES,
    UNIVERSAL_QUESTIONS,
    TaskSpec,
    _fmt_time,
    active_flag_at,
    compute_flag_intervals,
    format_day,
    format_reported_time,
    load_event_timestamps,
    slug,
)
from utils import get_base_parser, parse_snapshot_time, setup_logging

logger = logging.getLogger(__name__)


# ───────────────────────────── YAML config schema ────────────────────────────
@dataclass
class ExactBundle:
    phrasings: list[str] = field(default_factory=lambda: list(EXACT_PHRASINGS))
    ttd_minutes: list[int] = field(default_factory=list)
    offset_minutes: list[int] = field(default_factory=lambda: [0])


@dataclass
class ExactSection:
    enabled: bool = True
    bundles: list[ExactBundle] = field(default_factory=list)


@dataclass
class ExactRangeWindow:
    range_window: int
    offset_minutes: list[int]


@dataclass
class ExactRangeBundle:
    ttd_minutes: list[int] = field(default_factory=list)
    configs: list[ExactRangeWindow] = field(default_factory=list)


@dataclass
class ExactRangeSection:
    enabled: bool = True
    bundles: list[ExactRangeBundle] = field(default_factory=list)


@dataclass
class BroadPeriodBundle:
    ttd_minutes: list[int] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)


@dataclass
class BroadPeriodSection:
    enabled: bool = True
    bundles: list[BroadPeriodBundle] = field(default_factory=list)


@dataclass
class BroadDayBundle:
    ttd_minutes: list[int] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)


@dataclass
class BroadDaySection:
    enabled: bool = True
    bundles: list[BroadDayBundle] = field(default_factory=list)


@dataclass
class ControlSection:
    enabled: bool = True
    pre_incident_minutes: list[int] = field(default_factory=list)
    post_incident_minutes: list[int] = field(default_factory=list)
    pre_anchor_hour_local: int = 9
    post_anchor_hour_local: int = 21
    phrasings: list[str] = field(default_factory=lambda: list(EXACT_PHRASINGS))


@dataclass
class V2Config:
    schedule: Path
    exact: ExactSection
    exact_range: ExactRangeSection
    broad_range_time_of_day: BroadPeriodSection
    broad_range_day: BroadDaySection
    control: ControlSection


_KNOWN_SECTIONS = {
    "exact",
    "exact_range",
    "broad_range_time_of_day",
    "broad_range_day",
    "control",
}
_KNOWN_TOPLEVEL = _KNOWN_SECTIONS | {"schedule"}


def _check_keys(name: str, raw: Mapping[str, Any], allowed: set[str]) -> None:
    extras = set(raw) - allowed
    if extras:
        raise ValueError(f"section '{name}' has unknown keys: {sorted(extras)}")


def _parse_exact_bundles(name: str, raw_bundles: list[Any]) -> list[ExactBundle]:
    out: list[ExactBundle] = []
    for i, b in enumerate(raw_bundles):
        if not isinstance(b, dict):
            raise ValueError(f"{name}.bundles[{i}] must be a mapping")
        _check_keys(
            f"{name}.bundles[{i}]",
            b,
            {"phrasings", "ttd_minutes", "offset_minutes"},
        )
        bundle = ExactBundle(
            phrasings=list(b.get("phrasings", EXACT_PHRASINGS)),
            ttd_minutes=list(b.get("ttd_minutes", [])),
            offset_minutes=list(b.get("offset_minutes", [0])),
        )
        bad = [p for p in bundle.phrasings if p not in EXACT_PHRASINGS]
        if bad:
            raise ValueError(f"{name}.bundles[{i}].phrasings has unknown values: {bad}")
        out.append(bundle)
    return out


def _parse_exact_range_bundles(
    name: str, raw_bundles: list[Any]
) -> list[ExactRangeBundle]:
    out: list[ExactRangeBundle] = []
    for i, b in enumerate(raw_bundles):
        if not isinstance(b, dict):
            raise ValueError(f"{name}.bundles[{i}] must be a mapping")
        _check_keys(f"{name}.bundles[{i}]", b, {"ttd_minutes", "configs"})
        windows: list[ExactRangeWindow] = []
        for j, w in enumerate(b.get("configs", []) or []):
            if not isinstance(w, dict):
                raise ValueError(f"{name}.bundles[{i}].configs[{j}] must be a mapping")
            _check_keys(
                f"{name}.bundles[{i}].configs[{j}]",
                w,
                {"range_window", "offset_minutes"},
            )
            windows.append(
                ExactRangeWindow(
                    range_window=int(w["range_window"]),
                    offset_minutes=list(w["offset_minutes"]),
                )
            )
        out.append(
            ExactRangeBundle(
                ttd_minutes=list(b.get("ttd_minutes", [])),
                configs=windows,
            )
        )
    return out


def _parse_broad_period_bundles(
    name: str, raw_bundles: list[Any]
) -> list[BroadPeriodBundle]:
    out: list[BroadPeriodBundle] = []
    for i, b in enumerate(raw_bundles):
        if not isinstance(b, dict):
            raise ValueError(f"{name}.bundles[{i}] must be a mapping")
        _check_keys(f"{name}.bundles[{i}]", b, {"ttd_minutes", "categories"})
        bundle = BroadPeriodBundle(
            ttd_minutes=list(b.get("ttd_minutes", [])),
            categories=list(b.get("categories", [])),
        )
        bad = [c for c in bundle.categories if c not in PERIOD_CATEGORIES]
        if bad:
            raise ValueError(
                f"{name}.bundles[{i}].categories has unknown values: {bad}"
            )
        out.append(bundle)
    return out


def _parse_broad_day_bundles(name: str, raw_bundles: list[Any]) -> list[BroadDayBundle]:
    out: list[BroadDayBundle] = []
    for i, b in enumerate(raw_bundles):
        if not isinstance(b, dict):
            raise ValueError(f"{name}.bundles[{i}] must be a mapping")
        _check_keys(f"{name}.bundles[{i}]", b, {"ttd_minutes", "categories"})
        bundle = BroadDayBundle(
            ttd_minutes=list(b.get("ttd_minutes", [])),
            categories=list(b.get("categories", [])),
        )
        bad = [c for c in bundle.categories if c not in DAY_CATEGORIES]
        if bad:
            raise ValueError(
                f"{name}.bundles[{i}].categories has unknown values: {bad}"
            )
        out.append(bundle)
    return out


def load_yaml_config(path: Path) -> V2Config:
    """Load and validate the v2 YAML config."""
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    extras = set(raw) - _KNOWN_TOPLEVEL
    if extras:
        raise ValueError(
            f"unknown top-level keys in config: {sorted(extras)}; "
            f"expected subset of {sorted(_KNOWN_TOPLEVEL)}"
        )

    if "schedule" not in raw:
        raise ValueError("config is missing required top-level key 'schedule'")
    schedule_path = Path(raw["schedule"])

    def _section(name: str) -> dict[str, Any]:
        block = raw.get(name) or {}
        if not isinstance(block, dict):
            raise ValueError(f"section '{name}' must be a mapping")
        return block

    exact_raw = _section("exact")
    _check_keys("exact", exact_raw, {"enabled", "bundles"})
    exact = ExactSection(
        enabled=bool(exact_raw.get("enabled", False)),
        bundles=_parse_exact_bundles("exact", exact_raw.get("bundles") or []),
    )

    er_raw = _section("exact_range")
    _check_keys("exact_range", er_raw, {"enabled", "bundles"})
    exact_range = ExactRangeSection(
        enabled=bool(er_raw.get("enabled", False)),
        bundles=_parse_exact_range_bundles("exact_range", er_raw.get("bundles") or []),
    )

    bp_raw = _section("broad_range_time_of_day")
    _check_keys("broad_range_time_of_day", bp_raw, {"enabled", "bundles"})
    broad_period = BroadPeriodSection(
        enabled=bool(bp_raw.get("enabled", False)),
        bundles=_parse_broad_period_bundles(
            "broad_range_time_of_day", bp_raw.get("bundles") or []
        ),
    )

    bd_raw = _section("broad_range_day")
    _check_keys("broad_range_day", bd_raw, {"enabled", "bundles"})
    broad_day = BroadDaySection(
        enabled=bool(bd_raw.get("enabled", False)),
        bundles=_parse_broad_day_bundles(
            "broad_range_day", bd_raw.get("bundles") or []
        ),
    )

    ctrl_raw = _section("control")
    _check_keys(
        "control",
        ctrl_raw,
        {
            "enabled",
            "pre_incident_minutes",
            "post_incident_minutes",
            "pre_anchor_hour_local",
            "post_anchor_hour_local",
            "phrasings",
        },
    )
    control = ControlSection(
        enabled=bool(ctrl_raw.get("enabled", False)),
        pre_incident_minutes=list(ctrl_raw.get("pre_incident_minutes", [])),
        post_incident_minutes=list(ctrl_raw.get("post_incident_minutes", [])),
        pre_anchor_hour_local=int(ctrl_raw.get("pre_anchor_hour_local", 9)),
        post_anchor_hour_local=int(ctrl_raw.get("post_anchor_hour_local", 21)),
        phrasings=list(ctrl_raw.get("phrasings", EXACT_PHRASINGS)),
    )
    bad_ctrl_phrasings = [p for p in control.phrasings if p not in EXACT_PHRASINGS]
    if bad_ctrl_phrasings:
        raise ValueError(f"control.phrasings has unknown values: {bad_ctrl_phrasings}")
    bad_ctrl_pre = [n for n in control.pre_incident_minutes if n <= 0]
    if bad_ctrl_pre:
        raise ValueError(
            f"control.pre_incident_minutes must be positive minutes; got {bad_ctrl_pre}"
        )
    bad_ctrl_post = [n for n in control.post_incident_minutes if n <= 0]
    if bad_ctrl_post:
        raise ValueError(
            f"control.post_incident_minutes must be positive minutes; got {bad_ctrl_post}"
        )
    for name, h in (
        ("pre_anchor_hour_local", control.pre_anchor_hour_local),
        ("post_anchor_hour_local", control.post_anchor_hour_local),
    ):
        if not 0 <= h <= 23:
            raise ValueError(f"control.{name} must be in [0, 23]; got {h}")

    return V2Config(
        schedule=schedule_path,
        exact=exact,
        exact_range=exact_range,
        broad_range_time_of_day=broad_period,
        broad_range_day=broad_day,
        control=control,
    )


# ───────────────────────────── Loaders ─────────────────────────────
_KEEP_GRANULARITIES: tuple[str, ...] = ("hard", "medium", "easy")


def load_questions(args: Any, schedule: Path) -> list[dict]:
    """Load question JSON files; sample one question per (event, granularity).

    For each event we uniformly sample one question at each of ``hard``,
    ``medium``, and ``easy``. A single ``universal`` question is added
    downstream so the final per-event budget is 4 questions (universal +
    hard + medium + easy). Sampling is reproducible when ``--seed`` is set.
    """
    questions_dir = args.output_dir / "questions"
    effort = args.effort or "medium"
    schedule_name = schedule.stem
    pattern = f"{schedule_name}-*-{args.model}-{effort}.json"
    question_files = sorted(questions_dir.glob(pattern))
    if not question_files:
        logger.error(f"No question files matching '{pattern}' in {questions_dir}")
        return []
    rng = random.Random(args.seed)
    questions: list[dict] = []
    for question_file in question_files:
        logger.debug(f"Loading questions from {question_file}...")
        with question_file.open() as f:
            bundle = json.load(f)
        event = bundle["event"]
        answer = bundle["answer"]
        by_gran: dict[str, list[dict]] = {}
        for q in bundle["questions"]:
            g = q["granularity"]
            if g in _KEEP_GRANULARITIES:
                by_gran.setdefault(g, []).append(q)
        for g in _KEEP_GRANULARITIES:
            options = by_gran.get(g)
            if not options:
                continue
            chosen = rng.choice(options)
            questions.append(
                {
                    "id": chosen["id"],
                    "question": chosen["question"],
                    "granularity": g,
                    "event": event,
                    "answer": answer,
                }
            )
    if args.qid is not None:
        questions = [q for q in questions if q["id"] in args.qid]
        logger.info(
            f"Filtered questions to {len(questions)} matching qid(s) {args.qid}."
        )
    return questions


def load_snapshots(data_dir: Path) -> tuple[dict[datetime, str], list[datetime]]:
    """Discover Prometheus snapshots and return (sim_seconds → name, sorted times)."""
    prom_snap_dir = data_dir / "prometheus" / "snapshots"
    snap_names = sorted(
        [d.name for d in prom_snap_dir.iterdir() if d.is_dir()],
        key=parse_snapshot_time,
    )
    snapshots: dict[datetime, str] = {}
    times: list[datetime] = []
    for name in snap_names:
        sim_seconds = parse_snapshot_time(name)
        snapshots[sim_seconds] = name
        times.append(sim_seconds)
    logger.info(f"Found {len(snap_names)} snapshots")
    return snapshots, times


def load_flag_descriptions() -> dict[str, str]:
    """Load flag descriptions from ``demo.flagd.json`` (cwd), if present."""
    flagd_path = Path("demo.flagd.json")
    descriptions: dict[str, str] = {}
    if flagd_path.is_file():
        with flagd_path.open() as f:
            flagd_data = json.load(f)
        for flag_name, flag_def in flagd_data.get("flags", {}).items():
            if "description" in flag_def:
                descriptions[flag_name] = flag_def["description"]
        logger.info(f"Loaded {len(descriptions)} flag description(s) from {flagd_path}")
    else:
        logger.warning(
            f"demo.flagd.json not found at {flagd_path}; flag descriptions will be empty"
        )
    return descriptions


# ───────────────────────────── Build context ─────────────────────────────
@dataclass
class _SpecBuildContext:
    """Inputs shared by every iter_* helper."""

    specs_dir: Path
    code_only: bool
    grace_period: int
    snapshots: dict[datetime, str]
    snapshot_times: list[datetime]
    flag_descriptions: dict[str, str]
    flag_intervals: dict[str, list[tuple[datetime, datetime]]]
    written_task_ids: set[str]


def _pick_snapshot(
    incident_dt: datetime,
    ttd: int,
    ctx: _SpecBuildContext,
    qid: str,
) -> tuple[str | None, bool]:
    """Pick the snapshot for ``(incident_dt, ttd)``.

    Prefers the first snapshot strictly after
    ``incident_dt + ttd + grace_period``. When no such snapshot exists (the
    target lands past the last collected snapshot — common for high-TTD
    bundles like ``ttd=480m``), falls back to the latest available snapshot
    as long as it still postdates ``incident_dt`` so the snapshot's
    retention window covers the incident. Only when even the last snapshot
    predates the incident do we skip the spec.
    """
    if ctx.code_only:
        return None, False
    target = incident_dt + timedelta(minutes=ttd + ctx.grace_period)
    idx = bisect.bisect_right(ctx.snapshot_times, target)
    if idx < len(ctx.snapshot_times):
        snap_dt = ctx.snapshot_times[idx]
    else:
        snap_dt = ctx.snapshot_times[-1] if ctx.snapshot_times else None
        if snap_dt is None or snap_dt <= incident_dt:
            logger.warning(
                f"No snapshot postdating incident for question {qid} with TTD "
                f"{ttd}m. Skipping."
            )
            return None, False
        logger.info(
            f"Question {qid}: target {target.isoformat()} (TTD {ttd}m) exceeds "
            f"last snapshot {snap_dt.isoformat()}; falling back to last snapshot."
        )
    return ctx.snapshots[snap_dt], snap_dt <= incident_dt


def _emit(
    ctx: _SpecBuildContext,
    *,
    spec_kwargs: dict[str, Any],
    question: dict,
    snapshot_name: str | None,
    no_incident: bool,
) -> None:
    """Build a TaskSpec from spec_kwargs + question/event linkage and write it to disk."""
    event = question["event"]
    flag = None if no_incident else event.get("flag")
    flag_desc = (
        "" if no_incident else ctx.flag_descriptions.get(event.get("flag", ""), "")
    )

    spec = TaskSpec(
        **spec_kwargs,
        snapshot_name=snapshot_name,
        no_incident=no_incident,
        event_id=event["id"],
        qid=question["id"],
        granularity=question.get("granularity"),
        user_facing_issue=question["question"],
        flag=flag,
        flag_description=flag_desc,
    )

    if spec.task_id in ctx.written_task_ids:
        raise ValueError(
            f"duplicate task_id {spec.task_id!r} (section={spec.section}); "
            f"this means two bundles produced overlapping (TTD, offset/category, "
            f"phrasing) tuples — fix the YAML config to make bundles disjoint."
        )
    ctx.written_task_ids.add(spec.task_id)
    spec.write_json(ctx.specs_dir / f"{spec.task_id}.json")
    logger.info(f"Wrote spec {spec.task_id}")


# ─────────────────────────── Section iterators ───────────────────────────
def iter_exact(
    question: dict,
    incident_dt: datetime,
    cfg: ExactSection,
    ctx: _SpecBuildContext,
) -> None:
    """Generate exact-time tasks for one question (one or more bundles)."""
    qid = question["id"]
    earliest = ctx.snapshot_times[0] if ctx.snapshot_times else None
    for bundle in cfg.bundles:
        for ttd in bundle.ttd_minutes:
            snapshot_name, no_incident = _pick_snapshot(incident_dt, ttd, ctx, qid)
            if snapshot_name is None and not ctx.code_only:
                continue
            current_dt = incident_dt + timedelta(minutes=ttd)
            for offset in bundle.offset_minutes:
                if not (abs(offset) < ttd or (ttd == 0 and offset == 0)):
                    continue
                reported_dt = incident_dt + timedelta(minutes=offset)
                if earliest is not None and reported_dt < earliest:
                    continue
                for phrasing in bundle.phrasings:
                    styled, _ = format_reported_time(
                        section="exact",
                        phrasing=phrasing,
                        incident_dt=incident_dt,
                        current_dt=current_dt,
                        offset_minutes=offset,
                    )
                    _emit(
                        ctx,
                        spec_kwargs={
                            "task_id": f"{slug(qid)}_ttd{ttd}m_{phrasing}_off{offset:+d}m",
                            "section": "exact",
                            "phrasing": phrasing,
                            "reported_styled": styled,
                            "ttd_minutes": ttd,
                            "incident_dt": incident_dt,
                            "current_dt": current_dt,
                            "offset_minutes": offset,
                            "reported_dt": reported_dt,
                            "reported_correctly_specified": abs(offset) <= 10,
                        },
                        question=question,
                        snapshot_name=snapshot_name,
                        no_incident=no_incident,
                    )


def iter_exact_range(
    question: dict,
    incident_dt: datetime,
    cfg: ExactRangeSection,
    ctx: _SpecBuildContext,
) -> None:
    """Generate exact-range tasks for one question (one or more bundles)."""
    qid = question["id"]
    earliest = ctx.snapshot_times[0] if ctx.snapshot_times else None
    for bundle in cfg.bundles:
        for ttd in bundle.ttd_minutes:
            snapshot_name, no_incident = _pick_snapshot(incident_dt, ttd, ctx, qid)
            if snapshot_name is None and not ctx.code_only:
                continue
            current_dt = incident_dt + timedelta(minutes=ttd)
            for window_cfg in bundle.configs:
                window = window_cfg.range_window
                for offset in window_cfg.offset_minutes:
                    if not (abs(offset) < ttd or (ttd == 0 and offset == 0)):
                        continue
                    if not (offset + window <= ttd):
                        continue
                    reported_dt = incident_dt + timedelta(minutes=offset)
                    if (
                        earliest is not None
                        and reported_dt + timedelta(minutes=window) <= earliest
                    ):
                        continue
                    styled, interval = format_reported_time(
                        section="exact_range",
                        incident_dt=incident_dt,
                        current_dt=current_dt,
                        offset_minutes=offset,
                        range_window=window,
                    )
                    _emit(
                        ctx,
                        spec_kwargs={
                            "task_id": f"{slug(qid)}_ttd{ttd}m_range{window}m_off{offset:+d}m",
                            "section": "exact_range",
                            "phrasing": "between",
                            "reported_styled": styled,
                            "ttd_minutes": ttd,
                            "incident_dt": incident_dt,
                            "current_dt": current_dt,
                            "offset_minutes": offset,
                            "reported_dt": reported_dt,
                            "range_window": window,
                            "reported_period": interval,
                            "reported_correctly_specified": abs(offset) <= window,
                        },
                        question=question,
                        snapshot_name=snapshot_name,
                        no_incident=no_incident,
                    )


def iter_broad_period(
    question: dict,
    incident_dt: datetime,
    cfg: BroadPeriodSection,
    ctx: _SpecBuildContext,
) -> None:
    """Generate broad time-of-day tasks for one question (one or more bundles)."""
    qid = question["id"]
    earliest = ctx.snapshot_times[0] if ctx.snapshot_times else None
    for bundle in cfg.bundles:
        for ttd in bundle.ttd_minutes:
            snapshot_name, no_incident = _pick_snapshot(incident_dt, ttd, ctx, qid)
            if snapshot_name is None and not ctx.code_only:
                continue
            current_dt = incident_dt + timedelta(minutes=ttd)
            for category in bundle.categories:
                styled, interval = format_reported_time(
                    section="broad_range_time_of_day",
                    incident_dt=incident_dt,
                    current_dt=current_dt,
                    broad_category=category,
                )
                assert interval is not None
                if interval[0] > current_dt:
                    continue
                if earliest is not None and interval[1] <= earliest:
                    continue
                correctly_specified = interval[0] <= incident_dt < interval[1]
                _emit(
                    ctx,
                    spec_kwargs={
                        "task_id": f"{slug(qid)}_ttd{ttd}m_period_{category}",
                        "section": "broad_range_time_of_day",
                        "phrasing": "sometime-period",
                        "reported_styled": styled,
                        "ttd_minutes": ttd,
                        "incident_dt": incident_dt,
                        "current_dt": current_dt,
                        "broad_category": category,
                        "reported_period": interval,
                        "reported_correctly_specified": correctly_specified,
                    },
                    question=question,
                    snapshot_name=snapshot_name,
                    no_incident=no_incident,
                )


def iter_broad_day(
    question: dict,
    incident_dt: datetime,
    cfg: BroadDaySection,
    ctx: _SpecBuildContext,
) -> None:
    """Generate broad day tasks for one question (one or more bundles).

    ``universal`` questions are excluded from this section: at day-level
    granularity ("sometime today / yesterday") with a flag-agnostic prompt,
    almost every active flag in the day is plausible — the resulting answer
    sets are too broad to be useful.
    """
    if question.get("granularity") == "universal":
        return
    qid = question["id"]
    earliest = ctx.snapshot_times[0] if ctx.snapshot_times else None
    for bundle in cfg.bundles:
        for ttd in bundle.ttd_minutes:
            snapshot_name, no_incident = _pick_snapshot(incident_dt, ttd, ctx, qid)
            if snapshot_name is None and not ctx.code_only:
                continue
            current_dt = incident_dt + timedelta(minutes=ttd)
            for category in bundle.categories:
                styled, interval = format_reported_time(
                    section="broad_range_day",
                    incident_dt=incident_dt,
                    current_dt=current_dt,
                    broad_category=category,
                )
                assert interval is not None
                if earliest is not None and interval[1] <= earliest:
                    continue
                correctly_specified = interval[0] <= incident_dt < interval[1]
                _emit(
                    ctx,
                    spec_kwargs={
                        "task_id": f"{slug(qid)}_ttd{ttd}m_day_{category}",
                        "section": "broad_range_day",
                        "phrasing": "sometime-day",
                        "reported_styled": styled,
                        "ttd_minutes": ttd,
                        "incident_dt": incident_dt,
                        "current_dt": current_dt,
                        "broad_category": category,
                        "reported_period": interval,
                        "reported_correctly_specified": correctly_specified,
                    },
                    question=question,
                    snapshot_name=snapshot_name,
                    no_incident=no_incident,
                )


def _anchor_at_hour(reference_dt: datetime, hour_local: int) -> datetime:
    """Return ``hour_local:00`` LOCAL_TZ on ``reference_dt``'s LOCAL_TZ date."""
    local = reference_dt.astimezone(LOCAL_TZ)
    return local.replace(hour=hour_local, minute=0, second=0, microsecond=0)


def _incident_end_for(
    intervals: dict[str, list[tuple[datetime, datetime]]],
    flag: str | None,
    incident_dt: datetime,
) -> datetime | None:
    """Return the off-event time of the (on, off) interval for ``flag``
    containing ``incident_dt``.

    Returns ``None`` if the flag has no off event for that interval (i.e.,
    ``compute_flag_intervals`` left it open with the ``datetime.max`` sentinel
    — year 9999).
    """
    if not flag:
        return None
    for on, off in intervals.get(flag, []):
        if on <= incident_dt < off:
            return None if off.year >= 9999 else off
    return None


def _emit_control_if_quiet(
    ctx: _SpecBuildContext,
    question: dict,
    qid: str,
    incident_dt: datetime,
    current_dt: datetime,
    *,
    offset_minutes: int,
    task_id_suffix: str,
    phrasings: list[str],
    incident_end_dt: datetime | None,
) -> None:
    """Validate ``current_dt`` is in a quiet window, pick a snapshot, emit.

    ``incident_end_dt is None`` ⇒ pre task: snapshot must predate ``incident_dt``.
    Otherwise ⇒ post task: snapshot must postdate ``incident_end_dt`` so the
    captured telemetry reflects the resolved state.
    """
    earliest = ctx.snapshot_times[0] if ctx.snapshot_times else None
    if earliest is not None and current_dt < earliest:
        logger.info(
            f"Skipping {task_id_suffix} qid={qid}: current_dt="
            f"{current_dt.isoformat()} predates earliest snapshot {earliest.isoformat()}."
        )
        return
    active = active_flag_at(ctx.flag_intervals, current_dt)
    if active is not None:
        logger.info(
            f"Skipping {task_id_suffix} qid={qid}: flag {active!r} active at "
            f"current_dt={current_dt.isoformat()}"
        )
        return

    target = current_dt + timedelta(minutes=ctx.grace_period)
    idx = bisect.bisect_right(ctx.snapshot_times, target)
    if ctx.code_only:
        snapshot_name: str | None = None
    else:
        if idx >= len(ctx.snapshot_times):
            logger.warning(f"No snapshot for {task_id_suffix} qid={qid}. Skipping.")
            return
        snap_dt = ctx.snapshot_times[idx]
        if incident_end_dt is None:
            if snap_dt >= incident_dt:
                logger.warning(
                    f"{task_id_suffix} qid={qid}: snap_dt={snap_dt} >= "
                    f"incident_dt={incident_dt}; skipping (would not be no_incident)."
                )
                return
        else:
            if snap_dt <= incident_end_dt:
                logger.warning(
                    f"{task_id_suffix} qid={qid}: snap_dt={snap_dt} <= "
                    f"incident_end={incident_end_dt}; skipping (still in active window)."
                )
                return
        snapshot_name = ctx.snapshots[snap_dt]

    # Widest quiet window around ``current_dt`` across the whole schedule —
    # bounded below by the latest off-event at or before ``current_dt`` and
    # above by the earliest on-event strictly after. ``None`` on either side
    # means open-ended (no neighboring incident on that side in the schedule).
    # The ``forever`` sentinel returned by ``compute_flag_intervals`` for
    # never-closed intervals is filtered out so it doesn't poison the bound.
    forever_year = 9999
    qstart = max(
        (
            off
            for ivs in ctx.flag_intervals.values()
            for (_, off) in ivs
            if off.year < forever_year and off <= current_dt
        ),
        default=None,
    )
    qend = min(
        (
            on
            for ivs in ctx.flag_intervals.values()
            for (on, _) in ivs
            if on > current_dt
        ),
        default=None,
    )

    for phrasing in phrasings:
        verb = "at" if phrasing == "at" else "around"
        styled = (
            f"{verb} {_fmt_time(current_dt)} "
            f"{format_day(current_dt, current_dt)} {LOCAL_TZ_LABEL}"
        )
        _emit(
            ctx,
            spec_kwargs={
                "task_id": f"{slug(qid)}_{task_id_suffix}_{phrasing}",
                "section": "control",
                "phrasing": phrasing,
                "reported_styled": styled,
                "ttd_minutes": 0,
                "incident_dt": incident_dt,
                "current_dt": current_dt,
                "offset_minutes": offset_minutes,
                "reported_dt": current_dt,
                "reported_correctly_specified": False,
                "quiet_window_start": qstart,
                "quiet_window_end": qend,
            },
            question=question,
            snapshot_name=snapshot_name,
            no_incident=True,
        )


def iter_control(
    question: dict,
    incident_dt: datetime,
    cfg: ControlSection,
    ctx: _SpecBuildContext,
) -> None:
    """Generate control (no-actual-incident) tasks for one question.

    Two flavors:

    * **Pre** controls — ``current_dt = min(incident_dt - pre_minutes,
      pre_anchor_hour_local on incident_dt's local day)``. The clamp pushes
      the control to morning when the incident itself fires late in the day.
    * **Post** controls — ``current_dt = max(incident_end_dt + post_minutes,
      post_anchor_hour_local on incident_dt's local day)``. The clamp pushes
      the control to evening when the incident resolves early.

    Both validate that no flag is active at ``current_dt`` (via
    ``active_flag_at``) and that the picked snapshot is on the correct side
    of the incident window. Post controls are skipped for flags whose
    schedule has no off event.
    """
    qid = question["id"]
    flag = question["event"].get("flag")

    # ── Pre tasks ──
    for pre in cfg.pre_incident_minutes:
        desired = incident_dt - timedelta(minutes=pre)
        anchor = _anchor_at_hour(incident_dt, cfg.pre_anchor_hour_local)
        current_dt = min(desired, anchor)
        offset = int((current_dt - incident_dt).total_seconds() // 60)
        _emit_control_if_quiet(
            ctx,
            question,
            qid,
            incident_dt,
            current_dt,
            offset_minutes=offset,
            task_id_suffix=f"control_pre{pre}m",
            phrasings=cfg.phrasings,
            incident_end_dt=None,
        )

    # ── Post tasks ──
    if not cfg.post_incident_minutes:
        return

    incident_end_dt = _incident_end_for(ctx.flag_intervals, flag, incident_dt)
    if incident_end_dt is None:
        logger.info(
            f"Skipping post controls for qid={qid}: flag {flag!r} has no off "
            f"event in the schedule."
        )
        return

    for post in cfg.post_incident_minutes:
        desired = incident_end_dt + timedelta(minutes=post)
        # Post anchor sits on the incident-START day (per user decision); the
        # max() correctly bumps current_dt past incident_end + post for
        # cross-midnight incidents.
        anchor = _anchor_at_hour(incident_dt, cfg.post_anchor_hour_local)
        current_dt = max(desired, anchor)
        offset = int((current_dt - incident_dt).total_seconds() // 60)
        _emit_control_if_quiet(
            ctx,
            question,
            qid,
            incident_dt,
            current_dt,
            offset_minutes=offset,
            task_id_suffix=f"control_post{post}m",
            phrasings=cfg.phrasings,
            incident_end_dt=incident_end_dt,
        )


# ───────────────────────────── Main ─────────────────────────────
def main() -> None:
    """Generate per-task TaskSpec JSONs under output_dir/task_specs/."""
    parser = get_base_parser()
    parser.description = "Generate per-task TaskSpec JSONs (v2 framing) for Harbor."
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/harbor_tasks_v2.yaml"),
        help="Path to v2 YAML config (default: configs/harbor_tasks_v2.yaml).",
    )
    parser.add_argument(
        "--qid",
        action="append",
        default=None,
        help="Only build specs for specific qid(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--grace-period",
        type=int,
        default=5,
        help="Minutes of data to include after the start of investigation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output_dir/task_specs if it already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help=(
            "Random seed used for sampling one question per granularity in "
            "load_questions. Default: 1 (reproducible)."
        ),
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Build specs without telemetry-snapshot bindings (snapshot_name=null).",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    config = load_yaml_config(args.config)
    logger.info(f"Loaded v2 config from {args.config}; schedule={config.schedule}")
    if args.schedule is not None and Path(args.schedule) != config.schedule:
        logger.warning(
            f"Ignoring --schedule={args.schedule} (v2 reads schedule from "
            f"the YAML config: {config.schedule})."
        )

    output_dir: Path = args.output_dir
    specs_dir = output_dir / "task_specs"
    if specs_dir.exists():
        if args.force:
            logger.info(f"Removing existing specs dir at {specs_dir}...")
            shutil.rmtree(specs_dir)
        elif any(specs_dir.iterdir()):
            raise FileExistsError(
                f"{specs_dir} already exists. Re-run with --force to rebuild specs."
            )
    specs_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args, config.schedule)
    if not questions:
        return

    # Augment with one universal question per event.
    seen: dict[str, tuple[dict, str]] = {}
    for q in questions:
        eid = q["event"]["id"]
        if eid not in seen:
            seen[eid] = (q["event"], q["answer"])
    universal_rng = random.Random(args.seed)
    for eid, (event, answer) in seen.items():
        idx = universal_rng.randrange(len(UNIVERSAL_QUESTIONS))
        questions.append(
            {
                "id": f"{eid}-univ{idx:02d}-universal",
                "question": UNIVERSAL_QUESTIONS[idx],
                "granularity": "universal",
                "event": event,
                "answer": answer,
            }
        )

    event_timestamps = load_event_timestamps(args.data_dir)
    flag_intervals = compute_flag_intervals(event_timestamps)
    logger.info(f"Computed active-flag intervals for {len(flag_intervals)} flag(s)")

    snapshots: dict[datetime, str] = {}
    snapshot_times: list[datetime] = []
    if not args.code_only:
        snapshots, snapshot_times = load_snapshots(args.data_dir)
        if not snapshots:
            logger.error(
                f"No snapshots found in {args.data_dir / 'prometheus' / 'snapshots'}"
            )
            return

    flag_descriptions = load_flag_descriptions()

    ctx = _SpecBuildContext(
        specs_dir=specs_dir,
        code_only=args.code_only,
        grace_period=args.grace_period,
        snapshots=snapshots,
        snapshot_times=snapshot_times,
        flag_descriptions=flag_descriptions,
        flag_intervals=flag_intervals,
        written_task_ids=set(),
    )

    for question in questions:
        event_id = question["event"]["id"]
        if event_id not in event_timestamps:
            logger.warning(
                f"Event {event_id} has not been recorded in the events directory. "
                f"Skipping question {question['id']}."
            )
            continue
        incident_dt = datetime.fromisoformat(
            event_timestamps[event_id]["wall_clock_utc"]
        )

        if config.exact.enabled:
            iter_exact(question, incident_dt, config.exact, ctx)
        if config.exact_range.enabled:
            iter_exact_range(question, incident_dt, config.exact_range, ctx)
        if config.broad_range_time_of_day.enabled:
            iter_broad_period(
                question, incident_dt, config.broad_range_time_of_day, ctx
            )
        if config.broad_range_day.enabled:
            iter_broad_day(question, incident_dt, config.broad_range_day, ctx)
        if config.control.enabled:
            iter_control(question, incident_dt, config.control, ctx)

    logger.info(f"Wrote {len(ctx.written_task_ids)} task spec(s) to {specs_dir}")


if __name__ == "__main__":
    load_dotenv()
    main()
