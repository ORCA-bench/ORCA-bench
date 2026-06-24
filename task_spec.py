r"""Shared types and helpers for the v2 task pipeline.

The v2 task pipeline (``generate_task_specs.py`` -> ``generate_answers.py`` ->
``build_harbor_tasks.py``) uses ``output_dir/`` as its on-disk interface:

    output_dir/
    ├── questions/        (input — produced by upstream question-generation scripts)
    ├── json/             (input — rubrics, e.g. out-0427/json/)
    ├── task_specs/       ← generate_task_specs.py
    ├── answers/          ← generate_answers.py
    └── harbor/tasks/     ← build_harbor_tasks.py

This module hosts the dataclass and helpers shared by all three scripts so
none of them imports from another. Lifted (with small additions) from the
pre-split ``prepare_harbor_tasks_v2.py``.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from slugify import slugify

logger = logging.getLogger(__name__)

# ──────────────────────────── Period definitions ────────────────────────────
LOCAL_TZ = ZoneInfo("America/New_York")
LOCAL_TZ_LABEL = "ET"

PERIODS: list[tuple[str, int, int]] = [
    ("late night", 0, 5),
    ("early morning", 5, 8),
    ("morning", 8, 12),
    ("afternoon", 12, 17),
    ("evening", 17, 21),
    ("night", 21, 24),
]

EXACT_PHRASINGS = ("at", "around")
PERIOD_CATEGORIES = ("same-period", "previous-period", "next-period")
DAY_CATEGORIES = ("same-day", "previous-day")

# Flag-agnostic universal questions appended to every incident's task set.
UNIVERSAL_QUESTIONS: list[str] = [
    "users are reporting site issues",
]


def _period_index_for_hour(hour: int) -> int:
    for i, (_, start_h, end_h) in enumerate(PERIODS):
        if start_h <= hour < end_h:
            return i
    raise ValueError(f"hour {hour} not in any period")


def _period_bucket(
    day_start_local: datetime, idx: int
) -> tuple[str, datetime, datetime]:
    """Return (name, start_dt, end_dt) for bucket ``idx`` on ``day_start_local``."""
    name, start_h, end_h = PERIODS[idx]
    period_start = day_start_local.replace(hour=start_h)
    period_end = (
        day_start_local + timedelta(days=1)
        if end_h == 24
        else day_start_local.replace(hour=end_h)
    )
    return name, period_start, period_end


def period_for(dt: datetime) -> tuple[str, datetime, datetime]:
    """Return the bucket containing ``dt``'s LOCAL_TZ hour, on its LOCAL_TZ date."""
    local = dt.astimezone(LOCAL_TZ)
    day_start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return _period_bucket(day_start, _period_index_for_hour(local.hour))


def period_at_offset(dt: datetime, n: int) -> tuple[str, datetime, datetime]:
    """Return the bucket ``n`` positions away from ``dt``'s LOCAL_TZ bucket."""
    local = dt.astimezone(LOCAL_TZ)
    base_idx = _period_index_for_hour(local.hour)
    day_offset, new_idx = divmod(base_idx + n, len(PERIODS))
    base_day = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return _period_bucket(base_day + timedelta(days=day_offset), new_idx)


def start_of_local_day(dt: datetime) -> datetime:
    """Return midnight in LOCAL_TZ on ``dt``'s local date."""
    local = dt.astimezone(LOCAL_TZ)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def end_of_local_day(dt: datetime) -> datetime:
    """Return midnight in LOCAL_TZ at the start of the day *after* ``dt``'s local date.

    Exclusive upper bound of ``dt``'s local day: an event at exactly this instant
    belongs to the next day. Pairs with ``start_of_local_day`` to form the
    half-open day window ``[start_of_local_day(dt), end_of_local_day(dt))``.
    """
    return start_of_local_day(dt) + timedelta(days=1)


def format_day(dt: datetime, current_dt: datetime) -> str:
    """Return ``"today"`` / ``"yesterday"`` / ``"on Mon DD, YYYY"`` relative to current_dt."""
    local_dt = dt.astimezone(LOCAL_TZ)
    local_current = current_dt.astimezone(LOCAL_TZ)
    if local_dt.date() == local_current.date():
        return "today"
    if local_dt.date() == (local_current - timedelta(days=1)).date():
        return "yesterday"
    return local_dt.strftime("on %b %d, %Y")


def _fmt_time(dt: datetime) -> str:
    """Format datetime as ``H:MM AM/PM`` (no leading zero on hour) in LOCAL_TZ."""
    local = dt.astimezone(LOCAL_TZ)
    h = local.hour % 12 or 12
    return f"{h}:{local.minute:02d} {local.strftime('%p')}"


# ───────────────────────── Reported-time formatting ─────────────────────────
def format_reported_time(
    *,
    section: str,
    phrasing: str | None = None,
    incident_dt: datetime,
    current_dt: datetime,
    offset_minutes: int | None = None,
    range_window: int | None = None,
    broad_category: str | None = None,
) -> tuple[str, tuple[datetime, datetime] | None]:
    """Render the reported-time phrase for the prompt.

    Returns ``(styled_string, period_interval_or_None)``. The interval is the
    half-open ``[start, end)`` window the agent's reported phrasing implies.
    """
    if section == "exact":
        reported_dt = incident_dt + timedelta(minutes=offset_minutes or 0)
        verb = "at" if phrasing == "at" else "around"
        styled = (
            f"{verb} {_fmt_time(reported_dt)} "
            f"{format_day(reported_dt, current_dt)} {LOCAL_TZ_LABEL}"
        )
        return styled, None

    if section == "exact_range":
        reported_dt = incident_dt + timedelta(minutes=offset_minutes or 0)
        lo = reported_dt - timedelta(minutes=range_window or 0)
        hi = reported_dt + timedelta(minutes=range_window or 0)
        styled = (
            f"between {_fmt_time(lo)} and {_fmt_time(hi)} "
            f"{format_day(reported_dt, current_dt)} {LOCAL_TZ_LABEL}"
        )
        return styled, (lo, hi)

    if section == "broad_range_time_of_day":
        if broad_category == "same-period":
            name, start_dt, end_dt = period_for(incident_dt)
        elif broad_category == "previous-period":
            name, start_dt, end_dt = period_at_offset(incident_dt, -1)
        elif broad_category == "next-period":
            name, start_dt, end_dt = period_at_offset(incident_dt, +1)
        else:
            raise ValueError(
                f"unknown broad_range_time_of_day category: {broad_category}"
            )
        styled = f"sometime in the {name} {format_day(start_dt, current_dt)}"
        return styled, (start_dt, end_dt)

    if section == "broad_range_day":
        incident_local = incident_dt.astimezone(LOCAL_TZ)
        incident_day_start = incident_local.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if broad_category == "same-day":
            bucket_start = incident_day_start
        elif broad_category == "previous-day":
            bucket_start = incident_day_start - timedelta(days=1)
        else:
            raise ValueError(f"unknown broad_range_day category: {broad_category}")
        bucket_end = bucket_start + timedelta(days=1)
        styled = f"sometime {format_day(bucket_start, current_dt)}"
        return styled, (bucket_start, bucket_end)

    raise ValueError(f"unknown section: {section}")


# ───────────────────────────── TaskSpec ─────────────────────────────
@dataclass
class TaskSpec:
    """Per-task inputs that vary across the four sections, plus question/event linkage.

    Serialized to ``output_dir/task_specs/<task_id>.json`` by
    ``generate_task_specs.py`` and consumed by ``generate_answers.py`` and
    ``build_harbor_tasks.py``.
    """

    task_id: str
    section: str
    phrasing: str
    reported_styled: str
    ttd_minutes: int
    incident_dt: datetime
    current_dt: datetime
    # Section-specific time-axis fields:
    offset_minutes: int | None = None
    reported_dt: datetime | None = None
    range_window: int | None = None
    broad_category: str | None = None
    reported_period: tuple[datetime, datetime] | None = None
    reported_correctly_specified: bool = False
    # Snapshot binding (None for code-only tasks):
    snapshot_name: str | None = None
    no_incident: bool = False
    # Control-task quiet window: the widest contiguous interval around
    # ``current_dt`` in which no flag is active across the entire schedule.
    # Used by the LLM judge to decide whether the agent's report places any
    # incident time inside the window (false positive). Set only for control
    # tasks; ``None`` for incident tasks.
    quiet_window_start: datetime | None = None
    quiet_window_end: datetime | None = None
    # Question/event linkage (used by generate_answers + build_harbor_tasks):
    event_id: str | None = None
    qid: str | None = None
    granularity: str | None = None
    user_facing_issue: str | None = None
    flag: str | None = None
    flag_description: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (datetimes -> ISO 8601 strings)."""
        d = asdict(self)
        d["incident_dt"] = self.incident_dt.isoformat()
        d["current_dt"] = self.current_dt.isoformat()
        d["reported_dt"] = (
            self.reported_dt.isoformat() if self.reported_dt is not None else None
        )
        d["quiet_window_start"] = (
            self.quiet_window_start.isoformat()
            if self.quiet_window_start is not None
            else None
        )
        d["quiet_window_end"] = (
            self.quiet_window_end.isoformat()
            if self.quiet_window_end is not None
            else None
        )
        # Flatten reported_period from tuple[datetime, datetime] -> [str, str].
        if self.reported_period is not None:
            d["reported_period"] = [
                self.reported_period[0].isoformat(),
                self.reported_period[1].isoformat(),
            ]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        """Reverse of ``to_dict``."""
        period = d.get("reported_period")
        return cls(
            task_id=d["task_id"],
            section=d["section"],
            phrasing=d["phrasing"],
            reported_styled=d["reported_styled"],
            ttd_minutes=d["ttd_minutes"],
            incident_dt=datetime.fromisoformat(d["incident_dt"]),
            current_dt=datetime.fromisoformat(d["current_dt"]),
            offset_minutes=d.get("offset_minutes"),
            reported_dt=(
                datetime.fromisoformat(d["reported_dt"])
                if d.get("reported_dt") is not None
                else None
            ),
            range_window=d.get("range_window"),
            broad_category=d.get("broad_category"),
            reported_period=(
                (
                    datetime.fromisoformat(period[0]),
                    datetime.fromisoformat(period[1]),
                )
                if period is not None
                else None
            ),
            reported_correctly_specified=d.get("reported_correctly_specified", False),
            snapshot_name=d.get("snapshot_name"),
            no_incident=d.get("no_incident", False),
            quiet_window_start=(
                datetime.fromisoformat(d["quiet_window_start"])
                if d.get("quiet_window_start") is not None
                else None
            ),
            quiet_window_end=(
                datetime.fromisoformat(d["quiet_window_end"])
                if d.get("quiet_window_end") is not None
                else None
            ),
            event_id=d.get("event_id"),
            qid=d.get("qid"),
            granularity=d.get("granularity"),
            user_facing_issue=d.get("user_facing_issue"),
            flag=d.get("flag"),
            flag_description=d.get("flag_description", ""),
        )

    def write_json(self, path: Path) -> None:
        """Serialize to ``path`` (creates parent dirs)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def read_json(cls, path: Path) -> "TaskSpec":
        """Read a TaskSpec written by ``write_json``."""
        return cls.from_dict(json.loads(path.read_text()))


# ───────────────────────────── Event helpers ─────────────────────────────
def load_event_timestamps(data_dir: Path) -> dict[str, dict]:
    """Load saved event timestamps from ``data_dir/events/`` (id → metadata)."""
    events_dir = data_dir / "events"
    timestamps: dict[str, dict] = {}
    if events_dir.is_dir():
        for event_file in events_dir.glob("*.json"):
            with event_file.open() as f:
                meta = json.load(f)
            timestamps[meta["id"]] = meta
    logger.info(f"Loaded {len(timestamps)} event timestamp(s) from {events_dir}")
    return timestamps


def compute_flag_intervals(
    event_timestamps: dict[str, dict],
) -> dict[str, list[tuple[datetime, datetime]]]:
    """Pair each flag's non-``off`` event with its next ``off`` (or end of time).

    Multiple non-``off`` events for the same flag are merged into a single
    ``(on, off)`` interval; the flag is "on" continuously from the first
    non-``off`` value until the matching ``off`` event. If a flag is never
    turned off in the schedule, its closing time is ``datetime.max``.
    """
    sorted_events = sorted(
        event_timestamps.values(),
        key=lambda e: datetime.fromisoformat(e["wall_clock_utc"]),
    )
    by_flag: dict[str, list[dict]] = {}
    for ev in sorted_events:
        flag = ev.get("flag")
        if not flag:
            continue
        by_flag.setdefault(flag, []).append(ev)

    intervals: dict[str, list[tuple[datetime, datetime]]] = {}
    forever = datetime.max.replace(tzinfo=ZoneInfo("UTC"))
    for flag, events in by_flag.items():
        flag_ivs: list[tuple[datetime, datetime]] = []
        on_dt: datetime | None = None
        for ev in events:
            t = datetime.fromisoformat(ev["wall_clock_utc"])
            value = ev.get("value", "")
            if value == "off":
                if on_dt is not None:
                    flag_ivs.append((on_dt, t))
                    on_dt = None
            else:
                if on_dt is None:
                    on_dt = t
        if on_dt is not None:
            flag_ivs.append((on_dt, forever))
        intervals[flag] = flag_ivs
    return intervals


def active_flag_at(
    intervals: dict[str, list[tuple[datetime, datetime]]],
    t: datetime,
) -> str | None:
    """Return the name of any flag active at ``t`` (half-open ``[on, off)``), else None."""
    for flag, ivs in intervals.items():
        for on, off in ivs:
            if on <= t < off:
                return flag
    return None


def candidate_events_in_window(
    event_timestamps: dict[str, dict],
    lo: datetime,
    hi: datetime,
) -> list[dict]:
    """Return events from ``event_timestamps`` that fired in ``[lo, hi)`` and
    introduced a problem (``value != "off"``).

    Each returned dict is the event metadata loaded from
    ``data_dir/events/<event_id>.json``: ``id, flag, value, wall_clock_utc,
    description, ...``.
    """
    out: list[dict] = []
    for ev in event_timestamps.values():
        if ev.get("value") == "off":
            continue
        ts = datetime.fromisoformat(ev["wall_clock_utc"])
        if lo <= ts < hi:
            out.append(ev)
    out.sort(key=lambda e: e["wall_clock_utc"])
    return out


# ───────────────────────────── Misc ─────────────────────────────
def slug(qid: str) -> str:
    """Slugify a question id for embedding in a task_id."""
    return slugify(qid)
