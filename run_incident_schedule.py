#!/usr/bin/env python3
"""Incident Schedule Executor for AI SRE Evaluation

This script executes a predefined incident schedule by toggling feature flags
at specified times. It supports both real-time execution and accelerated mode
for testing.

TIMING MODEL:
    Times in schedule files are RELATIVE to when the script starts, not wall-clock times.
    The --start-time flag sets the "simulation clock" starting point.

    For example, if your schedule has an event at "10:30":
      - With --start-time 00:00 (default): event fires 10h 30m after script starts
      - With --start-time 10:25: event fires 5 minutes after script starts
      - With --start-time 10:30: event fires immediately

    To run a schedule starting at a specific wall-clock time, set --start-time to
    match the first event's time (or slightly before).

Example usage:
    # Run schedule starting from simulation time 00:00 (events fire relative to now)
    python run_incident_schedule.py schedules/my_schedule.json

    # Start simulation at 10:25, so event at 10:30 fires in 5 real minutes
    python run_incident_schedule.py schedules/my_schedule.json --start-time 10:25

    # Accelerated mode (1 hour = 1 minute)
    python run_incident_schedule.py schedules/my_schedule.json --accelerate 60

    # Dry run (show what would happen without making changes)
    python run_incident_schedule.py schedules/my_schedule.json --dry-run

    # Run specific category only
    python run_incident_schedule.py schedules/my_schedule.json --category cascading_issues
"""

import json
import subprocess
import time
import logging
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from slugify import slugify
from utils import DEFAULT_FLAGD_PATH, FlagdController, get_base_parser, setup_logging

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_SCHEDULE_PATH = Path("incident_schedule.json")

# OpenSearch snapshot API settings
OPENSEARCH_URL = "http://localhost:9200"
OPENSEARCH_REPO_NAME = "scheduled_backups"
OPENSEARCH_REPO_PATH = "/usr/share/opensearch/snapshots"

# Prometheus snapshot API settings
PROMETHEUS_URL = "http://localhost:9090"
PROMETHEUS_SNAPSHOT_DIR = "prometheus/snapshots"


@dataclass
class ScheduleEvent:
    """Represents a single event in the incident schedule."""

    id: str
    category: str
    description: str
    day: int
    time: str
    action: str
    flag: str
    value: str
    evaluation_note: Optional[str] = None

    @property
    def time_minutes(self) -> int:
        """Convert time string to minutes since midnight."""
        hours, minutes = map(int, self.time.split(":"))
        return hours * 60 + minutes

    @property
    def absolute_minutes(self) -> int:
        """Get absolute minutes from start of day 1."""
        return (self.day - 1) * 24 * 60 + self.time_minutes


class IncidentScheduler:
    """Executes the incident schedule."""

    def __init__(
        self,
        schedule_path: Path,
        flagd_controller: FlagdController,
        data_dir: Path,
        accelerate: float = 1.0,
        category_filter: Optional[str] = None,
        snapshot_interval: int = 5,
        dry_run: bool = False,
    ):
        self.flagd = flagd_controller
        self.accelerate = accelerate
        self.category_filter = category_filter
        self.events = self._load_schedule(schedule_path)
        self.running = True
        self.start_time: Optional[datetime] = None
        self.start_offset_minutes: int = 0

        # Snapshotting
        self.data_dir = data_dir
        self.snapshot_interval = snapshot_interval
        self.dry_run = dry_run
        self._last_snapshot_minutes: float = 0.0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: object) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("\nReceived shutdown signal. Cleaning up...")
        self.running = False

    def _load_schedule(self, schedule_path: Path) -> list[ScheduleEvent]:
        """Load and parse the schedule file."""
        if not schedule_path.exists():
            raise FileNotFoundError(f"Schedule not found: {schedule_path}")

        with open(schedule_path, "r") as f:
            data = json.load(f)

        events = []
        for item in data["schedule"]:
            event = ScheduleEvent(
                id=item["id"],
                category=item["category"],
                description=item["description"],
                day=item["day"],
                time=item["time"],
                action=item["action"],
                flag=item["flag"],
                value=item["value"],
                evaluation_note=item.get("evaluation_note"),
            )
            if self.category_filter is None or event.category == self.category_filter:
                events.append(event)

        # Sort by absolute time
        events.sort(key=lambda e: e.absolute_minutes)
        logger.info(f"Loaded {len(events)} events from schedule")
        return events

    def _get_elapsed_minutes(self) -> float:
        """Get elapsed simulation minutes since start."""
        if self.start_time is None:
            return 0
        elapsed_real = (datetime.now() - self.start_time).total_seconds() / 60
        return self.start_offset_minutes + (elapsed_real * self.accelerate)

    def _sim_time_label(self, sim_minutes: float) -> str:
        """Convert simulation minutes to a snapshot label like 'day1_09h15m'."""
        day = int(sim_minutes // (24 * 60)) + 1
        remaining = int(sim_minutes) % (24 * 60)
        hours = remaining // 60
        minutes = remaining % 60
        return f"day{day}_{hours:02d}h{minutes:02d}m"

    def _register_opensearch_repo(self) -> None:
        """Register the OpenSearch snapshot repository (idempotent)."""
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-X",
                    "PUT",
                    f"{OPENSEARCH_URL}/_snapshot/{OPENSEARCH_REPO_NAME}",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    json.dumps(
                        {
                            "type": "fs",
                            "settings": {"location": OPENSEARCH_REPO_PATH},
                        }
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            resp = json.loads(result.stdout)
            if resp.get("acknowledged"):
                logger.info(
                    f"OpenSearch snapshot repo '{OPENSEARCH_REPO_NAME}' registered"
                )
            else:
                logger.warning(
                    f"OpenSearch repo registration response: {result.stdout.strip()}"
                )
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to register OpenSearch snapshot repo: {e}")

    def _take_opensearch_snapshot(self, label: str) -> None:
        """Take an OpenSearch snapshot using the native snapshot API."""
        snapshot_name = slugify(label)
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-X",
                    "PUT",
                    f"{OPENSEARCH_URL}/_snapshot/{OPENSEARCH_REPO_NAME}/{snapshot_name}?wait_for_completion=true",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    json.dumps({"indices": "*", "include_global_state": False}),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            resp = json.loads(result.stdout)
            status = resp.get("snapshot", {}).get("state", "UNKNOWN")
            if status == "SUCCESS":
                logger.info(
                    f"OpenSearch snapshot '{snapshot_name}' completed successfully"
                )
            else:
                logger.warning(
                    f"OpenSearch snapshot state: {status} — {result.stdout.strip()}"
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"OpenSearch snapshot '{snapshot_name}' timed out")
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"OpenSearch snapshot '{snapshot_name}' failed: {e}")

    def _take_prometheus_snapshot(self) -> str | None:
        """Take a Prometheus snapshot via the admin TSDB API.

        Returns the hash-based snapshot name (e.g. ``20260210T190036Z-...``)
        on success, or ``None`` on failure.  This name is used as the
        canonical snapshot identifier for all backends.
        """
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-X",
                    "POST",
                    f"{PROMETHEUS_URL}/api/v1/admin/tsdb/snapshot",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            resp = json.loads(result.stdout)
            if resp.get("status") != "success":
                logger.warning(f"Prometheus snapshot failed: {result.stdout.strip()}")
                return None

            snap_name = resp["data"]["name"]
            logger.info(f"Prometheus snapshot completed: {snap_name}")
            return snap_name

        except subprocess.TimeoutExpired:
            logger.warning("Prometheus snapshot timed out")
            return None
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Prometheus snapshot failed: {e}")
            return None

    def _take_snapshot(self) -> None:
        """Take a snapshot of each telemetry component.

        Prometheus is snapshotted first; its hash-based name (e.g.
        ``20260210T190036Z-324b21f5fffc57d1``) becomes the canonical
        snapshot identifier used for all backends:

            prometheus/snapshots/<hash>/          — Prometheus TSDB snapshot
            opensearch-repo-snapshots/  (<hash>)  — OpenSearch native snapshot
                                                    (includes Jaeger traces + OTel logs)
        """
        sim_minutes = self._get_elapsed_minutes()
        if self.dry_run:
            logger.info(f"[DRY RUN] Would snapshot at sim {sim_minutes:.0f}m")
            self._last_snapshot_minutes = sim_minutes
            return

        t0 = time.monotonic()

        # Prometheus first — its hash becomes the canonical snapshot name
        snap_name = self._take_prometheus_snapshot()
        if snap_name is None:
            logger.warning("Skipping remaining snapshots (Prometheus failed)")
            return

        # OpenSearch snapshot using the same name (captures Jaeger traces + OTel logs)
        self._take_opensearch_snapshot(snap_name)

        elapsed = time.monotonic() - t0
        logger.info(f"Snapshot saved: {snap_name} ({elapsed:.1f}s)")
        self._last_snapshot_minutes = sim_minutes

    def _maybe_snapshot(self) -> None:
        """Take a snapshot if enough simulation time has elapsed since the last one.

        Returns the snapshot name if a snapshot was taken, or None otherwise.
        """
        sim_minutes = self._get_elapsed_minutes()
        if sim_minutes - self._last_snapshot_minutes >= self.snapshot_interval:
            self._take_snapshot()

    def run(
        self,
        start_day: int = 1,
        start_time: str = "00:00",
        reset_first: bool = True,
    ) -> None:
        """Execute the incident schedule.

        Args:
            start_day: Day to start from (1 or 2)
            start_time: Time to start from (HH:MM format)
            reset_first: Whether to reset all flags before starting

        """
        # Calculate start offset
        start_hours, start_mins = map(int, start_time.split(":"))
        self.start_offset_minutes = (
            (start_day - 1) * 24 * 60 + start_hours * 60 + start_mins
        )

        # Filter events to only those after start time
        events_to_run = [
            e for e in self.events if e.absolute_minutes >= self.start_offset_minutes
        ]

        if not events_to_run:
            logger.warning("No events to run after the specified start time")
            return

        logger.info("Starting schedule execution:")
        logger.info(f"  Start: Day {start_day}, {start_time}")
        logger.info(f"  Events remaining: {len(events_to_run)}")
        logger.info(
            f"  Acceleration: {self.accelerate}x (1 real minute = {self.accelerate} sim minutes)"
        )
        if self.category_filter:
            logger.info(f"  Category filter: {self.category_filter}")

        if reset_first:
            self.flagd.reset_all_flags()
        # Register OpenSearch snapshot repository
        self._register_opensearch_repo()

        self.start_time = datetime.now()
        # Take initial snapshot
        self._take_snapshot()

        # Execute events
        prev_minutes = self.start_offset_minutes
        for i, event in enumerate(events_to_run):
            if not self.running:
                break

            # Sleep for the interval between this event and the previous one
            wait_sim_minutes = event.absolute_minutes - prev_minutes
            wait_real_seconds = (wait_sim_minutes / self.accelerate) * 60
            logger.info(
                f"\n[{i + 1}/{len(events_to_run)}] Waiting for Day {event.day} {event.time}..."
            )
            # Take snapshots every `self.snapshot_interval` minutes during the wait
            while wait_real_seconds > 0 and self.running:
                sleep_time = min(wait_real_seconds, 1.0)
                time.sleep(sleep_time)
                wait_real_seconds -= sleep_time
                self._maybe_snapshot()
            if not self.running:
                break

            prev_minutes = event.absolute_minutes

            # Execute event
            self._execute_event(event)
            self._maybe_snapshot()

        if self.running:
            logger.info("\nSchedule completed successfully!")
        else:
            logger.info("\nSchedule interrupted. Current flag states:")
            for flag, value in self.flagd.get_current_state().items():
                if value != "off":
                    logger.info(f"  {flag}: {value}")

    def _execute_event(self, event: ScheduleEvent) -> None:
        """Execute a single schedule event."""
        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"EVENT: {event.description}")
        logger.info(f"  ID: {event.id}")
        logger.info(f"  Category: {event.category}")
        logger.info(f"  Time: Day {event.day} @ {event.time}")
        if event.evaluation_note:
            logger.info(f"  Evaluation: {event.evaluation_note}")
        logger.info(f"{'=' * 60}")

        if event.action == "set":
            self.flagd.set_flag(event.flag, event.value)
        else:
            logger.warning(f"Unknown action: {event.action}")

        if not self.dry_run:
            event_data = {
                "id": event.id,
                "category": event.category,
                "description": event.description,
                "day": event.day,
                "time": event.time,
                "action": event.action,
                "flag": event.flag,
                "value": event.value,
                "wall_clock_utc": datetime.now(timezone.utc).isoformat(),
            }
            event_dir = self.data_dir / "events"
            event_dir.mkdir(parents=True, exist_ok=True)
            (event_dir / f"{event.id}.json").write_text(
                json.dumps(event_data, indent=2)
            )


def main() -> None:
    """Execute incident schedule for AI SRE evaluation."""
    parser = get_base_parser()
    parser.description = "Execute incident schedule for AI SRE evaluation"
    parser.add_argument(
        "--accelerate",
        "-a",
        type=float,
        default=1.0,
        help="Acceleration factor (e.g., 60 means 1 hour = 1 minute)",
    )
    parser.add_argument(
        "--start-day",
        type=int,
        default=1,
        choices=[1, 2],
        help="Day to start execution from",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default="00:00",
        help="Time to start execution from (HH:MM format). Events before this time will be skipped.",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        choices=[
            "warmup",
            "independent_flags",
            "conflicting_flags",
            "cascading_issues",
            "deployment_impact",
        ],
        help="Only run events from this category",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would happen without making changes",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Don't reset flags to 'off' before starting",
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Only reset all flags to 'off' and exit",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=5,
        help="Snapshot interval in simulation minutes (default: 5)",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)

    # Initialize controller
    flagd = FlagdController(DEFAULT_FLAGD_PATH, dry_run=args.dry_run)

    # Handle reset-only mode
    if args.reset_only:
        flagd.reset_all_flags()
        logger.info("All flags reset to 'off'")
        return

    # Initialize scheduler
    scheduler = IncidentScheduler(
        schedule_path=args.schedule,
        flagd_controller=flagd,
        accelerate=args.accelerate,
        category_filter=args.category,
        data_dir=args.data_dir,
        snapshot_interval=args.snapshot_interval,
        dry_run=args.dry_run,
    )

    # Run the schedule
    scheduler.run(
        start_day=args.start_day,
        start_time=args.start_time,
        reset_first=not args.no_reset,
    )


if __name__ == "__main__":
    main()
