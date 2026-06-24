#!/usr/bin/env python3
"""Find error traces across all services via the Jaeger API.

Reads event JSON files from ``data-dir/events/``, then queries Jaeger for
error traces within a time window around each incident, resolves each span's
service via the processID -> processes mapping, and outputs unique
(service, operationName) pairs with span events.

Examples::

    # Query error traces for all events in a data directory:
    uv run python find_traces.py -dd data-0216 -od out

    # With a custom time window (default 10 minutes each side):
    uv run python find_traces.py -dd data-0216 -od out --window 20

    # Asymmetric window (5 min before, 15 min after):
    uv run python find_traces.py -dd data-0216 -od out --left-window 5 --right-window 15

    # Process a single event only (matches the "id" field in event JSON):
    uv run python find_traces.py -dd data-0216 -od out --event-id deployment-2a
"""

import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from utils import get_base_parser, setup_logging

logger = logging.getLogger(__name__)

JAEGER_BASE = "http://localhost:8080/jaeger/ui/api"

# EventStream spans are normal ~10-minute long-poll reconnection cycles for
# feature flag streaming.  They appear in EVERY observation window regardless
# of incidents and are always background noise.
_NOISE_OPERATIONS = frozenset({"flagd.evaluation.v1.Service/EventStream"})


def _is_noise_span(span: dict) -> bool:
    """Return True if this span is deterministic background noise."""
    return span.get("operationName", "") in _NOISE_OPERATIONS


def get_services() -> list[str]:
    """Fetch all known service names from Jaeger."""
    url = f"{JAEGER_BASE}/services"
    logger.debug(f"GET {url}")
    resp = urllib.request.urlopen(url)
    data = json.loads(resp.read())
    services = data.get("data", [])
    logger.info(f"Discovered {len(services)} services")
    return services


def get_error_traces(service: str, start_us: int, end_us: int) -> list[dict]:
    """Fetch traces with error tags for a service within a time range."""
    encoded_svc = urllib.parse.quote(service)
    tags = urllib.parse.quote(json.dumps({"error": "true"}))
    url = (
        f"{JAEGER_BASE}/traces?service={encoded_svc}"
        f"&tags={tags}&start={start_us}&end={end_us}&limit=100"
    )
    logger.debug(f"GET {url}")
    resp = urllib.request.urlopen(url)
    data = json.loads(resp.read())
    traces = data.get("data", [])
    logger.info(f"  {service}: {len(traces)} error traces")
    return traces


def extract_error_operations(traces: list[dict]) -> list[dict]:
    """Extract unique (service, operationName) pairs with events from error spans."""
    pairs: dict[tuple[str, str], list[dict]] = {}

    for trace in traces:
        processes = trace.get("processes", {})
        for span in trace.get("spans", []):
            is_error = any(
                t.get("key") == "error" and t.get("value") is True
                for t in span.get("tags", [])
            )
            if not is_error:
                continue
            if _is_noise_span(span):
                continue

            pid = span.get("processID", "")
            svc = processes.get(pid, {}).get("serviceName", "unknown")
            op = span["operationName"]
            key = (svc, op)
            if key not in pairs:
                pairs[key] = []

            # Extract events from logs, skipping noisy gRPC message events
            for log_entry in span.get("logs", []):
                fields = {f["key"]: f["value"] for f in log_entry.get("fields", [])}
                if not fields:
                    continue
                if fields.get("event") == "message":
                    continue
                if fields not in pairs[key]:
                    pairs[key].append(fields)

    result = []
    for (svc, op), events in sorted(pairs.items()):
        result.append({"service": svc, "operationName": op, "events": events})
    return result


def main() -> None:
    """Query Jaeger for error traces across all incidents in a data directory."""
    parser = get_base_parser()
    parser.description = (
        "Find error traces across all services via the Jaeger API "
        "for all incidents in a data directory"
    )
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        default=10,
        help="Minutes before/after incident to query (default: 10)",
    )
    parser.add_argument(
        "--left-window",
        "-lw",
        type=int,
        default=None,
        help="Minutes before incident to query (overrides --window for left side)",
    )
    parser.add_argument(
        "--right-window",
        "-rw",
        type=int,
        default=None,
        help="Minutes after incident to query (overrides --window for right side)",
    )
    parser.add_argument(
        "--event-id",
        type=str,
        default=None,
        help="Process only the event whose 'id' field matches this value (e.g. 'deployment-2a')",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir

    events_dir = data_dir / "events"
    event_files = sorted(events_dir.glob("*.json"))
    if not event_files:
        logger.error(f"No event files found in {events_dir}")
        return

    left_minutes = args.left_window if args.left_window is not None else args.window
    right_minutes = args.right_window if args.right_window is not None else args.window

    # Hardcoded service list — Jaeger /services only reflects whatever the live
    # jaeger-main-jaeger-service-* indices currently contain, which can be
    # incomplete depending on the loaded snapshot window.
    services = [
        "accounting",
        "ad",
        "cart",
        "checkout",
        "currency",
        "email",
        "flagd",
        "flagd-ui",
        "fraud-detection",
        "frontend",
        "frontend-proxy",
        "image-provider",
        "load-generator",
        "payment",
        "product-catalog",
        "product-reviews",
        "quote",
        "recommendation",
        "shipping",
    ]

    for event_path in event_files:
        event = json.loads(event_path.read_text())
        if event.get("value") == "off":
            logger.info(f"Skipping resolve event {event['id']}")
            continue
        event_id = event["id"]
        if args.event_id is not None and event_id != args.event_id:
            logger.debug(f"Skipping event {event_id} (does not match --event-id)")
            continue
        flag = event["flag"]
        wall_clock_utc = event["wall_clock_utc"]

        logger.info(f"Processing event {event_id} (flag={flag})")

        incident_dt = datetime.fromisoformat(wall_clock_utc)
        start_dt = incident_dt - timedelta(minutes=left_minutes)
        end_dt = incident_dt + timedelta(minutes=right_minutes)
        start_us = int(start_dt.timestamp() * 1_000_000)
        end_us = int(end_dt.timestamp() * 1_000_000)
        logger.info(f"Querying traces for {event_id} window=[{start_dt} .. {end_dt}]")

        # Fetch error traces for all services
        all_traces: list[dict] = []
        for svc in services:
            traces = get_error_traces(svc, start_us, end_us)
            all_traces.extend(traces)

        # Deduplicate traces by traceID, keeping only traces that have at
        # least one non-noise error span.
        seen_ids: set[str] = set()
        unique_traces: list[dict] = []
        for trace in all_traces:
            tid = trace.get("traceID", "")
            if tid in seen_ids:
                continue
            seen_ids.add(tid)
            has_signal = any(
                not _is_noise_span(span)
                and any(
                    t.get("key") == "error" and t.get("value") is True
                    for t in span.get("tags", [])
                )
                for span in trace.get("spans", [])
            )
            if has_signal:
                unique_traces.append(trace)
        logger.info(
            f"Collected {len(unique_traces)} unique traces with signal "
            f"({len(seen_ids)} unique, {len(all_traces)} total across services)"
        )

        # Extract error operations
        operations = extract_error_operations(unique_traces)
        logger.info(f"Found {len(operations)} unique (service, operation) error pairs")

        # Save outputs
        traces_dir = output_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        traces_path = traces_dir / f"traces-{event_id}-{flag}-error.json"
        with open(traces_path, "w") as f:
            json.dump(
                {"wall_clock_utc": wall_clock_utc, "data": unique_traces}, f, indent=2
            )
        logger.info(f"Saved raw traces to {traces_path}")

        ops_dir = output_dir / "operations"
        ops_dir.mkdir(parents=True, exist_ok=True)
        ops_path = ops_dir / f"operations-{event_id}-{flag}-error.json"
        with open(ops_path, "w") as f:
            json.dump(operations, f, indent=2)
        logger.info(f"Saved operations to {ops_path}")

        # Print summary
        for entry in operations:
            event_names = [e.get("event", "?") for e in entry["events"]]
            events_str = f" events={event_names}" if event_names else ""
            print(f"  {entry['service']} :: {entry['operationName']}{events_str}")


if __name__ == "__main__":
    main()
