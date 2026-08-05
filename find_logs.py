#!/usr/bin/env python3
"""Find logs from OpenSearch for all incidents in a data directory.

Reads event JSON files from ``data-dir/events/``, then queries OpenSearch for
all logs within a time window around each incident. We deliberately do *not*
filter by error severity or error-related keywords: keyword/severity matching
produced too many false positives (benign lines mentioning "error", "timeout",
etc.) and false negatives (real symptoms that don't use any of those words), so
we return the full window and leave relevance judgements to downstream analysis.
Outputs the matching logs and a per-service summary.

Examples::

    # Query error logs for all events in a data directory:
    uv run python find_logs.py -dd data-0216 -od out

    # With a custom time window (default 10 minutes each side):
    uv run python find_logs.py -dd data-0216 -od out --window 20

    # Asymmetric window (5 min before, 15 min after):
    uv run python find_logs.py -dd data-0216 -od out --left-window 5 --right-window 15

    # Process a single event only (matches the "id" field in event JSON):
    uv run python find_logs.py -dd data-0216 -od out --event-id deployment-2a

    # Output files per event (e.g., id=bonus-2a, flag=loadGeneratorFloodHomepage):
    #   out/logs/logs-bonus-2a-loadGeneratorFloodHomepage-error.json
    #   out/logs/unique-logs-bonus-2a-loadGeneratorFloodHomepage-error.json
"""

import json
import logging
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import requests
from tabulate import tabulate

from utils import get_base_parser, setup_logging

logger = logging.getLogger(__name__)

OPENSEARCH_BASE = "http://localhost:9200"
BATCH_SIZE = 10000
SCROLL_TTL = "2m"


def build_query(start_iso: str, end_iso: str) -> dict:
    """Build an OpenSearch query for all logs in a time range.

    We intentionally return the full time window rather than filtering by
    severity or keywords, which produced too many false positives/negatives.
    """
    return {
        "size": BATCH_SIZE,
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start_iso,
                                "lte": end_iso,
                            }
                        }
                    }
                ]
            }
        },
        "sort": [{"@timestamp": "asc"}],
    }


def fetch_error_logs(start_iso: str, end_iso: str) -> list[dict]:
    """Fetch error-relevant logs from OpenSearch using the scroll API."""
    # Determine which indices to query (cover start and end dates)
    start_date = start_iso[:10]
    end_date = end_iso[:10]
    if start_date == end_date:
        index = f"otel-logs-{start_date}"
    else:
        index = f"otel-logs-{start_date},otel-logs-{end_date}"

    query = build_query(start_iso, end_iso)
    logger.info(f"Querying index {index} for error-relevant logs")

    # Initial search with scroll
    resp = requests.post(
        f"{OPENSEARCH_BASE}/{index}/_search?scroll={SCROLL_TTL}", json=query
    )
    resp.raise_for_status()
    data = resp.json()

    scroll_id = data["_scroll_id"]
    hits = data["hits"]["hits"]
    all_hits = list(hits)
    total = data["hits"]["total"]["value"]
    logger.info(f"Total matching documents: {total}")
    logger.info(f"Batch 1: fetched {len(hits)} documents")

    # Scroll through remaining results
    while len(hits) == BATCH_SIZE:
        resp = requests.post(
            f"{OPENSEARCH_BASE}/_search/scroll",
            json={"scroll": SCROLL_TTL, "scroll_id": scroll_id},
        )
        resp.raise_for_status()
        data = resp.json()
        scroll_id = data["_scroll_id"]
        hits = data["hits"]["hits"]
        all_hits.extend(hits)
        if hits:
            logger.info(f"Fetched {len(hits)} more documents (total: {len(all_hits)})")

    # Clear scroll context
    try:
        requests.delete(
            f"{OPENSEARCH_BASE}/_search/scroll", json={"scroll_id": scroll_id}
        )
    except Exception:
        logger.debug("Failed to clear scroll context")

    return all_hits


def deduplicate_by_body(hits: list[dict]) -> list[dict]:
    """Deduplicate log hits by their body field, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[dict] = []
    for hit in hits:
        body = hit.get("_source", {}).get("body", "")
        if body not in seen:
            seen.add(body)
            hit["log_id"] = uuid.uuid4().hex
            unique.append(hit)
    return unique


def summarize_by_service(hits: list[dict]) -> Counter:
    """Count log hits by service name."""
    counter: Counter = Counter()
    for hit in hits:
        source = hit.get("_source", {})
        service = (
            source.get("resource", {}).get("service.name")
            or source.get("resource", {}).get("service", {}).get("name")
            or "unknown"
        )
        counter[service] += 1
    return counter


def main() -> None:
    """Query OpenSearch for error-relevant logs around all incidents."""
    parser = get_base_parser()
    parser.description = (
        "Find error-relevant logs from OpenSearch for all incidents in a data directory"
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

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

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
        start_iso = (incident_dt - timedelta(minutes=left_minutes)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        end_iso = (incident_dt + timedelta(minutes=right_minutes)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        logger.info(f"Querying logs window=[{start_iso} .. {end_iso}]")
        hits = fetch_error_logs(start_iso, end_iso)
        logger.info(f"Fetched {len(hits)} error-relevant log entries")

        # Save full output
        out_path = logs_dir / f"logs-{event_id}-{flag}-error.json"
        with open(out_path, "w") as f:
            json.dump({"wall_clock_utc": wall_clock_utc, "data": hits}, f, indent=2)
        logger.info(f"Saved logs to {out_path}")

        # Deduplicate by body and save unique logs
        unique_hits = deduplicate_by_body(hits)
        logger.info(
            f"Deduplicated {len(hits)} logs to {len(unique_hits)} unique entries"
        )

        unique_path = logs_dir / f"unique-logs-{event_id}-{flag}-error.json"
        with open(unique_path, "w") as f:
            json.dump(
                {"wall_clock_utc": wall_clock_utc, "data": unique_hits}, f, indent=2
            )
        logger.info(f"Saved unique logs to {unique_path}")

        # Print summary table by service
        total_counts = summarize_by_service(hits)
        unique_counts = summarize_by_service(unique_hits)
        all_services = sorted(total_counts.keys(), key=lambda s: -total_counts[s])

        rows = [
            [service, unique_counts[service], total_counts[service]]
            for service in all_services
        ]
        table = tabulate(
            rows, headers=["Service", "Unique", "Total"], tablefmt="github"
        )

        print(f"\nError-relevant logs ({len(unique_hits)} unique, {len(hits)} total):")
        print(table)


if __name__ == "__main__":
    main()
