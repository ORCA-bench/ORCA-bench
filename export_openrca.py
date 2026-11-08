#!/usr/bin/env python3
"""Export a loaded ORCA-bench snapshot into OpenRCA's telemetry CSV layout.

Reads the backends of a snapshot loaded by ``./load_snapshot.sh`` -- Prometheus
for metrics, OpenSearch for Jaeger spans and OTel logs -- and writes the flat
per-day CSVs OpenRCA expects:

    {SYSTEM}/telemetry/{YYYY_MM_DD}/
    |-- log/log_service.csv        log_id,timestamp,cmdb_id,log_name,value
    |-- metric/metric_app.csv      timestamp,rr,sr,cnt,mrt,tc
    |-- metric/metric_container.csv timestamp,cmdb_id,kpi_name,value
    `-- trace/trace_span.csv       timestamp,cmdb_id,parent_id,span_id,trace_id,duration

``cmdb_id`` is the OTel ``service.name`` throughout: it is flat, present in all
three telemetry types, and is the identity ORCA-bench incidents name -- the same
role ``cmdb_id`` plays in OpenRCA's Bank dataset, where every component is at
pod level.

Day directories are cut on **UTC**, unlike OpenRCA's own datasets which cut on
UTC+8. OpenRCA's ``main.generate`` takes a ``-t <timezone>`` argument, so pass
UTC there rather than distorting the timestamps here.

Examples::

    # One day, to the default destination beside Bank/, Market/, Telecom/
    uv run python export_openrca.py --days 2026-04-19

    # The full range
    uv run python export_openrca.py --start 2026-04-19 --end 2026-04-24

    # Just the traces, somewhere else
    uv run python export_openrca.py --days 2026-04-19 --only trace -od /tmp/out
"""

import csv
import hashlib
import logging
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from utils import get_base_parser, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("/mnt/volume_nyc3_1777562302334/OpenRCA/dataset")
DEFAULT_PROMETHEUS_URL = "http://localhost:9090"
DEFAULT_OPENSEARCH_URL = "http://localhost:9200"

SPAN_INDEX = "jaeger-main-jaeger-span-{date}*"
LOG_INDEX = "otel-logs-{date}*"

# Matches find_logs.py, which scrolls the same cluster.
BATCH_SIZE = 10000
SCROLL_TTL = "2m"

# Prometheus caps a query_range response at 11000 points per series; 6h at a
# 60s step is 360, so these windows exist to bound memory, not to dodge that.
WINDOW_SECONDS = 6 * 3600

# flagd long-poll reconnection cycles: ~600s spans that appear in every
# observation window regardless of incidents. find_traces.py:40 classifies
# these as background noise, and counting them as transactions puts a 600s mrt
# on flagd in metric_app. They stay in trace_span.csv -- they are real spans --
# but are left out of the roll-up unless --include-noise-spans is passed.
NOISE_OPERATIONS = frozenset({"flagd.evaluation.v1.Service/EventStream"})

# Resolution order for cmdb_id on a Prometheus series.  container_name comes
# before job because the collector's docker receiver labels container_* metrics
# with only container_name and host_name -- and its values are the plain service
# names, so they land in the same namespace as service_name.  host_name is the
# last resort: the hostmetrics system_* series carry nothing else, and a
# host-level pseudo-component is closer to Bank (which has host-level OSLinux_*
# KPIs on every component) than discarding them as "unknown".
CMDB_LABELS = (
    "service_name",
    "service",
    "container_name",
    "job",
    "instance",
    "host_name",
)

# Prometheus exposes no metric metadata for a frozen snapshot (it comes from
# live scrape/OTLP targets), so counters are recognised by the conventional
# suffixes instead.  Counters are summed when collapsing duplicate series,
# everything else is averaged.
COUNTER_SUFFIXES = ("_total", "_count", "_sum")

# Metadata markers rather than measurements: target_info is a single series
# pinned at 1 whose only content is the ~20 resource labels it carries, which
# would make for an enormous kpi_name and no signal.  Bank has no equivalent.
DROP_METRICS = frozenset({"target_info"})

# Labels that identify the emitting process rather than what is being measured.
# They are dropped from kpi_name: they duplicate cmdb_id, and several are
# per-process UUIDs that change on restart, which would make kpi_name unstable
# across the export (Bank's kpi_name is stable).
IDENTITY_LABELS = frozenset(
    {
        "service_name",
        "service",
        "job",
        "instance",
        "service_instance_id",
        "service_namespace",
        "service_version",
        "host_name",
        "host_arch",
        "container_id",
        "os_type",
        "os_description",
        "telemetry_sdk_language",
        "telemetry_sdk_name",
        "telemetry_sdk_version",
        "telemetry_distro_name",
        "telemetry_distro_version",
        "process_pid",
        "process_executable_path",
        "process_command_args",
        "process_runtime_name",
        "process_runtime_version",
        "process_runtime_description",
        "process_command_line",
        "process_owner",
        "container_name",
        "os_name",
        "os_version",
        "os_build_id",
    }
)

OPENRCA_HEADERS = {
    "metric_container": ["timestamp", "cmdb_id", "kpi_name", "value"],
    "metric_app": ["timestamp", "rr", "sr", "cnt", "mrt", "tc"],
    "trace_span": ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"],
    "log_service": ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
}

UNKNOWN = "unknown"

# OpenRCA's own CSVs are LF-terminated; csv.writer defaults to CRLF.
CSV_DIALECT = {"lineterminator": "\n"}


# ── helpers ─────────────────────────────────────────────────────────────────


def parse_day(text: str) -> datetime:
    """Parse YYYY-MM-DD (or YYYY_MM_DD) as UTC midnight."""
    return datetime.strptime(text.replace("_", "-"), "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )


def day_dir_name(day: datetime) -> str:
    """OpenRCA names day directories YYYY_MM_DD."""
    return day.strftime("%Y_%m_%d")


def preflight(session: requests.Session, prometheus_url: str, opensearch_url: str) -> None:
    """Fail early and legibly when the snapshot backends are not up."""
    for name, url in (
        ("Prometheus", f"{prometheus_url}/api/v1/status/buildinfo"),
        ("OpenSearch", f"{opensearch_url}/_cluster/health"),
    ):
        try:
            session.get(url, timeout=10).raise_for_status()
        except Exception as exc:
            raise SystemExit(
                f"{name} is not reachable at {url} ({exc}).\n"
                "Load a snapshot first, e.g.:\n"
                '  COMPOSE_SERVICES="prometheus opensearch" ./load_snapshot.sh '
                "data-0418 20260424T192542Z-64df31d58b702c09"
            ) from exc
    logger.info("Prometheus and OpenSearch are reachable")


def scroll_index(
    session: requests.Session,
    opensearch_url: str,
    index: str,
    body: dict,
) -> Iterator[list[dict]]:
    """Yield _source dicts for every hit, via the scroll API.

    Mirrors find_logs.py's fetch_error_logs, but yields batches instead of
    accumulating everything -- these indices hold hundreds of thousands of docs
    per day and the caller streams straight to CSV.
    """
    resp = session.post(
        f"{opensearch_url}/{index}/_search",
        params={"scroll": SCROLL_TTL},
        json=body,
        timeout=300,
    )
    if resp.status_code == 404:
        logger.warning(f"  index {index} not found -- skipping")
        return
    resp.raise_for_status()
    data = resp.json()

    scroll_id = data.get("_scroll_id")
    hits = data["hits"]["hits"]
    total = data["hits"]["total"]["value"]
    logger.info(f"  {index}: {total:,} documents")

    try:
        # Only an empty batch means exhausted. A short batch does not: with
        # track_total_hits unset the sorted query early-terminates, and
        # treating a short batch as the end silently truncated the export
        # (30,885 of 484,471 log records).
        while hits:
            yield [h["_source"] for h in hits]
            resp = session.post(
                f"{opensearch_url}/_search/scroll",
                json={"scroll": SCROLL_TTL, "scroll_id": scroll_id},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            scroll_id = data.get("_scroll_id")
            hits = data["hits"]["hits"]
    finally:
        if scroll_id:
            try:
                session.delete(
                    f"{opensearch_url}/_search/scroll",
                    json={"scroll_id": scroll_id},
                    timeout=30,
                )
            except Exception:
                logger.debug("Failed to clear scroll context")


def dig(source, dotted: str):
    """Look up a dotted field, tolerating however OpenSearch shaped it.

    These documents store flattened dotted keys, and _source filtering
    reshapes them inconsistently: asking for resource.service.name,
    severity.text and instrumentationScope.name in one request comes back as
    {"resource": {"service.name": ...}} but {"severity": {"text": ...}}.  Try
    the literal flat key, then progressively longer prefixes, accepting a flat
    remainder at any level.
    """
    if not isinstance(source, dict):
        return None
    if dotted in source:
        return source[dotted]
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        head = ".".join(parts[:i])
        if head in source:
            found = dig(source[head], ".".join(parts[i:]))
            if found is not None:
                return found
    return None


def parent_span_id(source: dict) -> str:
    """Return the CHILD_OF parent's spanID, or "" for a root span.

    The index mapping has a top-level parentSpanID field but Jaeger never
    populates it (verified: 0 documents), so the parent only exists inside the
    nested references array.
    """
    for ref in source.get("references") or ():
        if ref.get("refType") == "CHILD_OF":
            return ref.get("spanID", "")
    return ""


# ── traces + metric_app ─────────────────────────────────────────────────────


def export_traces(
    session: requests.Session,
    opensearch_url: str,
    day: datetime,
    trace_path: Path | None,
    include_noise_spans: bool = False,
) -> dict:
    """Write trace_span.csv, accumulating metric_app buckets along the way.

    metric_app is derived from the same pass rather than from Prometheus
    spanmetrics so the two files reconcile by construction. Only root spans
    count as transactions, which is the closest analogue to Bank's per-tc
    request counts.

    trace_path may be None, which walks the spans for the metric_app roll-up
    without writing the (much larger) trace CSV.
    """
    index = SPAN_INDEX.format(date=day.strftime("%Y-%m-%d"))
    start_ms = int(day.timestamp() * 1000)
    end_ms = start_ms + 86_400_000

    body = {
        "size": BATCH_SIZE,
        "track_total_hits": True,
        "query": {
            "range": {"startTimeMillis": {"gte": start_ms, "lt": end_ms}}
        },
        "sort": [{"startTimeMillis": "asc"}],
        "_source": {
            "includes": [
                "startTimeMillis",
                "duration",
                "spanID",
                "traceID",
                "references",
                "operationName",
                "process.serviceName",
                "tag",
            ]
        },
    }

    # (minute, service) -> [cnt, duration_sum_ms, errors, responded]
    buckets: dict = defaultdict(lambda: [0, 0.0, 0, 0])
    written = 0
    noise_skipped = 0

    fh = None
    writer = None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        fh = trace_path.open("w", newline="")
        writer = csv.writer(fh, **CSV_DIALECT)
        writer.writerow(OPENRCA_HEADERS["trace_span"])

    try:
        for batch in scroll_index(session, opensearch_url, index, body):
            rows = []
            for src in batch:
                ts_ms = src.get("startTimeMillis")
                if ts_ms is None:
                    continue
                service = dig(src, "process.serviceName") or UNKNOWN
                duration_us = src.get("duration")
                # Jaeger stores duration in microseconds; OpenRCA uses ms.
                duration_ms = round(duration_us / 1000) if duration_us is not None else 0
                parent = parent_span_id(src)
                if writer is not None:
                    rows.append(
                        [
                            ts_ms,
                            service,
                            parent,
                            src.get("spanID", ""),
                            src.get("traceID", ""),
                            duration_ms,
                        ]
                    )

                if not parent:  # root span == one transaction
                    if (
                        not include_noise_spans
                        and src.get("operationName") in NOISE_OPERATIONS
                    ):
                        noise_skipped += 1
                        continue
                    b = buckets[(ts_ms // 60000 * 60, service)]
                    b[0] += 1
                    b[1] += duration_ms
                    if (src.get("tag") or {}).get("error") is True:
                        b[2] += 1
                    if duration_us is not None:
                        b[3] += 1
            if writer is not None:
                writer.writerows(rows)
            written += len(batch)
    finally:
        if fh is not None:
            fh.close()

    if trace_path is not None:
        logger.info(f"  wrote {written:,} spans -> {trace_path.name}")
    else:
        logger.info(f"  walked {written:,} spans for metric_app")
    if noise_skipped:
        logger.info(
            f"  excluded {noise_skipped:,} background-noise root spans from "
            "metric_app (see NOISE_OPERATIONS)"
        )
    return buckets


def write_metric_app(buckets: dict, path: Path) -> None:
    """Write metric_app.csv from the per-(minute, service) span roll-up.

    rr is the fraction of transactions that produced a duration and sr the
    fraction without an error tag, both as percentages -- matching Bank, where
    both columns are percentages that sit at 100.0 in healthy windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, **CSV_DIALECT)
        writer.writerow(OPENRCA_HEADERS["metric_app"])
        for (minute, service) in sorted(buckets):
            cnt, dur_sum, errors, responded = buckets[(minute, service)]
            if not cnt:
                continue
            writer.writerow(
                [
                    minute,
                    round(100.0 * responded / cnt, 2),
                    round(100.0 * (cnt - errors) / cnt, 2),
                    cnt,
                    round(dur_sum / cnt, 2),
                    service,
                ]
            )
    logger.info(f"  wrote {len(buckets):,} rows -> {path.name}")


# ── logs ────────────────────────────────────────────────────────────────────


def export_logs(
    session: requests.Session,
    opensearch_url: str,
    day: datetime,
    path: Path,
) -> None:
    """Write log_service.csv.

    Field names here are the flattened dotted keys OpenSearch actually stores:
    resource.service.name, severity.text, instrumentationScope.name. They are
    text fields with no keyword subfield, so the cluster refuses aggregations --
    everything is computed while streaming.
    """
    index = LOG_INDEX.format(date=day.strftime("%Y-%m-%d"))
    start_iso = day.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = (day + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    body = {
        "size": BATCH_SIZE,
        "track_total_hits": True,
        "query": {"range": {"@timestamp": {"gte": start_iso, "lt": end_iso}}},
        "sort": [{"@timestamp": "asc"}],
        "_source": {
            "includes": [
                "@timestamp",
                "body",
                "resource.service.name",
                "severity.text",
                "instrumentationScope.name",
            ]
        },
    }

    written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, **CSV_DIALECT)
        writer.writerow(OPENRCA_HEADERS["log_service"])
        for batch in scroll_index(session, opensearch_url, index, body):
            rows = []
            for src in batch:
                raw_ts = src.get("@timestamp")
                if not raw_ts:
                    continue
                try:
                    ts = int(
                        datetime.fromisoformat(
                            raw_ts.replace("Z", "+00:00")
                        ).timestamp()
                    )
                except ValueError:
                    continue
                service = dig(src, "resource.service.name") or UNKNOWN
                # Bank's log_name is the log source (gc, catalina, ...); the
                # closest analogue here is the instrumentation scope, falling
                # back to severity.
                log_name = (
                    dig(src, "instrumentationScope.name")
                    or dig(src, "severity.text")
                    or "log"
                )
                body_text = dig(src, "body") or ""
                log_id = hashlib.md5(
                    f"{ts}|{service}|{body_text}".encode()
                ).hexdigest()
                rows.append([log_id, ts, service, log_name, body_text])
            writer.writerows(rows)
            written += len(rows)

    logger.info(f"  wrote {written:,} log records -> {path.name}")


# ── metrics ─────────────────────────────────────────────────────────────────


def build_kpi_name(metric: dict, cmdb_label: str | None) -> str:
    """Flatten a series' labels into a stable kpi_name.

    Mirrors how Bank encodes dimensions into the name itself, e.g.
    Tomcat-Requests_7441-"http-nio-8003"_ErrorCountRequestInfo.
    """
    name = metric.get("__name__", "")
    extras = sorted(
        (k, v)
        for k, v in metric.items()
        if k != "__name__" and k != cmdb_label and k not in IDENTITY_LABELS
    )
    if not extras:
        return name
    return name + "".join(f"-{k}={v}" for k, v in extras)


def metric_names(
    session: requests.Session, prometheus_url: str, include_histograms: bool
) -> list:
    """List metric names, optionally dropping histogram bucket series."""
    resp = session.get(f"{prometheus_url}/api/v1/label/__name__/values", timeout=60)
    resp.raise_for_status()
    names = [n for n in (resp.json().get("data") or []) if n not in DROP_METRICS]
    if not include_histograms:
        kept = [n for n in names if not n.endswith("_bucket")]
        logger.info(
            f"  {len(names)} metric names, {len(names) - len(kept)} histogram "
            f"bucket names dropped (use --include-histograms to keep them)"
        )
        return kept
    logger.info(f"  {len(names)} metric names (histograms included)")
    return names


def export_metrics(
    session: requests.Session,
    prometheus_url: str,
    day: datetime,
    path: Path,
    step: int,
    include_histograms: bool,
) -> None:
    """Write metric_container.csv by range-querying every series at `step`."""
    names = metric_names(session, prometheus_url, include_histograms)
    day_start = int(day.timestamp())
    day_end = day_start + 86400 - step  # inclusive end -> 1440 points at 60s

    written = 0
    collisions = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, **CSV_DIALECT)
        writer.writerow(OPENRCA_HEADERS["metric_container"])

        for i, name in enumerate(names, 1):
            window = day_start
            while window <= day_end:
                w_end = min(window + WINDOW_SECONDS - step, day_end)
                resp = session.get(
                    f"{prometheus_url}/api/v1/query_range",
                    params={
                        "query": name,
                        "start": window,
                        "end": w_end,
                        "step": step,
                    },
                    timeout=300,
                )
                if resp.status_code != 200:
                    logger.warning(f"  {name}: HTTP {resp.status_code} -- skipping window")
                    window += WINDOW_SECONDS
                    continue
                payload = resp.json()
                if payload.get("status") != "success":
                    logger.warning(f"  {name}: {payload.get('error')} -- skipping window")
                    window += WINDOW_SECONDS
                    continue

                # Series that differ only by labels dropped from kpi_name --
                # in practice the per-process service_instance_id/instance
                # UUIDs, so one series per incarnation of a service that
                # restarted mid-window -- would otherwise emit duplicate
                # (timestamp, cmdb_id, kpi_name) tuples and leave the file
                # ambiguous to anything that pivots on them.  Collapse them.
                bucket: dict = defaultdict(list)
                for series in payload["data"]["result"]:
                    metric = series.get("metric", {})
                    cmdb_label = next(
                        (lbl for lbl in CMDB_LABELS if metric.get(lbl)), None
                    )
                    cmdb_id = metric.get(cmdb_label, UNKNOWN) if cmdb_label else UNKNOWN
                    # An instance label is host:port; the docker receiver
                    # labels the same component by bare name, so drop the port
                    # to keep one component out of "postgresql" and
                    # "postgresql:5432".
                    if cmdb_label == "instance" and ":" in cmdb_id:
                        cmdb_id = cmdb_id.rsplit(":", 1)[0]
                    kpi_name = build_kpi_name(metric, cmdb_label)
                    for ts, value in series.get("values", []):
                        if value in ("NaN", "+Inf", "-Inf"):
                            continue
                        bucket[(int(ts), cmdb_id, kpi_name)].append(value)

                rows = []
                is_counter = name.endswith(COUNTER_SUFFIXES)
                for key in sorted(bucket):
                    values = bucket[key]
                    if len(values) == 1:
                        # Pass the raw Prometheus string through: reformatting
                        # via float() would introduce repr drift.
                        out = values[0]
                    else:
                        collisions += len(values) - 1
                        nums = [float(v) for v in values]
                        out = repr(sum(nums) if is_counter else sum(nums) / len(nums))
                    rows.append([key[0], key[1], key[2], out])
                writer.writerows(rows)
                written += len(rows)
                window += WINDOW_SECONDS

            if i % 50 == 0:
                logger.info(f"  {i}/{len(names)} metrics, {written:,} rows so far")

    if collisions:
        logger.info(
            f"  collapsed {collisions:,} duplicate samples (series differing only "
            "by a dropped per-process label); counters summed, others averaged"
        )
    logger.info(f"  wrote {written:,} samples -> {path.name}")


# ── driver ──────────────────────────────────────────────────────────────────


def main() -> None:
    """Export one or more days of a loaded snapshot into OpenRCA layout."""
    parser = get_base_parser()
    parser.description = (
        "Export a loaded ORCA-bench snapshot into OpenRCA's telemetry CSV layout"
    )
    parser.set_defaults(output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--system",
        default="OtelDemo",
        help="OpenRCA {SYSTEM} directory name (default: OtelDemo)",
    )
    parser.add_argument(
        "--days",
        nargs="+",
        help="Explicit days to export, YYYY-MM-DD (overrides --start/--end)",
    )
    parser.add_argument("--start", help="First day to export, YYYY-MM-DD")
    parser.add_argument("--end", help="Last day to export, inclusive, YYYY-MM-DD")
    parser.add_argument(
        "--step", type=int, default=60, help="Metric sampling step in seconds (default: 60)"
    )
    parser.add_argument(
        "--prometheus-url", default=DEFAULT_PROMETHEUS_URL, help="Prometheus base URL"
    )
    parser.add_argument(
        "--opensearch-url", default=DEFAULT_OPENSEARCH_URL, help="OpenSearch base URL"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["metric", "metric_app", "trace", "log"],
        help="Export only these telemetry types (default: all)",
    )
    parser.add_argument(
        "--include-noise-spans",
        action="store_true",
        help="Count flagd long-poll spans as metric_app transactions (see NOISE_OPERATIONS)",
    )
    parser.add_argument(
        "--include-histograms",
        action="store_true",
        help="Keep Prometheus *_bucket series (62%% of series; 2.6x the output)",
    )
    args = parser.parse_args()
    setup_logging(getattr(logging, args.log_level))

    if args.days:
        days = [parse_day(d) for d in args.days]
    elif args.start and args.end:
        first, last = parse_day(args.start), parse_day(args.end)
        if last < first:
            raise SystemExit("--end is before --start")
        days = []
        cur = first
        while cur <= last:
            days.append(cur)
            cur += timedelta(days=1)
    else:
        raise SystemExit("Pass --days, or both --start and --end")

    wanted = set(args.only) if args.only else {"metric", "metric_app", "trace", "log"}
    # metric_app is derived from the span pass, so it needs the traces walked.
    needs_traces = "trace" in wanted or "metric_app" in wanted

    session = requests.Session()
    preflight(session, args.prometheus_url, args.opensearch_url)

    for day in days:
        root = Path(args.output_dir) / args.system / "telemetry" / day_dir_name(day)
        logger.info(f"=== {day_dir_name(day)} -> {root} ===")

        if needs_traces:
            buckets = export_traces(
                session,
                args.opensearch_url,
                day,
                root / "trace" / "trace_span.csv" if "trace" in wanted else None,
                args.include_noise_spans,
            )
            if "metric_app" in wanted:
                write_metric_app(buckets, root / "metric" / "metric_app.csv")

        if "log" in wanted:
            export_logs(
                session,
                args.opensearch_url,
                day,
                root / "log" / "log_service.csv",
            )

        if "metric" in wanted:
            export_metrics(
                session,
                args.prometheus_url,
                day,
                root / "metric" / "metric_container.csv",
                args.step,
                args.include_histograms,
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
