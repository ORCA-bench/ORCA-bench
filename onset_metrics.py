#!/usr/bin/env python3
r"""Detect onset times for metric changes during an incident.

Queries Grafana/Prometheus for each metric in per-run JSON files, analyzes
the time series around the incident window, and detects when the
spike/collapse/create first occurred using Claude CLI.

Onset detection verifies that each metric change aligns with the incident
timing and establishes the propagation timeline (root cause metrics should
appear before downstream symptoms).

Input::

    {metrics_dir}/metrics-{flag}-*.json   (per-run metric lists)
    Incident time (UTC)

Output::

    {output_dir}/metrics_onset/metrics-{flag}-*.json   (same filenames)

Each entry gets additional ``onset_utc``, ``onset_offset_seconds``, and
``onset_reasoning`` fields.

Examples::

    # Detect onset times for productCatalogFailure:
    uv run python onset_metrics.py -f productCatalogFailure \
        -t "2026-04-02 06:25:59 UTC" -od metrics-0401

    # Custom input directory and window:
    uv run python onset_metrics.py -f productCatalogFailure \
        -t "2026-04-02 06:25:59 UTC" -od metrics-0401 \
        -dir metrics-0401/metrics -w 30

    # Use a different model:
    uv run python onset_metrics.py -f productCatalogFailure \
        -t "2026-04-02 06:25:59 UTC" -od metrics-0401 -m sonnet -e high
"""

import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from grafana_client import GrafanaApi
from grafana_client.model import DatasourceIdentifier

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "opus"
_DEFAULT_EFFORT = "max"


# ---------------------------------------------------------------------------
# Grafana query
# ---------------------------------------------------------------------------


def query_metric(
    grafana: GrafanaApi,
    datasource_uid: str,
    expression: str,
    start_ts: int,
    end_ts: int,
    interval_ms: int,
) -> list[tuple[list[float], list[float], str]]:
    """Run a PromQL query and return a list of (timestamps_ms, values, label) tuples."""
    response = grafana.datasource.smartquery(
        datasource=DatasourceIdentifier(uid=datasource_uid),
        expression=expression,
        attrs={
            "time_from": start_ts,
            "time_to": end_ts,
            "intervalMs": interval_ms,
        },
    )
    series = []
    for frame in response.get("results", {}).get("test", {}).get("frames", []):
        label = frame["schema"]["fields"][1]["config"].get("displayNameFromDS", "")
        timestamps = frame["data"]["values"][0]
        values = frame["data"]["values"][1]
        series.append((timestamps, values, label))
    return series


# ---------------------------------------------------------------------------
# Onset detection via Claude
# ---------------------------------------------------------------------------

_ONSET_PROMPT = """\
You are an expert SRE analyzing a Prometheus time series to detect when a \
metric change occurred during an incident.

Metric query: {query}
Metric type: {metric_type}
Incident time: {incident_time}

The time series data (timestamp UTC -> value):
{series_data}

Based on the data above, identify the **onset time** — the first timestamp \
where the metric clearly begins to {change_description}.

- For "spiked" or "created": look AFTER the incident time for the first \
  timestamp where the value rises significantly above the pre-incident \
  baseline (or appears for the first time if the series didn't exist before).
- For "collapsed": the pre-incident signal may be periodic or oscillating. \
  A collapse means the active pattern ceases — the signal goes flat or to \
  zero and never recovers. Identify the LAST timestamp where the \
  pre-incident pattern is still active (i.e. the last non-trivial value \
  before the signal dies). The onset may be slightly before or after the \
  incident time. If the signal cessation clearly does not align with the \
  incident timing (e.g. it died well before or was already inactive), \
  respond with null.

If no clear onset is visible, respond with "none".

Respond with ONLY valid JSON:
{{"onset_utc": "YYYY-MM-DD HH:MM:SS UTC", "reasoning": "brief explanation"}}
or
{{"onset_utc": null, "reasoning": "brief explanation"}}"""


def _format_series_data(
    timestamps_ms: list[float],
    values: list[float],
    incident_ts_ms: float,
) -> str:
    """Format time series as human-readable text with incident marker."""
    lines = []
    for ts_ms, val in zip(timestamps_ms, values):
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        marker = " <-- INCIDENT" if abs(ts_ms - incident_ts_ms) < 30000 else ""
        val_str = f"{val:.6g}" if val is not None else "null"
        lines.append(f"  {dt:%Y-%m-%d %H:%M:%S UTC}  {val_str}{marker}")
    return "\n".join(lines)


def detect_onset_claude(
    timestamps_ms: list[float],
    values: list[float],
    incident_ts_ms: float,
    query: str,
    metric_type: str,
    incident_time_str: str,
    model: str = _DEFAULT_MODEL,
    effort: str = _DEFAULT_EFFORT,
) -> tuple[str | None, str]:
    """Use Claude to detect the onset time from raw time series data.

    Returns (onset_utc_string_or_None, reasoning).
    """
    change_desc = {
        "spiked": "spike above baseline",
        "created": "appear (from zero or non-existent)",
        "collapsed": "cease its pre-incident activity (go flat or to zero)",
    }.get(metric_type, "change")

    series_text = _format_series_data(timestamps_ms, values, incident_ts_ms)

    prompt = _ONSET_PROMPT.format(
        query=query,
        metric_type=metric_type,
        incident_time=incident_time_str,
        series_data=series_text,
        change_description=change_desc,
    )

    result = subprocess.run(
        ["claude", "--model", model, "--effort", effort, "-p", prompt],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.warning(f"    Claude CLI failed: {result.stderr[:200]}")
        return None, "Claude CLI error"

    raw = result.stdout.strip()

    # Parse JSON (handle code fences)
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    # Extract JSON object
    json_match = re.search(r"\{[^{}]*\"onset_utc\"[^{}]*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"    Failed to parse response: {raw[:200]}")
        return None, "Parse error"

    return parsed.get("onset_utc"), parsed.get("reasoning", "")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _process_entries(
    entries: list[dict],
    grafana: GrafanaApi,
    datasource_uid: str,
    start_ts: int,
    end_ts: int,
    interval_ms: int,
    incident_ts_ms: int,
    incident_dt: datetime,
    incident_time: str,
    model: str = _DEFAULT_MODEL,
    effort: str = _DEFAULT_EFFORT,
) -> None:
    """Add onset fields to each metric entry in-place."""
    for i, entry in enumerate(entries):
        query = entry.get("query", "").strip()
        if not query:
            continue
        mtype = entry.get("type", "")
        if "inconclusive" in mtype.lower():
            logger.info(
                f"  [{i + 1}/{len(entries)}] Skipping inconclusive: {query[:80]}..."
            )
            continue
        if "onset_utc" in entry:
            logger.info(
                f"  [{i + 1}/{len(entries)}] Skipping (onset exists): {query[:80]}..."
            )
            continue
        logger.info(f"  [{i + 1}/{len(entries)}] {query[:80]}...")

        try:
            series = query_metric(
                grafana, datasource_uid, query, start_ts, end_ts, interval_ms
            )
        except Exception as e:
            logger.warning(f"    Query failed: {e}")
            entry["onset_utc"] = None
            entry["onset_offset_seconds"] = None
            entry["onset_reasoning"] = f"Query failed: {e}"
            continue

        if not series:
            logger.warning("    No data returned")
            entry["onset_utc"] = None
            entry["onset_offset_seconds"] = None
            entry["onset_reasoning"] = "No data returned from Prometheus"
            continue

        # Use the first series (most queries return a single series)
        timestamps_ms, values, _label = series[0]

        onset_utc, reasoning = detect_onset_claude(
            timestamps_ms,
            values,
            incident_ts_ms,
            query,
            mtype,
            incident_time,
            model,
            effort,
        )

        entry["onset_utc"] = onset_utc
        entry["onset_reasoning"] = reasoning

        if onset_utc is not None:
            try:
                clean_onset = re.sub(r"\s*(?:UTC|utc)\s*$", "", onset_utc.strip())
                onset_dt = datetime.strptime(clean_onset, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
                offset_sec = int((onset_dt - incident_dt).total_seconds())
                entry["onset_offset_seconds"] = offset_sec
                logger.info(f"    Onset: {onset_utc} (+{offset_sec}s) — {reasoning}")
            except ValueError:
                entry["onset_offset_seconds"] = None
                logger.info(f"    Onset: {onset_utc} — {reasoning}")
        else:
            entry["onset_offset_seconds"] = None
            logger.warning(f"    No onset detected — {reasoning}")


def detect_onsets(
    feature_flag: str,
    incident_time: str,
    metrics_dir: Path,
    output_dir: Path,
    grafana_url: str,
    datasource_uid: str,
    window_min: int,
    interval_ms: int,
    model: str = _DEFAULT_MODEL,
    effort: str = _DEFAULT_EFFORT,
) -> list[Path]:
    """Detect onset times for all per-run metric files for a feature flag."""
    prefix = f"metrics-{feature_flag}-"
    run_files = sorted(
        f
        for f in metrics_dir.glob(f"{prefix}*.json")
        if "-merged" not in f.name
        and "-labeled" not in f.name
        and f.name[len(prefix) :].split("-", 1)[0].isdigit()
    )
    if not run_files:
        logger.error(
            f"No run files found for exact flag '{feature_flag}' in {metrics_dir}"
        )
        sys.exit(1)

    logger.info(
        f"Found {len(run_files)} run file(s): " + ", ".join(f.name for f in run_files)
    )

    # Parse incident time
    clean_time = re.sub(r"\s*(?:UTC|utc)\s*$", "", incident_time.strip())
    incident_dt = datetime.strptime(clean_time, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    incident_ts = int(incident_dt.timestamp())
    incident_ts_ms = incident_ts * 1000
    start_ts = incident_ts - window_min * 60
    end_ts = incident_ts + window_min * 60

    logger.info(
        f"Incident: {incident_dt:%Y-%m-%d %H:%M:%S UTC}, "
        f"window: {window_min}min before/after"
    )

    # Connect to Grafana
    grafana = GrafanaApi.from_url(grafana_url)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for run_file in run_files:
        output_path = output_dir / run_file.name

        # If onset file already exists, load it and only process entries
        # that are missing onset data (resume support).
        if output_path.exists():
            logger.info(f"Resuming from existing onset file: {output_path}")
            entries = json.loads(output_path.read_text())
            pending = sum(
                1
                for e in entries
                if e.get("query", "").strip()
                and "onset_utc" not in e
                and "inconclusive" not in e.get("type", "").lower()
            )
            if pending == 0:
                logger.info("  All entries already have onset data, skipping.")
                output_paths.append(output_path)
                continue
            logger.info(f"  {pending} entries still need onset detection")
        else:
            logger.info(f"Processing {run_file}...")
            entries = json.loads(run_file.read_text())

        logger.info(f"  Loaded {len(entries)} entries")

        _process_entries(
            entries,
            grafana,
            datasource_uid,
            start_ts,
            end_ts,
            interval_ms,
            incident_ts_ms,
            incident_dt,
            incident_time,
            model,
            effort,
        )

        output_path.write_text(json.dumps(entries, indent=2) + "\n")
        logger.info(f"  Saved to {output_path}")
        output_paths.append(output_path)

    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Detect onset times for metric changes during an incident."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect when metrics spiked/collapsed/were created during an incident."
    )
    parser.add_argument(
        "--feature-flag",
        "-f",
        required=True,
        help="Feature flag name (e.g. productCatalogFailure)",
    )
    parser.add_argument(
        "--incident-time",
        "-t",
        required=True,
        help="Incident time in UTC (e.g. '2026-04-02 06:25:59 UTC')",
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        type=Path,
        default=Path("metrics-0401/metrics_onset"),
        help="Output directory for onset files (default: metrics-0401/metrics_onset)",
    )
    parser.add_argument(
        "--metrics-dir",
        "-dir",
        type=Path,
        default=None,
        help="Input directory containing metric JSON files (default: {output_dir}/metrics)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=_DEFAULT_MODEL,
        help=f"Claude model for onset detection (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--effort",
        "-e",
        type=str,
        default=_DEFAULT_EFFORT,
        help=f"Claude effort level (default: {_DEFAULT_EFFORT})",
    )
    parser.add_argument(
        "--grafana-url",
        default="http://admin:admin@localhost:8080/grafana/",
        help="Grafana URL (default: http://admin:admin@localhost:8080/grafana/)",
    )
    parser.add_argument(
        "--datasource-uid",
        default="webstore-metrics",
        help="Prometheus datasource UID (default: webstore-metrics)",
    )
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        default=20,
        help="Minutes before/after incident to query (default: 20)",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=60000,
        help="Query step interval in milliseconds (default: 60000 = 1min)",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()
    _setup_logging(args.log_level)

    output_dir: Path = args.output_dir
    metrics_dir = args.metrics_dir or (output_dir / "metrics")

    detect_onsets(
        feature_flag=args.feature_flag,
        incident_time=args.incident_time,
        metrics_dir=metrics_dir,
        output_dir=output_dir,
        grafana_url=args.grafana_url,
        datasource_uid=args.datasource_uid,
        window_min=args.window,
        interval_ms=args.interval_ms,
        model=args.model,
        effort=args.effort,
    )


if __name__ == "__main__":
    main()
