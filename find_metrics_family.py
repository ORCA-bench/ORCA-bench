#!/usr/bin/env python3
"""Find metrics via 5 parallel family-scoped sub-agents.

Splits the Prometheus metric namespace into 5 families (app, infra, db,
runtime, pipeline) and launches one Claude call per family in parallel.
Per-family results are saved individually, then merged and plotted.

The namespace is split to avoid overwhelming a single LLM call with the
full metric catalog, which leads to missed signals and hallucinated queries.

Examples::

    # Full pipeline (claude CLI):
    uv run python find_metrics_family.py -rf share-2w/json/d1-i2-adFailure-on.json -m opus -od out

    # Full pipeline (DigitalOcean / Gradient AI API):
    uv run python find_metrics_family.py -rf share-2w/json/d1-i2-adFailure-on.json -m opus -od out \
        --api-url https://inference.do-ai.run --api-key $MODEL_ACCESS_KEY

    # Loop over all rubrics:
    for f in share-2w/json/*.json; do
        uv run python find_metrics_family.py -rf "$f" -m opus -od out
    done

    # Re-plot existing metrics (skip Claude):
    uv run python find_metrics_family.py -rf share-2w/json/d1-i2-adFailure-on.json -m opus -od out --skip-claude
"""

import json
import logging
import math
import os
import re
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from grafana_client import GrafanaApi
from grafana_client.model import DatasourceIdentifier
from utils import get_base_parser, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric families — parsed from metrics_prefix.md at runtime
# ---------------------------------------------------------------------------

# Maps section headers in metrics_prefix.md to short family keys.
_SECTION_TO_KEY: dict[str, str] = {
    "Application & Service Metrics": "app",
    "Infrastructure Metrics": "infra",
    "Database & Data Store Metrics": "db",
    "Runtime Metrics": "runtime",
    "Telemetry Pipeline & Span-Derived Metrics": "pipeline",
}

DEFAULT_PREFIX_PATH = Path("prompts/metrics_prefix.md")


def parse_prefix_families(
    prefix_path: Path = DEFAULT_PREFIX_PATH,
) -> dict[str, dict[str, str]]:
    """Parse metrics_prefix.md into a dict keyed by family short name.

    Each value has ``name`` (the section heading without " Metrics" suffix)
    and ``prefixes`` (the raw markdown table for that section).

    Sections not listed in ``_SECTION_TO_KEY`` (e.g. Excluded, Totals) are
    skipped.
    """
    text = prefix_path.read_text()
    # Split on ## headings, keeping the heading text
    sections = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
    # sections[0] is preamble before the first ##, then alternating header/body

    families: dict[str, dict[str, str]] = {}
    for i in range(1, len(sections), 2):
        heading = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""

        key = _SECTION_TO_KEY.get(heading)
        if key is None:
            continue

        # Extract the markdown table (lines starting with |)
        table_lines = [
            line for line in body.splitlines() if line.strip().startswith("|")
        ]
        if not table_lines:
            continue

        # Derive display name by stripping trailing " Metrics"
        display_name = heading.removesuffix(" Metrics").strip()

        families[key] = {
            "name": display_name,
            "prefixes": "\n".join(table_lines),
        }

    return families


# ---------------------------------------------------------------------------
# Template rendering (safe with literal braces in JSON)
# ---------------------------------------------------------------------------

_LBRACE = "\0LBRACE\0"
_RBRACE = "\0RBRACE\0"


def _format_template(template: str, values: dict[str, str]) -> str:
    """Render a template with {NAME} placeholders, ignoring unmatched braces."""
    protected = template.replace("{{", _LBRACE).replace("}}", _RBRACE)

    def _replace(m: re.Match[str]) -> str:
        key = m.group(1)
        return str(values[key]) if key in values else m.group(0)

    rendered = re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", _replace, protected)
    return rendered.replace(_LBRACE, "{").replace(_RBRACE, "}")


# ---------------------------------------------------------------------------
# Claude invocation
# ---------------------------------------------------------------------------


# Model name mapping for Gradient AI / OpenAI-compatible endpoints.
# Keys are the short names accepted by --model; values are the model IDs
# expected by the remote API.
_API_MODEL_MAP: dict[str, str] = {
    "opus": "anthropic-claude-opus-4.6",
    "claude-opus-4-6": "anthropic-claude-opus-4.6",
    "sonnet": "anthropic-claude-4.5-sonnet",
    "claude-sonnet-4-5-20250929": "anthropic-claude-4.5-sonnet",
    "haiku": "anthropic-claude-haiku-4.5",
    "claude-haiku-4-5-20251001": "anthropic-claude-haiku-4.5",
}


def _parse_metrics_json(raw: str, family_key: str) -> list[dict] | None:
    """Extract a JSON metrics list from raw LLM output."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*\n(\[.*?\])\s*\n```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        logger.error(f"  [{family_key}] Could not parse JSON from output:\n{raw[:500]}")
        return None


def call_claude(
    prompt: str,
    model: str,
    effort: str,
    family_key: str,
    api_url: str | None = None,
    api_key: str | None = None,
) -> list[dict] | None:
    """Call Claude and parse the JSON metrics list from its output.

    When *api_url* and *api_key* are provided, uses the OpenAI-compatible API
    directly (e.g. DigitalOcean Gradient AI).  Otherwise falls back to the
    ``claude`` CLI.
    """
    if api_url and api_key:
        from openai import OpenAI

        api_model = _API_MODEL_MAP.get(model, model)
        logger.info(f"  [{family_key}] Calling API {api_url} model={api_model} ...")
        try:
            client = OpenAI(base_url=api_url, api_key=api_key)
            response = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"  [{family_key}] API call failed: {e}")
            return None
        return _parse_metrics_json(raw.strip(), family_key)

    # Fallback: use the claude CLI
    logger.info(
        f"  [{family_key}] Calling claude --model {model} --effort {effort} ..."
    )
    result = subprocess.run(
        ["claude", "--model", model, "--effort", effort, "-p", prompt],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(
            f"  [{family_key}] Claude failed (exit {result.returncode}): {result.stderr}"
        )
        return None
    return _parse_metrics_json(result.stdout.strip(), family_key)


# ---------------------------------------------------------------------------
# Grafana query + plotting
# ---------------------------------------------------------------------------

TYPE_COLORS = {
    "spiked": "red",
    "collapsed": "blue",
}


def query_metric(
    grafana: GrafanaApi,
    datasource_uid: str,
    expression: str,
    start_ts: int,
    end_ts: int,
    interval_ms: int,
) -> list[tuple[list[float], list[float], str]]:
    """Run a PromQL query and return a list of (timestamps, values, label) tuples."""
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


def plot_metrics(
    metrics: list[dict],
    results: list[list[tuple[list[float], list[float], str]]],
    incident_dt: datetime,
    output: Path | None,
) -> None:
    """Plot each metric's time series in a subplot grid."""
    n = len(metrics)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows), squeeze=False)

    for idx, (metric, series_list) in enumerate(zip(metrics, results)):
        ax = axes[idx // cols][idx % cols]
        color = TYPE_COLORS.get(metric.get("type", ""), "gray")

        if not series_list:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
        else:
            for timestamps_ms, values, label in series_list:
                dts = [
                    datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                    for ts in timestamps_ms
                ]
                ax.plot(dts, values, color=color, alpha=0.8, linewidth=1.2, label=label)

            if len(series_list) > 1:
                ax.legend(fontsize=12, loc="best")

        ax.axvline(
            incident_dt, color="black", linestyle="--", linewidth=1, label="incident"
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlabel("Time (UTC)", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)

        desc = metric.get("description", "")
        title = f"[{metric.get('type', '?')}] {desc}"
        title = "\n".join(textwrap.wrap(title, width=60))
        ax.set_title(title, fontsize=13)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        f"Metric Verification — incident at {incident_dt:%Y-%m-%d %H:%M:%S} UTC",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}", file=sys.stderr)
    else:
        plt.show()


def query_and_plot(
    metrics: list[dict],
    incident_dt: datetime,
    figure_path: Path,
    grafana_url: str,
    datasource_uid: str,
    start_ts: int,
    end_ts: int,
    interval_ms: int,
) -> None:
    """Query Grafana for each metric and save the plot."""
    logger.info(f"Querying {len(metrics)} metrics from Grafana...")
    grafana = GrafanaApi.from_url(grafana_url)

    results = []
    for i, metric in enumerate(metrics):
        query = metric.get("query", "")
        if not query:
            logger.info(
                f"  [{i + 1}/{len(metrics)}] Skipping (no query): "
                f"{metric.get('description', '')[:60]}"
            )
            results.append([])
            continue
        logger.info(f"  [{i + 1}/{len(metrics)}] {query[:80]}")
        try:
            series = query_metric(
                grafana, datasource_uid, query, start_ts, end_ts, interval_ms
            )
            results.append(series)
        except Exception as e:
            logger.warning(f"    Error: {e}")
            results.append([])

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_metrics(metrics, results, incident_dt, figure_path)
    logger.info(f"Saved figure to {figure_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    """Find metrics via 5 parallel family-scoped sub-agents."""
    # Parse families early so --families choices are derived from the file
    prefix_path = Path("prompts/metrics_prefix.md")
    metric_families = parse_prefix_families(prefix_path)
    family_keys = list(metric_families.keys())

    parser = get_base_parser()
    parser.description = (
        "Find metrics via 5 parallel family-scoped sub-agents "
        f"({', '.join(family_keys)})"
    )
    parser.add_argument(
        "--rubric-file",
        "-rf",
        type=Path,
        required=True,
        help="Path to rubric JSON file. feature_flag, incident_time, and "
        "symptoms are read directly from this file.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("prompts/metrics_family.md"),
        help="Path to prompt template (default: prompts/metrics_family.md)",
    )
    parser.add_argument(
        "--window",
        "-w",
        type=int,
        default=720,
        help="Minutes before/after incident to query (default: 720)",
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
        "--interval-ms",
        type=int,
        default=60000,
        help="Data point interval in milliseconds (default: 60000 = 1 minute)",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("CLAUDE_API_URL"),
        help="OpenAI-compatible API base URL (e.g. https://inference.do-ai.run). "
        "Falls back to CLAUDE_API_URL env var. When set with --api-key, "
        "uses the API directly instead of the claude CLI.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CLAUDE_API_KEY") or os.environ.get("MODEL_ACCESS_KEY"),
        help="API key for the endpoint specified by --api-url. "
        "Falls back to CLAUDE_API_KEY or MODEL_ACCESS_KEY env vars.",
    )
    parser.add_argument(
        "--skip-claude",
        action="store_true",
        help="Skip Claude step and load existing metrics JSON",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=family_keys,
        default=family_keys,
        help=f"Families to run (default: all). Choices: {', '.join(family_keys)}",
    )
    parser.add_argument(
        "--prefix-file",
        type=Path,
        default=prefix_path,
        help=f"Path to metrics_prefix.md (default: {prefix_path})",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    model = args.model
    window = args.window
    effort = args.effort
    families = args.families
    output_dir: Path = args.output_dir

    # Load rubric
    if not args.rubric_file.exists():
        logger.error(f"Rubric file not found: {args.rubric_file}")
        sys.exit(1)
    rubric = json.loads(args.rubric_file.read_text())
    flag = rubric["feature_flag"]
    rubric_id = rubric.get("id") or args.rubric_file.stem
    logger.info(f"Loaded rubric from {args.rubric_file} (id={rubric_id}, flag={flag})")

    # Resolve incident time from rubric
    if not rubric.get("incident_time"):
        logger.error(f"Rubric {args.rubric_file} has no incident_time field.")
        sys.exit(1)
    incident_dt = datetime.fromisoformat(rubric["incident_time"]).astimezone(
        timezone.utc
    )
    utc_time_str = incident_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.info(f"Incident time: {utc_time_str}")

    base_stem = f"metrics-{rubric_id}-{window}-{model}-{effort}"
    metrics_dir = output_dir / "metrics"
    families_dir = metrics_dir / f"{base_stem}-families"
    figures_dir = output_dir / "figures"
    incident_ts = int(incident_dt.timestamp())
    start_ts = incident_ts - window * 60
    end_ts = incident_ts + window * 60

    merged_metrics_path = metrics_dir / f"{base_stem}.json"
    merged_figure_path = figures_dir / f"{base_stem}.pdf"

    if args.skip_claude:
        # Load and merge existing per-family files
        all_metrics: list[dict] = []
        for fk in families:
            family_path = families_dir / f"{base_stem}-{fk}.json"
            if not family_path.exists():
                logger.warning(f"Family file not found: {family_path}")
                continue
            family_metrics = json.loads(family_path.read_text())
            logger.info(
                f"  [{fk}] Loaded {len(family_metrics)} metrics from {family_path}"
            )
            all_metrics.extend(family_metrics)

        if not all_metrics:
            logger.error("No metrics loaded from any family file.")
            return

        merged_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        merged_metrics_path.write_text(json.dumps(all_metrics, indent=2) + "\n")
        logger.info(f"Merged {len(all_metrics)} metrics to {merged_metrics_path}")

        query_and_plot(
            all_metrics,
            incident_dt,
            merged_figure_path,
            args.grafana_url,
            args.datasource_uid,
            start_ts,
            end_ts,
            args.interval_ms,
        )
        return

    # Build prompt template
    if not args.template.exists():
        logger.error(f"Template not found: {args.template}")
        sys.exit(1)
    template = args.template.read_text()
    gs = rubric.get("golden_signals") or rubric.get("symptoms", {})

    # Keys that are document references, not useful context for metric finding.
    _SKIP_KEYS = {"log_documents", "trace_documents"}

    def _format_symptoms(entries: list) -> str:
        """Format a list of symptom entries (strings or dicts) as markdown."""
        parts = []
        for e in entries:
            if isinstance(e, str):
                parts.append(e)
                continue
            lines = [f"- **{e.get('name', '')}**: {e.get('description', '')}"]
            for k, v in e.items():
                if k in ("name", "description") or k in _SKIP_KEYS:
                    continue
                if isinstance(v, dict):
                    v = ", ".join(f"{dk}={dv}" for dk, dv in v.items())
                elif isinstance(v, list):
                    v = "; ".join(str(item) for item in v)
                lines.append(f"  - {k}: {v}")
            parts.append("\n".join(lines))
        return "\n".join(parts)

    # Common template values (shared across all families)
    common_values = {
        "FEATURE_FLAG": rubric["feature_flag"],
        "UTC_TIME": utc_time_str,
        "MECHANISM": ""
        if rubric.get("mechanism") in (None, "TODO")
        else rubric["mechanism"],
        "LOGS": _format_symptoms(gs.get("logs", [])),
        "SPANS": _format_symptoms(gs.get("spans") or gs.get("traces", [])),
        "FRONTEND": _format_symptoms(gs.get("frontend", [])),
    }

    # Re-parse prefix file if user overrode it via --prefix-file
    if args.prefix_file != prefix_path:
        metric_families = parse_prefix_families(args.prefix_file)

    # Build per-family prompts
    family_prompts: dict[str, str] = {}
    for fk in families:
        fam = metric_families[fk]
        values = {
            **common_values,
            "FAMILY_NAME": fam["name"],
            "FAMILY_PREFIXES": fam["prefixes"],
        }
        family_prompts[fk] = _format_template(template, values)

    # Launch family calls in parallel
    logger.info(f"Launching {len(families)} family sub-agent(s) in parallel...")
    families_dir.mkdir(parents=True, exist_ok=True)
    family_results: dict[str, list[dict] | None] = {}

    api_url = args.api_url
    api_key = args.api_key
    if api_url and api_key:
        logger.info(f"Using API endpoint: {api_url}")
    elif api_url and not api_key:
        logger.warning("--api-url set but no --api-key provided; falling back to CLI.")
        api_url = None

    with ThreadPoolExecutor(max_workers=len(families)) as executor:
        futures = {
            executor.submit(
                call_claude, family_prompts[fk], model, effort, fk, api_url, api_key
            ): fk
            for fk in families
        }
        for future in as_completed(futures):
            fk = futures[future]
            try:
                family_results[fk] = future.result()
            except Exception as e:
                logger.error(f"  [{fk}] Exception: {e}")
                family_results[fk] = None

    # Save per-family results
    for fk in families:
        family_path = families_dir / f"{base_stem}-{fk}.json"
        metrics = family_results.get(fk)

        if metrics is None:
            logger.error(f"  [{fk}] No metrics returned, skipping.")
            continue

        family_path.write_text(json.dumps(metrics, indent=2) + "\n")
        logger.info(f"  [{fk}] Saved {len(metrics)} metrics to {family_path}")

    # Merge all family files from disk (includes untouched families from prior runs)
    all_metrics: list[dict] = []
    for family_path in sorted(families_dir.glob(f"{base_stem}-*.json")):
        family_metrics = json.loads(family_path.read_text())
        logger.info(
            f"  [merge] Loaded {len(family_metrics)} metrics from {family_path.name}"
        )
        all_metrics.extend(family_metrics)

    if not all_metrics:
        logger.error("No metrics found in any family file. Nothing to merge or plot.")
        return

    # Save merged result
    metrics_dir.mkdir(parents=True, exist_ok=True)
    merged_metrics_path.write_text(json.dumps(all_metrics, indent=2) + "\n")
    logger.info(f"Merged {len(all_metrics)} metrics to {merged_metrics_path}")

    # Plot merged metrics
    query_and_plot(
        all_metrics,
        incident_dt,
        merged_figure_path,
        args.grafana_url,
        args.datasource_uid,
        start_ts,
        end_ts,
        args.interval_ms,
    )


if __name__ == "__main__":
    main()
