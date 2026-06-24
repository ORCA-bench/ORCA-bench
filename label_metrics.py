#!/usr/bin/env python3
r"""Group onset metrics into families and annotate signal layers.

Groups metrics by exact (base_metric_name, label_set) equality after
stripping standard Prometheus histogram/summary suffixes (_count, _sum,
_total, etc.).  Each family is then annotated with a signal layer
(root_cause, propagation, symptom, meta) via Claude CLI using the
incident mechanism from a rubric file.

Exact label equality (rather than subset clustering) ensures that all
members of a family share the same causal position in the incident chain,
so the signal layer annotation applies uniformly to every variant.

Input::

    {metrics_dir}/metrics-{flag}-*.json   (per-run onset files)
    rubrics_kc/{flag_lower}.md            (for mechanism section)

Output::

    {output_dir}/labeled_metrics/metrics-{flag}-labeled.json

Examples::

    # Label metrics for productCatalogFailure:
    uv run python label_metrics.py -f productCatalogFailure -od metrics-0401

    # Custom input directory:
    uv run python label_metrics.py -f productCatalogFailure -od metrics-0401 \
        --metrics-dir metrics-0401/metrics_onset

    # Dry run (no LLM, signal_layer set to "unknown"):
    uv run python label_metrics.py -f productCatalogFailure -od metrics-0401 --skip-llm
"""

import json
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "opus"
_DEFAULT_EFFORT = "max"

# Prometheus histogram/summary suffixes that indicate variants of the same
# underlying metric family.
_METRIC_SUFFIXES = (
    "_count",
    "_sum",
    "_bucket",
    "_total",
    "_created",
    "_avg",
    "_max",
    "_min",
)

# Matches label selectors: {key="val", ...}
_LABELS_RE = re.compile(r"\{([^}]*)\}")

# Matches individual label pairs: key="val" or key=~"val"
_LABEL_PAIR_RE = re.compile(r'(\w+)\s*[=!~]+\s*"([^"]*)"')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_metric_suffix(name: str) -> str:
    """Strip standard Prometheus suffixes to get the base metric family name.

    E.g. ``rpc_client_call_duration_seconds_count`` ->
    ``rpc_client_call_duration_seconds``.
    """
    for suffix in _METRIC_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _extract_metric_name(query: str) -> str:
    """Extract the raw metric name from a PromQL query.

    Handles wrapped queries like ``rate(metric{labels}[5m])`` and bare
    metrics like ``metric{labels}``.
    """
    inner = query
    while True:
        m = re.match(r"\w+\((.+)\)", inner, re.DOTALL)
        if m:
            inner = m.group(1)
        else:
            break
    inner = re.sub(r"\[\d+[smhd](?:\d+[smhd])*\]", "", inner)
    inner = re.sub(r"\{[^}]*\}", "", inner)
    return inner.strip()


def _extract_labels(query: str) -> dict[str, str]:
    """Extract label key-value pairs from a PromQL query."""
    label_match = _LABELS_RE.search(query)
    if not label_match:
        return {}
    return dict(_LABEL_PAIR_RE.findall(label_match.group(1)))


# ---------------------------------------------------------------------------
# Step 1: Group into metric families (exact label equality)
# ---------------------------------------------------------------------------


def group_into_families(entries: list[dict]) -> list[dict]:
    """Group entries into metric families by (base_name, exact_labels).

    Entries are grouped when they share the same base metric name (after
    suffix stripping) AND have identical label sets.  No subset logic —
    labels must match exactly.

    Returns a list of family dicts, each with:
    - metric_family: base metric name
    - signal_layer: "unknown" (populated later by annotation step)
    - defining_labels: the shared label dict
    - variants: list of variant dicts
    """
    groups: dict[tuple[str, tuple[tuple[str, str], ...]], list[dict]] = {}

    for entry in entries:
        metric_name = _extract_metric_name(entry["query"])
        base_name = _strip_metric_suffix(metric_name)
        labels = _extract_labels(entry["query"])
        key = (base_name, tuple(sorted(labels.items())))
        groups.setdefault(key, []).append(entry)

    families = []
    for (base_name, label_pairs), group_entries in groups.items():
        variants = []
        for entry in group_entries:
            variants.append(
                {
                    "type": entry["type"],
                    "query": entry["query"],
                    "description": entry["description"],
                    "onset_utc": entry.get("onset_utc"),
                    "onset_offset_seconds": entry.get("onset_offset_seconds"),
                    "onset_reasoning": entry.get("onset_reasoning"),
                }
            )

        families.append(
            {
                "metric_family": base_name,
                "signal_layer": "unknown",
                "defining_labels": dict(label_pairs),
                "variants": variants,
            }
        )

    return families


# ---------------------------------------------------------------------------
# Step 2: Annotate signal layers via Claude
# ---------------------------------------------------------------------------


_ANNOTATION_PROMPT = """\
You are an expert SRE. Given the mechanism of an incident and a list of \
metric families observed during the incident, classify each metric family \
into one of these signal layers:

- **root_cause**: The metric closest to the fault injection point. This is \
the first service/component where the error originates.
- **propagation**: Intermediate services that reflect the error as it travels \
through the call chain (e.g. checkout returning Internal because payment failed).
- **symptom**: User-facing impact metrics (e.g. HTTP 500s on the frontend, \
load generator seeing errors).
- **meta**: Feature flag counters, internal plumbing, or metrics that confirm \
the flag was toggled but don't represent the actual error.

## Incident Mechanism

{mechanism}

## Metric Families

{families_description}

For each metric family, respond with ONLY valid JSON — a list of objects:
[
  {{"family": "metric_family_name", "labels": {{...}}, "signal_layer": "root_cause|propagation|symptom|meta", "reasoning": "brief explanation"}},
  ...
]
"""


def _extract_mechanism(rubric_path: Path) -> str | None:
    """Extract the Mechanism section from a rubric markdown file."""
    if not rubric_path.exists():
        logger.warning(f"Rubric not found: {rubric_path}")
        return None

    text = rubric_path.read_text()
    match = re.search(r"## Mechanism\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if not match:
        logger.warning(f"No '## Mechanism' section found in {rubric_path}")
        return None

    return match.group(1).strip()


def _extract_mechanism_from_dir(rubrics_dir: Path, feature_flag: str) -> str | None:
    """Extract the mechanism from a rubrics directory.

    Supports two formats:
    - JSON task files (with ``feature_flag`` and ``mechanism`` keys)
    - Markdown rubric files (with ``## Mechanism`` section, named ``{flag_lower}.md``)
    """
    # Try JSON task files first
    for json_path in rubrics_dir.glob("*.json"):
        try:
            data = json.loads(json_path.read_text())
            if data.get("feature_flag") == feature_flag and "mechanism" in data:
                logger.info(f"Read mechanism from {json_path}")
                return data["mechanism"]
        except (json.JSONDecodeError, AttributeError):
            continue

    # Fall back to markdown rubric
    md_path = rubrics_dir / f"{feature_flag.lower()}.md"
    return _extract_mechanism(md_path)


def annotate_signal_layers(
    families: list[dict],
    mechanism: str,
    model: str = _DEFAULT_MODEL,
    effort: str = _DEFAULT_EFFORT,
) -> list[dict]:
    """Use Claude CLI to annotate each family with a signal_layer."""
    families_desc_lines = []
    for i, fam in enumerate(families):
        labels_str = (
            json.dumps(fam["defining_labels"])
            if fam["defining_labels"]
            else "(no labels)"
        )
        variant_types = ", ".join(
            f"{v['type']} ({v['query'][:80]}...)"
            if len(v["query"]) > 80
            else f"{v['type']} ({v['query']})"
            for v in fam["variants"]
        )
        families_desc_lines.append(
            f"{i + 1}. **{fam['metric_family']}** {labels_str}\n"
            f"   Variants: {variant_types}"
        )

    prompt = _ANNOTATION_PROMPT.format(
        mechanism=mechanism,
        families_description="\n".join(families_desc_lines),
    )

    logger.info(f"Asking Claude to annotate {len(families)} metric families...")

    result = subprocess.run(
        ["claude", "--model", model, "--effort", effort, "-p", prompt],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.warning(f"Claude CLI failed: {result.stderr}")
        return families

    raw = result.stdout.strip()

    # Parse JSON from response (handle markdown code fences)
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        annotations = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse Claude response as JSON:\n{raw[:500]}")
        return families

    # Build lookup by (family_name, labels)
    annotation_map: dict[tuple[str, str], str] = {}
    for ann in annotations:
        key = (ann["family"], json.dumps(ann.get("labels", {}), sort_keys=True))
        annotation_map[key] = ann["signal_layer"]
        reasoning = ann.get("reasoning", "")
        if reasoning:
            logger.info(f"  {ann['family']}: {ann['signal_layer']} — {reasoning}")

    for fam in families:
        key = (
            fam["metric_family"],
            json.dumps(fam["defining_labels"], sort_keys=True),
        )
        if key in annotation_map:
            fam["signal_layer"] = annotation_map[key]
        else:
            # Fallback: match by family name only
            for ann_key, layer in annotation_map.items():
                if ann_key[0] == fam["metric_family"]:
                    fam["signal_layer"] = layer
                    break
            else:
                logger.warning(
                    f"No annotation found for {fam['metric_family']}, keeping 'unknown'"
                )

    return families


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

_LAYER_DESCRIPTIONS = {
    "root_cause": "metric closest to the fault injection point",
    "propagation": "intermediate services reflecting the error through the call chain",
    "symptom": "user-facing impact visible to end users or clients",
    "meta": "feature flag counters or internal plumbing",
    "unknown": "unclassified",
}


def _format_onset(offset: int | None) -> str:
    """Format an onset offset as '+Xs' or 'N/A'."""
    if offset is None:
        return "N/A"
    return f"+{offset}s"


def _format_label_values(labels: dict[str, str]) -> str:
    """Format label values as a compact comma-separated string."""
    if not labels:
        return ""
    return ", ".join(labels.values())


def _earliest_onset(variants: list[dict]) -> int | None:
    """Return the earliest onset_offset_seconds across variants, or None."""
    offsets = [
        v["onset_offset_seconds"]
        for v in variants
        if v.get("onset_offset_seconds") is not None
    ]
    return min(offsets) if offsets else None


def format_families_markdown(families: list[dict]) -> str:
    """Render labeled families as markdown grouped by signal layer."""
    # Group by signal_layer
    by_layer: dict[str, list[dict]] = {}
    for fam in families:
        by_layer.setdefault(fam["signal_layer"], []).append(fam)

    layer_order = ["root_cause", "propagation", "symptom", "meta", "unknown"]
    lines = ["# Metrics Families", ""]

    for layer in layer_order:
        layer_families = by_layer.get(layer, [])
        if not layer_families:
            continue

        # Earliest onset across all families in this layer
        layer_onsets = [_earliest_onset(f["variants"]) for f in layer_families]
        layer_onsets = [o for o in layer_onsets if o is not None]
        layer_onset_str = (
            f" (earliest onset: {_format_onset(min(layer_onsets))})"
            if layer_onsets
            else ""
        )

        display_layer = layer.replace("_", " ").title()
        desc = _LAYER_DESCRIPTIONS.get(layer, "")
        lines.append(f"## {display_layer} — {desc}{layer_onset_str}")
        lines.append("")

        for fam in layer_families:
            fam_onset = _earliest_onset(fam["variants"])
            label_vals = _format_label_values(fam["defining_labels"])
            label_part = f" ({label_vals})" if label_vals else ""

            lines.append(
                f"### {fam['metric_family']} family{label_part}"
                f" — onset: {_format_onset(fam_onset)}"
            )
            lines.append("")

            # Use first variant's description as the family description
            first_desc = fam["variants"][0].get("description", "")
            if first_desc:
                lines.append(first_desc)
                lines.append("")

            for v in fam["variants"]:
                v_onset = _format_onset(v.get("onset_offset_seconds"))
                lines.append(f"- **{v['type']}** (onset: {v_onset})")
                lines.append(f"  `{v['query']}`")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def label_metrics(
    feature_flag: str,
    metrics_dir: Path,
    rubrics_dir: Path,
    output_dir: Path,
    model: str = _DEFAULT_MODEL,
    effort: str = _DEFAULT_EFFORT,
    skip_llm: bool = False,
) -> Path:
    """Run the grouping and annotation pipeline for one feature flag."""
    # Discover input files
    prefix = f"metrics-{feature_flag}-"
    candidates = sorted(
        f
        for f in metrics_dir.glob(f"{prefix}*.json")
        if f.name[len(prefix) :].split("-", 1)[0].isdigit()
    )
    if not candidates:
        logger.error(
            f"No metric files found for exact flag '{feature_flag}' in {metrics_dir}"
        )
        sys.exit(1)

    input_path = candidates[0]
    logger.info(f"Loading metrics from {input_path}")

    entries = json.loads(input_path.read_text())
    logger.info(f"Loaded {len(entries)} entries")

    # Filter out inconclusive metrics
    entries = [e for e in entries if "inconclusive" not in e.get("type", "").lower()]
    logger.info(f"After filtering inconclusive: {len(entries)} entries")

    if not entries:
        logger.warning(
            f"No actionable metrics for '{feature_flag}' after filtering — skipping."
        )
        return output_dir / f"metrics-{feature_flag}-labeled.json"

    # Step 1: Group into families
    families = group_into_families(entries)
    logger.info(f"Step 1: Grouped into {len(families)} metric families")

    for fam in families:
        labels_str = (
            json.dumps(fam["defining_labels"])
            if fam["defining_labels"]
            else "(no labels)"
        )
        n_variants = len(fam["variants"])
        logger.info(f"  {fam['metric_family']} {labels_str} — {n_variants} variant(s)")

    # Step 2: Annotate signal layers
    if not skip_llm:
        mechanism = _extract_mechanism_from_dir(rubrics_dir, feature_flag)
        if mechanism:
            families = annotate_signal_layers(families, mechanism, model, effort)
        else:
            logger.warning("No mechanism found; skipping signal layer annotation")

    # Sort by layer order
    layer_order = {
        "root_cause": 0,
        "propagation": 1,
        "symptom": 2,
        "meta": 3,
        "unknown": 99,
    }
    families.sort(
        key=lambda f: (layer_order.get(f["signal_layer"], 99), f["metric_family"])
    )

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"metrics-{feature_flag}-labeled.json"
    output_path.write_text(json.dumps(families, indent=2) + "\n")
    logger.info(f"Saved to {output_path}")

    md_text = format_families_markdown(families)
    md_path = output_dir / f"metrics-{feature_flag}-labeled.md"
    md_path.write_text(md_text + "\n")
    logger.info(f"Saved markdown to {md_path}")

    return output_path


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
    """Group onset metrics into families and annotate signal layers."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Group onset metrics into families and annotate signal layers."
    )
    parser.add_argument(
        "--feature-flag",
        "-f",
        required=True,
        help="Feature flag name (e.g. productCatalogFailure)",
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        type=Path,
        default=Path("out"),
        help="Output directory (default: out)",
    )
    parser.add_argument(
        "--metrics-dir",
        "-dir",
        type=Path,
        default=None,
        help="Input directory containing onset JSON files (default: {output_dir}/metrics_onset)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=_DEFAULT_MODEL,
        help=f"Claude model for annotation (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--effort",
        "-e",
        type=str,
        default=_DEFAULT_EFFORT,
        help=f"Claude effort level (default: {_DEFAULT_EFFORT})",
    )
    parser.add_argument(
        "--rubrics-dir",
        "-rd",
        type=Path,
        default=Path("rubrics_kc"),
        help="Directory containing rubric files — JSON task files (with 'mechanism' key) or markdown (with '## Mechanism' section). Default: rubrics_kc",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip Claude annotation — signal_layer set to 'unknown'.",
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
    metrics_dir = args.metrics_dir or (output_dir / "metrics_onset")

    label_metrics(
        feature_flag=args.feature_flag,
        metrics_dir=metrics_dir,
        rubrics_dir=args.rubrics_dir,
        output_dir=output_dir / "labeled_metrics",
        model=args.model,
        effort=args.effort,
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()
