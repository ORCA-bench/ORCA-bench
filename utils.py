import json
import logging
import re
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import sem

from harbor_utils.plotting_utils import get_display_label

logger = logging.getLogger(__name__)


def format_group_label(agent: str, model: str, effort: str | None) -> str:
    """Return an ``agent (model) (effort)`` label, omitting effort if unset."""
    base = get_display_label(agent, model)
    return f"{base} ({effort})" if effort else base


# Default paths
DEFAULT_FLAGD_PATH = Path("opentelemetry-demo/src/flagd/demo.flagd.json")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with a consistent format across scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def get_base_parser() -> ArgumentParser:
    """Get the base parser for all scripts."""
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="openai-gpt-5.4")
    parser.add_argument(
        "--output-dir",
        "-od",
        type=Path,
        default=Path("out"),
        help="Path to output directory",
    )
    parser.add_argument(
        "--data-dir",
        "-dd",
        type=Path,
        default=Path("data"),
        help="Path to telemetry data",
    )
    parser.add_argument(
        "--jobs-dir",
        "-jd",
        type=Path,
        default=None,
        help=(
            "Harbor job directory holding one subdirectory per trial. Scores are "
            "read from each trial's verifier/reward-details.json."
        ),
    )
    parser.add_argument(
        "--schedule",
        "-s",
        type=Path,
        default=Path("schedules/incident_schedule.json"),
        help="Path to the schedule JSON file",
    )
    parser.add_argument(
        "--log_level",
        "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--filter-xlsx",
        type=Path,
        default=None,
        help="Excel file with is_valid_issue column to filter invalid trials.",
    )
    parser.add_argument(
        "--effort",
        "-e",
        type=str,
        choices=["low", "medium", "high", "max"],
        default=None,
        help="Reasoning effort level for LLM calls / trial filtering.",
    )
    parser.add_argument(
        "--feature-flag",
        "-f",
        type=str,
        help="Feature flag name to look up",
    )
    return parser


def parse_snapshot_time(name: str) -> datetime:
    """Parse the ISO 8601 timestamp prefix from a snapshot name.

    Example: '20260210T190036Z-324b21f5fffc57d1' -> datetime(2026, 2, 10, 19, 0, 36, tzinfo=UTC)
    """
    ts_str = name.split("-")[0]  # '20260210T190036Z'
    return datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


# Column label → JSON field mapping. Values are the rollup keys produced by
# check_prediction.aggregate_judge_response() and materialized on the
# DataFrame at load time by load_data.
Q_COLS: dict[str, str] = {
    "RCA Accuracy": "rca_accuracy",
    "% Complete Hallucination": "hallucinate_any",
    "Mechanism": "mechanism_match",
    "Inc. Time": "incident_time_within_10min",
    "Metrics (all)": "metrics_all_match",
    "Metrics (any)": "metrics_any_match",
    "Logs (all)": "logs_all_match",
    "Logs (any)": "logs_any_match",
    "Traces (all)": "traces_all_match",
    "Traces (any)": "traces_any_match",
}

ROLLUP_KEYS: list[str] = [
    "incident_time_within_10min",
    "hallucinate_any",
    "rca_accuracy",
    "mechanism_match",
    "metrics_all_match",
    "metrics_any_match",
    "logs_all_match",
    "logs_any_match",
    "traces_all_match",
    "traces_any_match",
    "rca_depth",
]


def model_display(model: str) -> str:
    """Strip common provider prefixes for display."""
    for prefix in (
        "openai-",
        "gradient_ai/openai-",
        "gradient_ai/anthropic-",
        "gradient_ai/",
    ):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


# Canonical left-to-right order for ``_agent_model``: closed-source frontier
# models first (Opus, Sonnet, GPT), then open-weight (GLM, DeepSeek). Matched
# as substrings against the display name produced by :func:`model_display`,
# so e.g. ``claude-4.6-sonnet`` matches ``sonnet``.
MODEL_ORDER: tuple[str, ...] = ("opus", "sonnet", "gpt", "glm", "deepseek")


def model_sort_key(agent_model_display: str) -> tuple[int, str]:
    """Sort key for ``_agent_model`` values following :data:`MODEL_ORDER`.

    Unknown models trail in alphabetical order.
    """
    name = agent_model_display.lower()
    for i, token in enumerate(MODEL_ORDER):
        if token in name:
            return (i, name)
    return (len(MODEL_ORDER), name)


# Compact display aliases for plot tick labels. Keys are the ``_agent_model``
# values produced by :func:`model_display` (after provider-prefix stripping).
MODEL_ALIASES: dict[str, str] = {
    "claude-4.6-sonnet": "Sonnet 4.6",
    "claude-opus-4.7": "Opus 4.7",
    "gpt-5.5": "GPT-5.5",
    "glm-5": "GLM-5",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
}


def model_alias(agent_model_display: str) -> str:
    """Return :data:`MODEL_ALIASES` for ``agent_model_display`` if known.

    Falls through to the raw display name for unmapped models.
    """
    return MODEL_ALIASES.get(agent_model_display, agent_model_display)


# Granularity: generation now emits the final user-facing ladder
# {easy, medium, hard} directly (easy=detailed, medium=broad feature area,
# hard=flag-agnostic), so this map is a passthrough. Kept as an explicit
# identity for clarity and for normalizing legacy data via ``.get(g, g)``.
GRANULARITY_ALIASES: dict[str, str] = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}


def find_trial_dirs(root: Path) -> list[Path]:
    """Trial directories under ``root``.

    Accepts either a single job directory (whose children are trial dirs) or a
    tree of job directories (grandchildren are trial dirs). A trial dir is
    identified by having a ``verifier/`` subdirectory.
    """
    direct = [p for p in root.iterdir() if p.is_dir() and (p / "verifier").is_dir()]
    if direct:
        return sorted(direct)
    return sorted(
        trial
        for job in root.iterdir()
        if job.is_dir()
        for trial in job.iterdir()
        if trial.is_dir() and (trial / "verifier").is_dir()
    )


def load_trials(jobs_dir: Path, reasoning_effort: str | None) -> list[dict]:
    """Load all trial entries from ``<trial>/verifier/reward-details.json``.

    ``jobs_dir`` is either a single Harbor job directory or a tree of them (see
    :func:`find_trial_dirs`). Trials with no ``reward-details.json`` (never
    scored) are skipped.

    Each returned dict has ``_display_model``, ``_trial_id``, ``task_name``,
    ``_score_path`` and ``_report_path`` injected. Aggregated rollups from
    ``aggregate_judge_response`` (including ``rca_depth`` and every key in
    :data:`ROLLUP_KEYS`) are also spread onto each trial dict so non-DataFrame
    callers can read them directly without going through ``load_data``.

    The trial directory name doubles as ``_trial_id`` and ``task_name``: the
    readable task id is not recoverable from a trial dir (``result.json``
    carries only the hashed package name).
    """
    from check_prediction import aggregate_judge_response

    trials: list[dict] = []
    for trial_dir in find_trial_dirs(jobs_dir):
        details_path = trial_dir / "verifier" / "reward-details.json"
        if not details_path.is_file():
            continue
        try:
            trial = json.loads(details_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Skipping {trial_dir.name}: {exc}")
            continue
        re_val = trial.get("reasoning_effort")
        if reasoning_effort is not None and re_val != reasoning_effort:
            continue
        nested = trial.get("nested") or {}
        if not nested.get("rubrics"):
            continue
        trial.update(aggregate_judge_response(nested, mode=trial.get("mode")))
        trial["_display_model"] = model_display(trial.get("model", ""))
        trial["_trial_id"] = trial_dir.name
        trial["task_name"] = trial_dir.name
        trial["_trial_dir"] = str(trial_dir.resolve())
        trial["_score_path"] = str(details_path.resolve())
        trial["_report_path"] = str((trial_dir / "verifier" / "report.md").resolve())
        trials.append(trial)
    return trials


def load_agent_config(trial_dir: Path) -> dict:
    """Return the ``agent`` block from a trial's ``config.json`` (``{}`` if absent)."""
    config_path = Path(trial_dir) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text()).get("agent", {}) or {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not read {config_path}: {exc}")
        return {}


def load_task_metadata(trial_dirs: list[Path]) -> dict[str, dict]:
    """Fetch ``task.toml`` ``[metadata]`` from the registry for each trial.

    Reads ``config.json``'s ``task.name`` / ``task.ref`` per trial, downloads
    the deduplicated set in one batch, and returns ``{trial_dir_name: metadata}``.

    Harbor caches downloads under ``TASK_CACHE_DIR`` keyed by digest
    (``<cache>/<org>/<name>/<digest>/``), so repeat runs hit no network. The
    dataset is private, so this needs an authenticated harbor session
    (``harbor auth login``) or ``HARBOR_API_KEY``.
    """
    import asyncio

    from harbor.models.task.id import PackageTaskId
    from harbor.tasks.client import TaskClient
    from harbor_utils.export_predictions import _parse_task_metadata

    # trial name -> (org/name, ref); dedupe the task ids we actually fetch.
    per_trial: dict[str, tuple[str, str]] = {}
    for trial_dir in trial_dirs:
        config_path = Path(trial_dir) / "config.json"
        if not config_path.is_file():
            continue
        task = json.loads(config_path.read_text()).get("task") or {}
        name, ref = task.get("name"), task.get("ref")
        if name and ref:
            per_trial[Path(trial_dir).name] = (name, ref)

    unique = sorted(set(per_trial.values()))
    if not unique:
        return {}
    logger.info(f"Fetching task.toml metadata for {len(unique)} task(s)...")

    task_ids = {
        (name, ref): PackageTaskId(
            org=name.split("/")[0], name=name.split("/")[1], ref=ref
        )
        for name, ref in unique
    }
    asyncio.run(TaskClient().download_tasks(list(task_ids.values()), export=False))

    # ``export=False`` caches at PACKAGE_CACHE_DIR/<org>/<name>/<digest>/, which
    # get_local_path() computes directly — no need to rely on result ordering.
    meta_by_task: dict[tuple[str, str], dict] = {}
    for key, task_id in task_ids.items():
        toml_path = task_id.get_local_path() / "task.toml"
        if not toml_path.is_file():
            logger.warning(f"No task.toml at {toml_path}")
            continue
        meta_by_task[key] = _parse_task_metadata(toml_path.read_text())

    return {
        trial: meta_by_task[key]
        for trial, key in per_trial.items()
        if key in meta_by_task
    }


def fmt_acc(values: list[bool | None]) -> str:
    """Format accuracy as mean ± SE percentage, ignoring None values."""
    valid = np.array([int(v) for v in values if v is not None])
    if len(valid) == 0:
        return "—"
    mean = np.mean(valid) * 100
    if len(valid) == 1:
        return f"{mean:.1f}%"
    se = sem(valid) * 100
    return f"{mean:.1f} ± {se:.1f}%"


def fmt_mean_score(scores: list[int]) -> str:
    """Format mean score as mean ± SE."""
    if not scores:
        return "—"
    arr = np.array(scores)
    if len(arr) == 1:
        return f"{np.mean(arr):.2f}"
    return f"{np.mean(arr):.2f} ± {sem(arr):.2f}"


def fmt_mean_score_pct(scores: list[float]) -> str:
    """Format mean score as percent of max (3) ± SE."""
    if not scores:
        return "—"
    arr = np.array(scores) / 3.0 * 100
    if len(arr) == 1:
        return f"{np.mean(arr):.1f}%"
    return f"{np.mean(arr):.1f} ± {sem(arr):.1f}%"


def load_invalid_trial_ids(xlsx_path: Path) -> set[str]:
    """Load trial IDs marked as invalid from an Excel file.

    Reads the ``all-predictions`` sheet and returns the set of ``trial_id``
    values where ``is_valid_issue == "n"``.
    """
    df = pd.read_excel(xlsx_path, sheet_name="all-predictions")
    return set(df.loc[df["is_valid_issue"] == "n", "trial_id"])


def load_data(
    jobs_dir: Path,
    reasoning_effort: str | None = None,
    filter_xlsx: Path | None = None,
) -> pd.DataFrame:
    """Load judge trials from a Harbor jobs directory.

    ``jobs_dir`` holds one subdirectory per trial. Scores come from each
    trial's ``verifier/reward-details.json``, the agent identity from its
    ``config.json``, and the task dimensions from the published task's
    ``task.toml`` ``[metadata]`` fetched from the registry.

    Returns a DataFrame with one row per scored trial. Each row carries the
    raw fields from ``reward-details.json`` plus the following derived columns:

    - ``_trial_id``, ``task_name``, ``_display_model``, ``_score_path``,
      ``_report_path`` (from :func:`load_trials`)
    - ``_agent``, ``_agent_model`` (display name), ``_effort``
    - ``_section``, ``_ttd_minutes``, ``_phrasing``, ``_granularity``, ``flag``
    - ``_detected`` (100 iff the agent wrote a report exactly when the task had
      an incident)
    - ``_is_control`` (True iff the task's ``events`` is empty — no-incident task)
    - ``_n_events`` (number of plausible incidents)
    """
    if not jobs_dir.is_dir():
        logger.error(f"Jobs directory not found: {jobs_dir}")
        sys.exit(1)

    trials = load_trials(jobs_dir, reasoning_effort)
    if filter_xlsx is not None:
        invalid_ids = load_invalid_trial_ids(filter_xlsx)
        trials = [t for t in trials if t["_trial_id"] not in invalid_ids]
    if not trials:
        logger.error(f"No scored trials found under {jobs_dir}.")
        sys.exit(1)

    df = pd.DataFrame(trials)

    # -- Agent identity from each trial's config.json --
    agent_cfgs = {t["_trial_id"]: load_agent_config(t["_trial_dir"]) for t in trials}
    df["_agent"] = df["_trial_id"].map(
        lambda t: agent_cfgs[t].get("name", "unknown") or "unknown"
    )
    df["_agent_model"] = df["_trial_id"].map(
        lambda t: model_display(agent_cfgs[t].get("model_name", "unknown"))
    )
    # Make ``_agent_model`` an ordered Categorical so that downstream
    # ``sort_values`` and ``groupby`` consistently follow MODEL_ORDER without
    # each caller having to re-sort.
    ordered_models = sorted(df["_agent_model"].dropna().unique(), key=model_sort_key)
    df["_agent_model"] = pd.Categorical(
        df["_agent_model"], categories=ordered_models, ordered=True
    )
    df["_effort"] = df["_trial_id"].map(
        lambda t: (agent_cfgs[t].get("kwargs") or {}).get("reasoning_effort") or ""
    )

    # -- Task dimensions from the registry's task.toml [metadata] --
    meta = load_task_metadata([t["_trial_dir"] for t in trials])
    missing = [t for t in df["_trial_id"] if t not in meta]
    if missing:
        logger.warning(
            f"No task metadata for {len(missing)} trial(s), e.g. {missing[:3]}"
        )

    def _meta(trial_id: str) -> dict:
        return meta.get(trial_id, {})

    df["_section"] = df["_trial_id"].map(lambda t: _meta(t).get("section"))
    df["_ttd_minutes"] = df["_trial_id"].map(lambda t: _meta(t).get("ttd_minutes"))
    df["_phrasing"] = df["_trial_id"].map(lambda t: _meta(t).get("phrasing"))
    df["_granularity"] = df["_trial_id"].map(
        lambda t: GRANULARITY_ALIASES.get(
            _meta(t).get("granularity"), _meta(t).get("granularity")
        )
    )
    df["flag"] = df["_trial_id"].map(lambda t: _meta(t).get("flag"))
    df["_is_control"] = df["_trial_id"].map(lambda t: not bool(_meta(t).get("events")))
    df["_n_events"] = df["_trial_id"].map(lambda t: len(_meta(t).get("events") or []))

    # Detection accuracy: agent wrote a report iff the task had an incident.
    def _wrote_report(path: str) -> bool:
        report = Path(path)
        return report.is_file() and bool(report.read_text().strip())

    df["_detected"] = 100 * (
        df["_report_path"].map(_wrote_report) == ~df["_is_control"]
    ).astype(int)

    # Derive per-section rollups from the stored nested judge output. Local
    # import to avoid a circular dep at module load (check_prediction doesn't
    # import utils, but utils.load_data needs check_prediction's aggregator).
    from check_prediction import aggregate_judge_response

    def _row_rollup(row: pd.Series) -> dict[str, bool | None]:
        nested = row.get("nested")
        if not isinstance(nested, dict):
            return {k: None for k in ROLLUP_KEYS}
        agg = aggregate_judge_response(nested, mode=row.get("mode"))
        return {k: agg.get(k) for k in ROLLUP_KEYS}

    # Drop any rollup columns inherited from legacy score JSONs; recompute
    # from nested so the aggregator is the single source of truth.
    df = df.drop(columns=ROLLUP_KEYS, errors="ignore")
    rollup_df = pd.DataFrame(df.apply(_row_rollup, axis=1).tolist(), index=df.index)
    for key in ROLLUP_KEYS:
        df[key] = rollup_df[key]

    # Stable canonical order: agent, model (Categorical -> MODEL_ORDER),
    # effort, then trial id.
    return df.sort_values(
        ["_agent", "_agent_model", "_effort", "_trial_id"], kind="stable"
    ).reset_index(drop=True)


def collect_durations(jobs_dir: Path) -> list[dict]:
    """Collect trial durations from result.json files.

    Returns:
        List of dicts with keys: group, trial_id, duration_min.

    """
    jobs_dir = jobs_dir.resolve()
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")

    rows: list[dict] = []

    for batch_dir in sorted(jobs_dir.iterdir()):
        if not batch_dir.is_dir():
            continue

        agent_name, model_name = "unknown", "unknown"
        effort: str | None = None
        config_path = batch_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                if config.get("agents"):
                    agent_cfg = config["agents"][0]
                    agent_name = agent_cfg.get("name", "unknown")
                    model_name = agent_cfg.get("model_name", "unknown")
                    effort = (agent_cfg.get("kwargs") or {}).get("reasoning_effort")
            except Exception:
                pass

        group = format_group_label(agent_name, model_name, effort)

        for trial_dir in sorted(batch_dir.iterdir()):
            if not trial_dir.is_dir():
                continue

            result_path = trial_dir / "result.json"
            if not result_path.exists():
                continue

            try:
                result = json.loads(result_path.read_text())
            except Exception as e:
                logger.warning(f"Skipping {trial_dir}: {e}")
                continue

            agent_exec = result.get("agent_execution") or {}
            started_at = agent_exec.get("started_at")
            finished_at = agent_exec.get("finished_at")
            if not started_at or not finished_at:
                continue

            start = datetime.fromisoformat(started_at)
            end = datetime.fromisoformat(finished_at)
            duration_min = (end - start).total_seconds() / 60.0

            rows.append(
                {
                    "group": group,
                    "trial_id": trial_dir.name,
                    "duration_min": duration_min,
                }
            )

    return rows


class FlagdController:
    """Controls feature flags in the flagd configuration."""

    def __init__(self, flagd_path: Path, dry_run: bool = False):
        self.flagd_path = flagd_path
        self.dry_run = dry_run
        self._load_config()

    def _load_config(self) -> None:
        """Load the flagd configuration from disk."""
        if not self.flagd_path.exists():
            raise FileNotFoundError(f"Flagd config not found: {self.flagd_path}")
        with open(self.flagd_path, "r") as f:
            self.config = json.load(f)
        logger.debug(f"Loaded flagd config with {len(self.config['flags'])} flags")

    def _save_config(self) -> None:
        """Save the flagd configuration to disk."""
        if self.dry_run:
            logger.info("[DRY RUN] Would save config to disk")
            return
        with open(self.flagd_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def set_flag(self, flag_name: str, value: str) -> None:
        """Set a flag to the specified value."""
        if flag_name not in self.config["flags"]:
            raise ValueError(
                f"Flag '{flag_name}' not found. Available: {list(self.config['flags'].keys())}"
            )

        flag = self.config["flags"][flag_name]
        if value not in flag["variants"]:
            raise ValueError(
                f"Value '{value}' not valid for flag '{flag_name}'. "
                f"Available: {list(flag['variants'].keys())}"
            )

        old_value = flag["defaultVariant"]
        if self.dry_run:
            logger.info(f"[DRY RUN] Would set {flag_name}: {old_value} -> {value}")
        else:
            flag["defaultVariant"] = value
            self._save_config()
            utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.info(f"Set {flag_name}: {old_value} -> {value}")
            logger.info(f"Incident time: {utc_now}")

    def reset_all_flags(self) -> None:
        """Reset all flags to their 'off' state."""
        logger.info("Resetting all flags to 'off' state...")
        for flag_name, flag in self.config["flags"].items():
            if "off" in flag["variants"]:
                self.set_flag(flag_name, "off")
            else:
                logger.warning(f"Flag {flag_name} has no 'off' variant, skipping")

    def get_current_state(self) -> dict:
        """Get the current state of all flags."""
        return {
            name: flag["defaultVariant"] for name, flag in self.config["flags"].items()
        }


# ── Tool-call classification constants ─────────────────────────────────
CODE_CATEGORIES = {
    "code:ls",
    "code:find",
    "code:read_file",
    "code:search",
    "code:docker",
    "code:container_inspect",
}

# Harness control flow (terminus-2): waiting for output, interrupting a running
# command. These are mechanics of driving the terminal, not investigation, so
# they are intentionally kept out of CODE_CATEGORIES / TELEM_* sets.
CONTROL_CATEGORIES = {
    "control:wait",
    "control:interrupt",
}

TELEM_DATA_CATEGORIES = {
    "telemetry:metrics",
    "telemetry:spans",
    "telemetry:logs",
    "telemetry:docker_logs",
    "telemetry:grafana_query_unresolved",
}

TELEM_DISCOVERY_CATEGORIES = {
    "telemetry:grafana_health",
    "telemetry:grafana_ds_list",
    "telemetry:grafana_dashboard",
}

TELEM_ALL_CATEGORIES = TELEM_DATA_CATEGORIES | TELEM_DISCOVERY_CATEGORIES


def classify_cmd(cmd: str) -> list[str]:
    """Classify a single exec_command into one or more category labels.

    Returns a deduplicated list of category strings. A command can receive
    multiple labels (e.g., a Grafana datasource proxy call that queries
    Prometheus gets both telemetry:grafana_ds_list and telemetry:metrics).

    Categories:
        Code:
            code:ls                 — directory listing (ls)
            code:find               — find over a source tree (by ext or src/ path)
            code:read_file          — cat/head/tail/less/sed/nl/awk on source files
            code:search             — grep/rg targeting source files/dirs
            code:docker             — docker-compose file reads
            code:container_inspect  — docker inspect/exec/ps

        Control (terminus-2 harness mechanics, not investigation):
            control:wait            — empty/duration wait, readiness sentinel echo
            control:interrupt       — Ctrl-C/Ctrl-D/Ctrl-Z

        Telemetry discovery:
            telemetry:grafana_health    — /api/health
            telemetry:grafana_ds_list   — /api/datasources listing
            telemetry:grafana_dashboard — dashboard search

        Telemetry data queries:
            telemetry:metrics                   — Prometheus queries, PromQL, spanmetrics
            telemetry:spans                     — TraceQL, /api/traces, Jaeger API, Tempo proxy
            telemetry:logs                      — OpenSearch/webstore-logs, Loki
            telemetry:docker_logs               — docker logs commands
            telemetry:grafana_query_unresolved  — /api/ds/query with unidentifiable target

        Setup:
            setup:env_discovery     — printenv, echo $GRAFANA_URL
            setup:package_install   — pip install, apt install

        Utility:
            utility:timestamp           — date/time conversion
            utility:write_report        — writing /app/report.md
            utility:read_report         — reading /app/report.md
            utility:read_scratch        — reading the agent's own /tmp/*.txt notes
            utility:json_processing     — JSON parsing/building

        other — residual
    """
    cl = cmd.lower()
    categories = []

    # ── Harness control flow (match the RAW cmd; ctrl-keys are case-sensitive) ──
    # terminus-2 drives the terminal with bare waits, interrupts, and sentinel
    # echoes to detect shell readiness. These are mechanics, not commands, so we
    # return early before any code/telemetry rule can fire.
    raw = cmd.strip()
    if raw == "":
        return ["control:wait"]
    if re.fullmatch(r"C-[cdz]\s*", cmd):
        return ["control:interrupt"]
    if re.fullmatch(
        r"(echo|printf)\s+'?\"?(ready|recovered|done|ok|shell_ok)\\?n?'?\"?\s*",
        raw,
        re.IGNORECASE,
    ):
        return ["control:wait"]
    if re.fullmatch(r"(clear|reset|enter)\s*", raw, re.IGNORECASE):
        return ["control:wait"]

    # /app/report.md I/O is agent output, not code investigation. Detect it
    # up front so the cat/head/tail/sed/less/wc/grep code:* rules don't fire.
    is_report_io = bool(re.search(r"\breport\.md\b", cl))
    is_report_write = is_report_io and bool(
        re.search(r">>?\s*\S*\breport\.md\b", cl)
        or re.search(r"\btee\b\s+\S*\breport\.md\b", cl)
    )

    # Reads of the agent's own /tmp/*.txt evidence notes are scratch reads, not
    # source-code reads — detect up front so the cat/head/tail/sed code:* rules
    # don't claim them.
    is_scratch_io = bool(re.search(r"/tmp/\S*\.txt\b", cl))

    # ── Code ──
    if (
        re.search(r"\bls\b", cl)
        and "api" not in cl
        and "grafana" not in cl
        and not is_report_io
    ):
        categories.append("code:ls")
    if re.search(r"\bfind\b", cl) and (
        re.search(r"\.(go|ts|js|py|java|cs|rb|proto|yaml|yml)", cl)
        or "src/" in cl
        or "/app/opentelemetry-demo" in cl
    ):
        categories.append("code:find")
    if (
        re.search(r"\bcat\b|\bhead\b|\btail\b|\bless\b|\bwc\b|\bnl\b|\bawk\b", cl)
        and "api" not in cl
        and not is_report_io
        and not is_scratch_io
    ):
        categories.append("code:read_file")
    if re.search(r"\bsed\b.*src/", cl) and not is_report_io and not is_scratch_io:
        categories.append("code:read_file")
    if (
        re.search(r"\bgrep\b|\brg\b", cl)
        and (
            "src" in cl
            or any(ext in cl for ext in [".go", ".ts", ".js", ".py", ".java"])
        )
        and not is_report_io
    ):
        categories.append("code:search")
    if re.search(r"docker-compose|docker compose", cl) and not re.search(
        r"docker-compose logs|docker compose logs", cl
    ):
        categories.append("code:docker")
    if re.search(r"\bdocker\s+(inspect|exec|ps)\b", cl) and "grafana" not in cl:
        categories.append("code:container_inspect")

    # ── Telemetry discovery ──
    if re.search(r"api/health", cl):
        categories.append("telemetry:grafana_health")
    if re.search(r"api/datasources\b", cl) and not re.search(
        r"api/datasources/\d+/proxy", cl
    ):
        categories.append("telemetry:grafana_ds_list")
    if re.search(r"api/search\b|dashboard", cl):
        categories.append("telemetry:grafana_dashboard")

    # ── Telemetry: metrics ──
    if re.search(r"prometheus|prom|api/v1/query|api/v1/label", cl):
        categories.append("telemetry:metrics")
    if re.search(r"rate\(|histogram_quantile|sum\(|avg\(|increase\(", cl):
        categories.append("telemetry:metrics")
    if re.search(
        r'_total["\s\)]|_bucket["\s\)]|_count["\s\)]|_sum["\s\)]|_duration["\s\)]',
        cl,
    ):
        categories.append("telemetry:metrics")
    if re.search(r"webstore-metrics", cl):
        categories.append("telemetry:metrics")
    if re.search(r"datasources/proxy.*api/v1/", cl):
        categories.append("telemetry:metrics")

    # ── Telemetry: spans ──
    if re.search(r"traceql|/api/traces", cl):
        categories.append("telemetry:spans")
    if re.search(r"webstore-traces", cl):
        categories.append("telemetry:spans")
    if (
        re.search(r"jaeger|/api/services", cl)
        and "grep" not in cl
        and "printenv" not in cl
    ):
        categories.append("telemetry:spans")
    if re.search(r"datasources/proxy.*(?:jaeger|traces|tempo)", cl):
        categories.append("telemetry:spans")

    # ── Telemetry: logs ──
    if (
        re.search(r"webstore-logs|opensearch", cl)
        and "grep" not in cl
        and "printenv" not in cl
    ):
        categories.append("telemetry:logs")
    if re.search(r"loki|/loki/", cl) and "grep" not in cl and "printenv" not in cl:
        categories.append("telemetry:logs")
    if re.search(r"docker logs|docker-compose logs|docker compose logs", cl):
        categories.append("telemetry:docker_logs")

    # ── Telemetry: unresolved ds/query ──
    if re.search(r"api/ds/query", cl) and not any(
        c.startswith("telemetry:metrics")
        or c.startswith("telemetry:spans")
        or c.startswith("telemetry:logs")
        for c in categories
    ):
        categories.append("telemetry:grafana_query_unresolved")

    # ── Setup ──
    if re.search(r"\bprintenv\b", cl) or (
        re.search(r"\benv\b", cl) and re.search(r"grep|GRAFANA", cl)
    ):
        categories.append("setup:env_discovery")
    if re.search(r"pip install|apt-get|apt install", cl):
        categories.append("setup:package_install")
    if "echo" in cl and "grafana_url" in cl:
        categories.append("setup:env_discovery")

    # ── Utility ──
    if re.search(r"\bdate\b", cl) and not any(c for c in categories):
        categories.append("utility:timestamp")
    if is_report_io:
        categories.append(
            "utility:write_report" if is_report_write else "utility:read_report"
        )
    if is_scratch_io and re.search(
        r"\bcat\b|\bhead\b|\btail\b|\bless\b|\bsed\b|\bnl\b|\bawk\b|\bwc\b", cl
    ):
        categories.append("utility:read_scratch")

    # Deduplicate preserving order
    categories = list(dict.fromkeys(categories))

    # ── Fallback ──
    if not categories:
        if "python" in cl and (
            "datetime" in cl or "timestamp" in cl or "time.time" in cl
        ):
            categories.append("utility:timestamp")
        elif "python" in cl and "json" in cl:
            categories.append("utility:json_processing")
        elif "python" in cl and ("urllib" in cl or "request" in cl):
            categories.append("telemetry:grafana_query_unresolved")
        elif re.search(r"\bcurl\b", cl) and "http" in cl:
            categories.append("telemetry:grafana_query_unresolved")
        elif "grafana" in cl:
            categories.append("telemetry:grafana_query_unresolved")
        else:
            categories.append("other")

    return categories
