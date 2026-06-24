"""Token usage collection and formatting utilities.

Usage:
    from pbench_eval.token_utils import (
        collect_harbor_token_usage,
        collect_zeroshot_token_usage,
        format_token_statistics,
        count_trials_per_group,
    )
"""

import json
import re
from pathlib import Path
from typing import TypedDict
import yaml

import pandas as pd
from joblib import Memory
from natsort import natsorted
from tqdm import tqdm

from .stats import mean_sem_with_n


# joblib cache shared across all callers of ``collect_harbor_token_usage``.
# That walk reads every ``trajectory.json`` and per-call ``debug.json`` under a
# jobs directory (~minutes over a large run), so the result is memoized on the
# resolved ``jobs_dir`` path + flags. The cache invalidates only when those
# arguments change, so if the underlying trajectories/configs are updated in
# place, manually clear ``cachedir/joblib/`` to force a re-walk. Matches the
# pattern in :mod:`src.melt.utils` and ``examples/otel-demo/plot_efficiency.py``.
_cache = Memory("cachedir", verbose=1)


# Load model pricing configuration from YAML
_PRICING_CONFIG_PATH = Path(__file__).parent / "model_pricing.yaml"
with open(_PRICING_CONFIG_PATH, "r") as f:
    MODEL_PRICING: dict = yaml.safe_load(f)


# Exception types that represent legitimate agent-side outcomes (e.g. the
# agent ran past its time budget) rather than harness/transport failures.
# `get_predictions(drop_errored=True)` keeps these and drops everything else.
KEEP_EXCEPTION_TYPES: set[str] = {"AgentTimeoutError"}


class TokenUsageRecord(TypedDict, total=False):
    """Single token usage record from a trial or zeroshot run."""

    batch: str
    trial_id: str
    agent: str
    model_name: str
    reasoning_effort: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int
    total_cache_creation_5m_tokens: int
    total_cache_creation_1h_tokens: int
    total_thinking_tokens: int
    total_steps: int
    total_cost_usd: float | None


def get_predictions(
    jobs_dir: Path,
    drop_errored: bool = True,
) -> dict[str, dict]:
    """Collect predictions and results from all trials in a Harbor jobs directory.

    Args:
        jobs_dir: Path to the Harbor jobs directory
            Structure: <jobs_dir>/<job_name>/<trial_name>/verifier/report.md
                       <jobs_dir>/<job_name>/<trial_name>/result.json
        drop_errored: When True (default), drop trials whose ``result.exception_info``
            is set, EXCEPT for exception types in ``KEEP_EXCEPTION_TYPES`` (currently
            ``AgentTimeoutError``, which is a legitimate agent-side outcome rather
            than a harness failure). This removes harness failures (e.g.
            ``APIConnectionError`` rate-limit drops) that would otherwise show up
            alongside the resume-rerun successful trial as duplicate entries for
            the same task.

    Returns:
        Dict mapping trial_id to a dict with keys ``predictions`` and ``result``.
        If a file is missing, the corresponding value is ``{}``.

    """
    jobs_dir = Path(jobs_dir).resolve()
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")

    results: dict[str, dict] = {}
    dropped_by_type: dict[str, int] = {}
    kept_by_type: dict[str, int] = {}

    for batch_dir in natsorted(jobs_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        for trial_dir in natsorted(batch_dir.iterdir()):
            if not trial_dir.is_dir():
                continue

            predictions_path = trial_dir / "verifier" / "report.md"
            if predictions_path.exists():
                try:
                    predictions = predictions_path.read_text()
                except Exception:
                    predictions = ""
            else:
                predictions = ""

            result_path = trial_dir / "result.json"
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text())
                except Exception:
                    result = {}
            else:
                result = {}

            if drop_errored:
                exc_type = ((result or {}).get("exception_info") or {}).get(
                    "exception_type"
                )
                if exc_type is not None:
                    if exc_type in KEEP_EXCEPTION_TYPES:
                        kept_by_type[exc_type] = kept_by_type.get(exc_type, 0) + 1
                    else:
                        dropped_by_type[exc_type] = dropped_by_type.get(exc_type, 0) + 1
                        continue

            results[trial_dir.name] = {
                "predictions": predictions,
                "result": result,
            }

    if dropped_by_type:
        print(
            f"Dropped {sum(dropped_by_type.values())} errored trial(s) by "
            f"exception_type: {dropped_by_type}"
        )
    if kept_by_type:
        print(
            f"Kept {sum(kept_by_type.values())} errored trial(s) (legitimate "
            f"agent outcomes) by exception_type: {kept_by_type}"
        )

    return results


def _collect_per_call_usage(agent_dir: Path) -> dict | None:
    """Sum per-call cost + cache-creation tokens from ``episode-*/debug.json`` files.

    Returns ``None`` if no debug.json files exist or none of them yield a cost
    (e.g. unknown model in the pricing table); callers should fall back to the
    aggregated ``final_metrics`` path in that case.
    """
    debug_files = sorted(agent_dir.glob("episode-*/debug.json"))
    if not debug_files:
        return None

    cache_5m_total = 0
    cache_1h_total = 0
    cost_total: float | None = None
    for debug_path in debug_files:
        try:
            with open(debug_path) as f:
                debug = json.load(f)
        except Exception:
            continue
        if debug.get("log_event_type") != "post_api_call":
            continue
        try:
            response = json.loads(debug["original_response"])
        except (KeyError, ValueError):
            continue
        usage = response.get("usage") or {}
        model = response.get("model") or debug.get("model") or ""

        cache_creation = usage.get("cache_creation") or {}
        cache_5m_total += cache_creation.get("ephemeral_5m_input_tokens", 0) or 0
        cache_1h_total += cache_creation.get("ephemeral_1h_input_tokens", 0) or 0

        call_cost = calculate_cost_from_litellm_usage(model, usage)
        if call_cost is not None:
            cost_total = (cost_total or 0.0) + call_cost

    if cost_total is None:
        return None

    return {
        "cost": cost_total,
        "cache_creation_5m_tokens": cache_5m_total,
        "cache_creation_1h_tokens": cache_1h_total,
    }


@_cache.cache
def collect_harbor_token_usage(
    jobs_dir: Path,
    include_reasoning_effort: bool = False,
) -> list[TokenUsageRecord]:
    """Collect token usage from all trials in a Harbor jobs directory.

    Args:
        jobs_dir: Path to the Harbor jobs directory
        include_reasoning_effort: If True, extract reasoning_effort from trial config

    Returns:
        List of TokenUsageRecord dicts

    """
    jobs_dir = jobs_dir.resolve()
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")

    results: list[TokenUsageRecord] = []

    for batch_dir in tqdm(natsorted(jobs_dir.iterdir()), desc="Batches"):
        if not batch_dir.is_dir():
            continue

        # Batch config provides agent/model defaults, but a single batch can run
        # multiple agents (config["agents"] has >1 entry), so the per-trial
        # config is the source of truth.
        batch_agent_name, batch_model_name = "unknown", "unknown"
        config_path = batch_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                if config.get("agents"):
                    batch_agent_name = config["agents"][0].get("name", "unknown")
                    batch_model_name = config["agents"][0].get("model_name", "unknown")
            except Exception:
                pass

        for trial_dir in natsorted(batch_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            # Skip non-trial directories (e.g., config.json, result.json)
            if not (trial_dir / "agent").exists():
                continue

            trajectory_path = trial_dir / "agent/trajectory.json"
            if not trajectory_path.exists():
                continue

            try:
                with open(trajectory_path) as f:
                    trajectory = json.load(f)
            except Exception as e:
                print(f"Skipping {trial_dir}: {e}")
                continue

            # Read the per-trial config once: it carries the actual agent,
            # model, and reasoning_effort for this trial.
            agent_name, model_name = batch_agent_name, batch_model_name
            reasoning_effort = ""
            trial_config_path = trial_dir / "config.json"
            if trial_config_path.exists():
                try:
                    trial_config = json.loads(trial_config_path.read_text())
                    trial_agent = trial_config.get("agent", {})
                    agent_name = trial_agent.get("name", agent_name)
                    model_name = trial_agent.get("model_name", model_name)
                    if include_reasoning_effort:
                        reasoning_effort = trial_agent.get("kwargs", {}).get(
                            "reasoning_effort", ""
                        )
                except Exception:
                    pass

            # Extract final metrics
            final_metrics = trajectory.get("final_metrics", {})
            extra_metrics = final_metrics.get("extra", {})

            # Get step count: use "steps" field length for terminus-2, otherwise use final_metrics
            if agent_name == "terminus-2":
                total_steps = len(trajectory.get("steps", []))
            else:
                total_steps = final_metrics.get("total_steps", 0)

            # Get thinking tokens: check extra.reasoning_output_tokens (OpenAI) first
            thinking_tokens = extra_metrics.get(
                "reasoning_output_tokens",
                final_metrics.get("total_thinking_tokens", 0),
            )

            # Walk per-call debug.json to honor Anthropic's tiered + cache-write
            # billing; the trajectory's final_metrics aggregates lose the per-call
            # detail needed for tiering and don't track cache-creation tokens.
            per_call = _collect_per_call_usage(trial_dir / "agent")

            if per_call is not None:
                cache_5m = per_call["cache_creation_5m_tokens"]
                cache_1h = per_call["cache_creation_1h_tokens"]
                total_cost = per_call["cost"]
            else:
                cache_5m = 0
                cache_1h = 0
                total_cost = final_metrics.get("total_cost_usd")
                if total_cost is None:
                    total_cost = calculate_cost(
                        model=model_name,
                        prompt_tokens=final_metrics.get("total_prompt_tokens", 0),
                        completion_tokens=final_metrics.get(
                            "total_completion_tokens", 0
                        ),
                        cached_tokens=final_metrics.get("total_cached_tokens", 0),
                    )

            record: TokenUsageRecord = {
                "batch": batch_dir.name,
                "trial_id": trial_dir.name,
                "agent": agent_name,
                "model_name": model_name,
                "total_prompt_tokens": final_metrics.get("total_prompt_tokens", 0),
                "total_completion_tokens": final_metrics.get(
                    "total_completion_tokens", 0
                ),
                "total_cached_tokens": final_metrics.get("total_cached_tokens", 0),
                "total_cache_creation_5m_tokens": cache_5m,
                "total_cache_creation_1h_tokens": cache_1h,
                "total_thinking_tokens": thinking_tokens,
                "total_steps": total_steps,
                "total_cost_usd": total_cost,
            }
            if include_reasoning_effort:
                record["reasoning_effort"] = reasoning_effort

            results.append(record)

    return results


def _collect_from_usage_dir(output_dir: Path) -> list[TokenUsageRecord]:
    """Collect token usage from usage/ directory (supercon format).

    Args:
        output_dir: Path to the output directory containing usage/ subdirectory

    Returns:
        List of TokenUsageRecord dicts

    """
    usage_dir = output_dir / "usage"
    if not usage_dir.exists():
        raise FileNotFoundError(f"Usage directory not found: {usage_dir}")

    # Parse usage JSON files: usage__model=<model>__refno=<refno>.json
    # or usage__agent=<agent>__model=<model>__refno=<refno>.json
    usage_pattern = re.compile(r"usage__(?:agent=.+__)?model=(.+?)__refno=(.+)\.json")

    results: list[TokenUsageRecord] = []

    for usage_file in tqdm(natsorted(usage_dir.glob("*.json")), desc="Usage files"):
        match = usage_pattern.match(usage_file.name)
        if not match:
            print(f"Skipping unrecognized file: {usage_file.name}")
            continue

        model_name = match.group(1)
        refno = match.group(2)

        try:
            with open(usage_file) as f:
                usage = json.load(f)
        except Exception as e:
            print(f"Skipping {usage_file}: {e}")
            continue

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)
        thinking_tokens = usage.get("thinking_tokens", 0)

        # Calculate cost from tokens
        total_cost = calculate_cost(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )

        results.append(
            {
                "batch": "zeroshot",
                "trial_id": refno,
                "agent": "zeroshot",
                "model_name": model_name,
                "total_prompt_tokens": prompt_tokens,
                "total_completion_tokens": completion_tokens,
                "total_cached_tokens": cached_tokens,
                "total_thinking_tokens": thinking_tokens,
                "total_steps": 1,
                "total_cost_usd": total_cost,
            }
        )

    return results


def _collect_from_trajectories_dir(
    output_dir: Path,
    include_reasoning_effort: bool = False,
) -> list[TokenUsageRecord]:
    """Collect token usage from trajectories/ directory (biosurfactants format).

    Args:
        output_dir: Path to the output directory containing trajectories/ subdirectory
        include_reasoning_effort: If True, extract reasoning_effort from inf_gen_config

    Returns:
        List of TokenUsageRecord dicts

    """
    trajectory_dir = output_dir / "trajectories"
    if not trajectory_dir.exists():
        raise FileNotFoundError(f"Trajectories directory not found: {trajectory_dir}")

    # Parse trajectory JSON files:
    # trajectory__agent={agent}__model={model}__refno={refno}.json
    # trajectory__agent={agent}__model={model}__reasoning_effort={effort}__refno={refno}.json
    trajectory_pattern = re.compile(
        r"trajectory__agent=([^_]+)__model=([^_]+)__(?:reasoning_effort=[^_]+__)?refno=(.+)\.json"
    )

    results: list[TokenUsageRecord] = []

    for traj_file in tqdm(
        natsorted(trajectory_dir.glob("*.json")), desc="Trajectory files"
    ):
        match = trajectory_pattern.match(traj_file.name)
        if not match:
            print(f"Skipping unrecognized file: {traj_file.name}")
            continue

        agent = match.group(1)
        model_name = match.group(2).replace("--", "/")  # Convert back from safe name
        refno = match.group(3)

        try:
            with open(traj_file) as f:
                trajectory = json.load(f)
        except Exception as e:
            print(f"Skipping {traj_file}: {e}")
            continue

        # Get reasoning_effort from inf_gen_config
        reasoning_effort = ""
        if include_reasoning_effort:
            inf_gen_config = trajectory.get("inf_gen_config", {})
            reasoning_effort = inf_gen_config.get("reasoning_effort", "")
            if reasoning_effort is None:
                reasoning_effort = ""

        # Extract usage from llm_response (single) or llm_responses (list)
        llm_response = trajectory.get("llm_response")
        llm_responses = trajectory.get("llm_responses")

        if llm_responses is not None:
            # List of responses - aggregate usage across all
            usage_list = [r.get("usage", {}) for r in llm_responses if r.get("usage")]
            if not usage_list:
                print(f"No usage data in {traj_file.name}")
                continue
        elif llm_response is not None:
            # Single response
            usage = llm_response.get("usage", {})
            if not usage:
                print(f"No usage data in {traj_file.name}")
                continue
            usage_list = [usage]
        else:
            print(f"No llm_response or llm_responses in {traj_file.name}")
            continue

        # Aggregate usage across all responses
        prompt_tokens = 0
        cached_tokens = 0
        completion_tokens = 0
        thinking_tokens = 0
        total_cost: float | None = 0.0

        for usage in usage_list:
            u_prompt_tokens = usage.get("prompt_tokens", 0)
            u_cached_tokens = usage.get("cached_tokens", 0)

            # OpenAI uses output_tokens (includes reasoning_tokens)
            # Gemini uses completion_tokens (excludes thinking_tokens)
            if "output_tokens" in usage:
                # OpenAI format
                u_completion_tokens = usage["output_tokens"]
                u_thinking_tokens = usage.get("reasoning_tokens", 0)
                u_thinking_included_in_completion = True
            else:
                # Gemini format
                # NOTE: completion_tokens may be None
                u_completion_tokens = usage.get("completion_tokens", 0) or 0
                u_thinking_tokens = usage.get("thinking_tokens", 0)
                u_thinking_included_in_completion = False

            prompt_tokens += u_prompt_tokens
            cached_tokens += u_cached_tokens
            completion_tokens += u_completion_tokens
            thinking_tokens += u_thinking_tokens

            # Calculate cost for this response
            cost = calculate_cost(
                model=model_name,
                prompt_tokens=u_prompt_tokens,
                completion_tokens=u_completion_tokens,
                cached_tokens=u_cached_tokens,
                thinking_tokens=u_thinking_tokens,
                thinking_included_in_completion=u_thinking_included_in_completion,
            )
            if cost is not None and total_cost is not None:
                total_cost += cost
            else:
                total_cost = None

        record: TokenUsageRecord = {
            "trial_id": refno,
            "agent": agent,
            "model_name": model_name,
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_cached_tokens": cached_tokens,
            "total_thinking_tokens": thinking_tokens,
            "total_steps": len(usage_list),
            "total_cost_usd": total_cost,
        }
        if include_reasoning_effort:
            record["reasoning_effort"] = reasoning_effort

        results.append(record)

    return results


def count_trials_per_group(
    jobs_dir: Path,
    include_reasoning_effort: bool = False,
) -> dict[tuple, int]:
    """Count trials per agent/model (optionally with reasoning_effort).

    Args:
        jobs_dir: Path to the Harbor jobs directory
        include_reasoning_effort: If True, group by (agent, model, reasoning_effort)

    Returns:
        Dict mapping group tuple to trial count

    """
    jobs_dir = jobs_dir.resolve()
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")

    counts: dict[tuple, int] = {}

    for batch_dir in natsorted(jobs_dir.iterdir()):
        if not batch_dir.is_dir():
            continue

        # Batch config provides agent/model defaults, but a single batch can run
        # multiple agents (config["agents"] has >1 entry), so the per-trial
        # config is the source of truth.
        batch_agent, batch_model = None, None
        config_path = batch_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                if config.get("agents"):
                    batch_agent = config["agents"][0].get("name")
                    batch_model = config["agents"][0].get("model_name")
            except Exception:
                pass

        for trial_dir in natsorted(batch_dir.iterdir()):
            if not trial_dir.is_dir():
                continue

            agent, model = batch_agent, batch_model
            reasoning_effort = ""
            trial_config_path = trial_dir / "config.json"
            if trial_config_path.exists():
                try:
                    trial_config = json.loads(trial_config_path.read_text())
                    trial_agent = trial_config.get("agent", {})
                    agent = trial_agent.get("name", agent)
                    model = trial_agent.get("model_name", model)
                    if include_reasoning_effort:
                        reasoning_effort = trial_agent.get("kwargs", {}).get(
                            "reasoning_effort", ""
                        )
                except Exception:
                    pass

            if include_reasoning_effort:
                key = (agent, model, reasoning_effort)
            else:
                key = (agent, model)

            counts[key] = counts.get(key, 0) + 1

    return counts


def format_token_statistics(
    records: list[TokenUsageRecord],
    group_cols: list[str],
    trials_lookup: dict[tuple, int] | None = None,
    scale_factor: float = 1e6,
    scale_suffix: str = "M",
) -> pd.DataFrame:
    """Compute and format token usage statistics.

    Args:
        records: List of TokenUsageRecord dicts
        group_cols: Columns to group by (e.g., ["agent", "model_name"])
        trials_lookup: Optional dict mapping group keys to trial counts.
                       If None, counts unique trial_ids per group.
        scale_factor: Factor to divide token counts by (default: 1e6 for millions)
        scale_suffix: Suffix for scaled columns (default: "M")

    Returns:
        DataFrame with mean +/- SEM statistics per group

    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["total_tokens"] = df["total_prompt_tokens"] + df["total_completion_tokens"]

    # Fill missing thinking tokens with 0
    if "total_thinking_tokens" not in df.columns:
        df["total_thinking_tokens"] = 0
    df["total_thinking_tokens"] = df["total_thinking_tokens"].fillna(0)

    # Get trial counts for normalization
    if trials_lookup is None:
        trials_lookup = df.groupby(group_cols)["trial_id"].nunique().to_dict()

    # Metric columns and their display names
    metric_cols = [
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_thinking_tokens",
        "total_tokens",
        "total_steps",
        "total_cost_usd",
    ]

    # Scale token columns (except steps and cost)
    scale_cols = [
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_thinking_tokens",
        "total_tokens",
    ]
    df[scale_cols] = df[scale_cols] / scale_factor

    # Display column names
    display_cols = [
        f"Prompt ({scale_suffix})",
        f"Completion ({scale_suffix})",
        f"Cached ({scale_suffix})",
        f"Thinking ({scale_suffix})",
        f"Total ({scale_suffix})",
        "Steps",
        "Cost ($)",
    ]

    def compute_stats(g: pd.DataFrame) -> pd.Series:
        """Compute mean ± SEM for each metric column."""
        key = g.name  # tuple from groupby
        n_trials = trials_lookup.get(key, len(g))

        stats = {
            new_col: mean_sem_with_n(g[old_col].tolist(), n_trials)
            for old_col, new_col in zip(metric_cols, display_cols)
        }
        stats["successful_count"] = len(g)
        stats["num_trials"] = n_trials
        return pd.Series(stats)

    result = (
        df.groupby(group_cols).apply(compute_stats, include_groups=False).reset_index()
    )

    return result


_TIER_THRESHOLD = (
    200_000  # Anthropic's >200K-token tier kicks in above this prompt size.
)


def _resolve_price(prices: dict, base_key: str, over_threshold: bool) -> float | None:
    """Pick the >200K tier rate when applicable, falling back to the base rate."""
    if over_threshold:
        tier_price = prices.get(f"{base_key}_over_200k")
        if tier_price is not None:
            return tier_price
    return prices.get(base_key)


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    thinking_tokens: int = 0,
    thinking_included_in_completion: bool = False,
    cache_creation_5m_tokens: int = 0,
    cache_creation_1h_tokens: int = 0,
) -> float | None:
    """Calculate dollar cost based on token usage for a single API call.

    Honors Anthropic's tiered billing: when ``prompt_tokens`` exceeds 200K, the
    >200K rate is used for input/output (and cache rates if a ``*_over_200k``
    field is present in the model pricing). Cache-creation tokens are billed at
    their write rates instead of the base input rate.

    Args:
        model: Model name (must be in MODEL_PRICING).
        prompt_tokens: Total prompt/input tokens for this single call. For
            Anthropic-via-LiteLLM this already includes cache_read and
            cache_creation tokens.
        completion_tokens: Completion/output tokens.
        cached_tokens: Cache-read input tokens (billed at cache-read rate).
        thinking_tokens: Thinking/reasoning tokens (priced as output).
        thinking_included_in_completion: If True (OpenAI models), thinking_tokens are
            already included in completion_tokens and should not be added separately.
            If False (Gemini models), thinking_tokens are separate and should be added.
        cache_creation_5m_tokens: Tokens written to the 5-minute prompt cache.
        cache_creation_1h_tokens: Tokens written to the 1-hour prompt cache.

    Returns:
        Total cost in USD, or None if model pricing is unavailable.

    """
    # Strip provider prefix (e.g., "openai/gpt-5" -> "gpt-5")
    model_key = model.split("/")[-1] if "/" in model else model

    if model_key not in MODEL_PRICING:
        return None

    prices = MODEL_PRICING[model_key]
    if prices is None or not isinstance(prices, dict):
        return None

    over_threshold = prompt_tokens > _TIER_THRESHOLD

    input_price = _resolve_price(prices, "input", over_threshold)
    output_price = _resolve_price(prices, "output", over_threshold)
    cache_read_price = _resolve_price(prices, "context_cache_read", over_threshold)
    if cache_read_price is None:
        cache_read_price = _resolve_price(prices, "cached_input", over_threshold)
    cache_5m_price = _resolve_price(prices, "cache_creation_5m", over_threshold)
    cache_1h_price = _resolve_price(prices, "cache_creation_1h", over_threshold)

    if input_price is None or output_price is None:
        return None

    # Anthropic's prompt_tokens (in OpenAI-compat form) already counts cache_read
    # and cache_creation tokens, so the fresh-input portion is what's left over.
    fresh_input = (
        prompt_tokens
        - cached_tokens
        - cache_creation_5m_tokens
        - cache_creation_1h_tokens
    )
    fresh_input = max(fresh_input, 0)

    prompt_cost = fresh_input * input_price / 1_000_000
    cache_read_cost = cached_tokens * (cache_read_price or 0) / 1_000_000
    # If a model's YAML doesn't list cache-creation rates, fall back to the
    # input rate so we don't silently undercharge.
    cache_5m_cost = (
        cache_creation_5m_tokens * (cache_5m_price or input_price) / 1_000_000
    )
    cache_1h_cost = (
        cache_creation_1h_tokens * (cache_1h_price or input_price) / 1_000_000
    )
    completion_cost = completion_tokens * output_price / 1_000_000

    # For Gemini, thinking_tokens are separate from completion_tokens
    # For OpenAI, thinking_tokens are already included in completion_tokens
    if thinking_included_in_completion:
        thinking_cost = 0
    else:
        thinking_cost = thinking_tokens * output_price / 1_000_000

    return (
        prompt_cost
        + cache_read_cost
        + cache_5m_cost
        + cache_1h_cost
        + thinking_cost
        + completion_cost
    )


def calculate_cost_from_litellm_usage(model: str, usage: dict) -> float | None:
    """Compute per-call cost from a LiteLLM-format ``usage`` dict.

    Reads Anthropic-specific fields (``cache_read_input_tokens``,
    ``cache_creation.ephemeral_5m_input_tokens``,
    ``cache_creation.ephemeral_1h_input_tokens``) when present so that cache
    reads, cache writes, and the >200K tier are billed correctly.
    """
    cache_creation = usage.get("cache_creation") or {}
    return calculate_cost(
        model=model,
        prompt_tokens=usage.get("prompt_tokens", 0) or 0,
        completion_tokens=usage.get("completion_tokens", 0) or 0,
        cached_tokens=usage.get("cache_read_input_tokens", 0) or 0,
        cache_creation_5m_tokens=cache_creation.get("ephemeral_5m_input_tokens", 0)
        or 0,
        cache_creation_1h_tokens=cache_creation.get("ephemeral_1h_input_tokens", 0)
        or 0,
    )
