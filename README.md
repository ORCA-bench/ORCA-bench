# ORCA-bench: How Ready Are Language Model Agents for Oncall?

[![Dataset](https://img.shields.io/badge/Dataset-Harbor%20Hub-0b7285?style=flat-square&logo=harbor&logoColor=white)](https://hub.harborframework.com/datasets/orca-bench/orca-bench/latest)
[![Website](https://img.shields.io/badge/Website-orca--bench.github.io-1c7ed6?style=flat-square&logo=githubpages&logoColor=white)](https://orca-bench.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2607.28545-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.28545)

This repo contains the instructions to submit to our [leaderboard](https://hub.harborframework.com/datasets/orca-bench/orca-bench/latest?tab=leaderboard&leaderboard=orca-bench) and reproduce the dataset construction and evaluation experiments in our paper [ORCA-bench: How Ready Are Language Model Agents for Oncall?](https://arxiv.org/abs/2607.28545).

```bibtex
@article{gong2026orca,
  title={ORCA-bench: How Ready Are Language Model Agents for Oncall?},
  author={Gong, Albert and Choi, Kyuseong and Agarwal, Abhineet and Schechner, Jason and Huang, Ryan and Agrawal, Raj and Agarwal, Anish and Dwivedi, Raaz},
  journal={arXiv preprint arXiv:2607.28545},
  year={2026}
}
```

## Setup Instructions

1. Install Python dependencies:

```bash
pip install uv
uv sync --all-groups
```

### Additional setup instructions for data collection

The telemetry data collection was performed on a DigitalOcean Droplet with 32 GB / 8 CPUs.

We collect telemetry data using the codebase at https://github.com/open-telemetry/opentelemetry-demo with the following modifications:

1. We use OpenSearch as the storage backend for Jaeger as recommended [here](https://www.jaegertracing.io/docs/1.76/faq/#what-is-the-recommended-storage-backend). Jaeger uses date-based indices (e.g. `jaeger-main-jaeger-span-2026-02-17`) and only looks back `max_span_age` when querying — it builds the index-name list from `now - max_span_age`, so once that window no longer reaches the snapshot's dates, the service list silently comes back empty. Because snapshot data is frozen while wall-clock time advances, `max_span_age` is **computed at start-up** by `set_max_span_age.sh` from `SNAPSHOT_DATA_DATE` (default `2026-04-18`) plus `MAX_SPAN_AGE_MARGIN_DAYS` (default 7); the value in `jaeger-config-snapshot.yml` is only a placeholder. Jaeger names one index per day in the query URL, so a long window needs a matching `http.max_initial_line_length` on the opensearch service (set to `64kb` in `patches/docker-compose.yml`, ~1724 days of headroom); `set_max_span_age.sh` warns when the estimate passes 75% of that limit.

2. We also increase the memory resources for certain containers:

| Service | Original | New | Notes |
|---------|----------|-----|-------|
| `opensearch` | 1G | 2G | Prevents OOM kills |
| `llm` | 50M | 100M | Prevents OOM kills |
| `product-catalog` | 20M | 80M | Requires more memory to start |
| `product-catalog` GOMEMLIMIT | 16MiB | 64MiB | Go runtime memory limit |

Please follow these steps to set up the OpenTelemetry Demo system:

```bash
# Initialize submodules
git submodule update --init --recursive
# Patch: increase Docker resources and add configurable volume mount directories
cp patches/docker-compose.yml opentelemetry-demo/
# Patch: use OpenSearch backend for Jaeger
cp patches/opensearch-config.yml opentelemetry-demo/src/jaeger/
# Patch: pin otel-demo service image tags to 2.2.0 (DEMO_VERSION=latest → 2.2.0)
# so runtime images match the submodule's v2.2.0 source line. Required — the
# upstream :latest-frontend-proxy tag added a telemetry-docs cluster whose env
# vars our compose doesn't set, causing envoy to fail with
# "SocketAddressValidationError.Address: value length must be at least 1 characters".
cp patches/opentelemetry-demo.env opentelemetry-demo/.env
```

### Additional setup instructions for Harbor

Patch Harbor's `Task.checksum` to ignore `environment/.trials/`. The harbor-template entrypoint stages each trial's OpenTelemetry demo + live OpenSearch data under `environment/.trials/<PROJECT_HASH>/`; without this patch, `dirhash` walks into another in-flight trial's snapshot tree and races OpenSearch's segment renames, raising `FileNotFoundError` mid-scan. The unpatched checksum is also non-deterministic for the same reason.

```bash
cp patches/harbor_models_task_task.py \
  "$(uv run python -c "import harbor.models.task.task as m; print(m.__file__)")"
```

The patch tracks `harbor.models.task.task` verbatim except for the `ignore=[".trials/"]` argument, so re-copy it from the installed module and re-apply that one hunk whenever harbor is upgraded. As of harbor 0.20.0 `Task.checksum` raises a `DeprecationWarning` in favor of `TrialLock.task.digest`, but `trial/trial.py` still calls it once per trial, so the patch remains required until that call goes away.

<details>
<summary>Using GradientAI's serverless inference from DigitalOcean</summary>

1. Please go to https://cloud.digitalocean.com/gen-ai/model-access-keys -> Create model access key. Then configure the following environment variables:

```bash
# Set Gradient AI API key to access GradientAI models via LiteLLM
export GRADIENT_AI_API_KEY=$MODEL_ACCESS_KEY

# Set OpenAI API key and base URL for LLM judge in Harbor task verifier
export OPENAI_API_KEY=$MODEL_ACCESS_KEY
export OPENAI_BASE_URL=https://inference.do-ai.run/v1
```

2. Patch LiteLLM to support `reasoning_effort` and Anthropic `cache_control` passthrough for Gradient AI:

```bash
cp patches/litellm_gradient_ai_chat_transformation.py \
  "$(uv run python -c "import litellm; print(litellm.__path__[0])")/llms/gradient_ai/chat/transformation.py"
```

</details>

## Quick start

1. Pull the Docker image (only need to do this step once):
```bash
IMAGE=orcabench/sre-otel-snapshot:data-0418-harbor-template-v2
CACHE_ROOT="/root/.cache/sre-snapshot-cache"
CACHE_DIR="$(./stage_snapshot_cache.sh --image "$IMAGE" --cache-root "$CACHE_ROOT")"
```

2. Run the oracle solution on a single task:
```bash
SNAPSHOT_CACHE_HOST_DIR="$CACHE_DIR" uv run harbor run \
    --mounts-json "[{\"type\":\"bind\",\"source\":\"$CACHE_DIR\",\"target\":\"$CACHE_DIR\",\"read_only\":true}]" \
    -t orca-bench/05216f608f940a48
```

2. Run Claude Sonnet 4.6 using the Terminus-2 agent harness on a single task:
```bash
SNAPSHOT_CACHE_HOST_DIR="$CACHE_DIR" uv run harbor run \
    --mounts-json "[{\"type\":\"bind\",\"source\":\"$CACHE_DIR\",\"target\":\"$CACHE_DIR\",\"read_only\":true}]" \
    -t orca-bench/05216f608f940a48 \
    -a terminus-2 \
    -m gradient_ai/anthropic-claude-4.6-sonnet \
    --ak temperature=1 \
    --ak reasoning_effort=medium \
    --ak 'llm_kwargs={"max_tokens": 16384}'
```

## Submitting to the leaderboard

See [`leaderboard/SUBMIT.md`](./leaderboard/SUBMIT.md) for the full walkthrough; the short version:

1. **Run and upload.** Use [`job-config.yaml`](./job-config.yaml) at the repo
   root — it pins the dataset as published on the Harbor Hub, which is what CI
   validates against. Add `--upload --public` so the trials are readable from
   the hub:

```bash
SNAPSHOT_IMAGE=orcabench/sre-otel-snapshot:data-0418-harbor-template \
    ./run_harbor_cached.sh -c job-config.yaml \
      -a <agent> -m <provider/model> \
      --upload --public
```

2. **Open the PR.** The `lb` CLI turns finished jobs into one submission JSON
   and one PR per (agent, agent version, model, reasoning effort):

```bash
cd leaderboard
uv run lb submit https://hub.harborframework.com/jobs/<uuid> [more...]
```

3. **Follow the PR.** CI checks the submission against the job config the hub
   recorded with your run — dataset ref, task coverage, `timeout_multiplier`,
   and per-trial task digests — then promotes it to a bot PR carrying the
   computed metrics. A maintainer merges that, and your row appears on the
   leaderboard.

The board publishes three metrics, each as `mean ± one standard error` and each
restricted to **incident** tasks:

| Column | What it measures |
| --- | --- |
| **RCA Accuracy (Medium)** ↑ | did the report name the root cause, on the broad-feature-area tier |
| **RCA Accuracy (Hard)** ↑ | the same, on the flag-agnostic tier — no hint which feature flag failed |
| **Hallucination Rate** ↓ | did the report name a root cause matching none of the plausible ones |

Rows rank by RCA Accuracy (Medium). Control tasks back none of these: they have
no root cause, so RCA accuracy is 0 there by construction and hallucination is
undefined. The easy tier is measured but not published — it looks deceptively
good and pulls attention from the realistic setting.

A trial that produced no verdict — the run died before the verifier scored it —
is excluded rather than counted as zero, matching how `format_acc.py` treats
unscored trials.

Maintainers setting the repo up for the first time should read
[`leaderboard/SETUP.md`](./leaderboard/SETUP.md); publishing a new dataset
revision is covered by [`PUBLISH.md`](./PUBLISH.md).

## Constructing the dataset

1. Initialize data volumes for persisting telemetry data and follow the provided instructions to export the data mount environment variables:

```bash
./init_data_dirs.sh DATA_DIR
```

2. Launch the demo with the following command (source: https://opentelemetry.io/docs/demo/docker-deployment/):

> \[!IMPORTANT\]
> The environment variables `PROMETHEUS_MOUNT_DIR`, `OPENSEARCH_DATA_MOUNT_DIR`, and `OPENSEARCH_SNAPSHOTS_MOUNT_DIR` must be set for `docker compose` to work. See Step 1.

```bash
docker compose -f opentelemetry-demo/docker-compose.yml up --force-recreate --remove-orphans --detach
```

> \[!TIP\]
> To check the health of the telemetry data, please run:
> ```bash
> ./check_health.sh
> ```

3. Run the incident schedule using the following command, saving snapshots at regular intervals:

```bash
# Run the script at a scheduled time using the `atd` tool
TZ="America/New_York" at midnight -f run_scheduled.sh
```

4. After the incident schedule is over, tear down the docker to stop recording data:

```bash
docker compose -f opentelemetry-demo/docker-compose.yml down
```

<details>
  <summary>Steps to share the data</summary>

Zip does not perserve Prometheus snapshot hardlinks, leading to large amounts of duplicated data. Use tar instead. `--zstd` provides faster compression over gzip.

```bash
LATEST=$(ls -t data-0418/prometheus/snapshots | head -1)
tar --zstd -cf data-0418.tar.zst \
    --exclude="data-0418/prometheus/snapshots/${LATEST}" \
    data-0418/prometheus/snapshots \
    data-0418/opensearch-repo-snapshots

# To minimize load on the VM, please use the following version of the command
nice -n 19 ionice -c 3 \
  tar --zstd -cf data-0418.tar.zst \
      --exclude="data-0418/prometheus/snapshots/${LATEST}" \
      data-0418/prometheus/snapshots \
      data-0418/opensearch-repo-snapshots

# Extract with the following command
tar --zstd -xf data-0418.tar.zst
```

</details>

5. Please follow the steps below to [curate ground-truth symptoms](#curating-ground-truth-symptoms). The ground-truth data should follow the schema at [incident_schema_2w.json](./incident_schema_2w.json).

6. Generate questions using GPT-5.4 (high):

```bash
uv run python generate_questions.py --schedule SCHEDULE_PATH -od OUTPUT_DIR

# Example: uv run python generate_questions.py --schedule schedules/incident_schedule_dev_single_2.json -od OUTPUT_DIR
```

7. Create Harbor tasks and publish them to the [Harbor Hub](https://hub.harborframework.com/datasets/orca-bench/orca-bench):

```bash
uv run python generate_task_specs.py \
  -od OUTPUT_DIR -dd data-0418 -e high \
  --config configs/harbor_tasks_v2_share2w.yaml --force
uv run python generate_answers.py -od OUTPUT_DIR -dd data-0418 -e high --concurrency 1600

# build_harbor_tasks.py writes OUTPUT_DIR/harbor/:
#   - tasks/<hash>/            one Harbor task per question; task.toml [task].name is a
#                              hash of the task id so the published name doesn't leak the
#                              feature flag, and all metadata is mirrored into [metadata].
#   - dataset.toml             ready-to-publish manifest (no separate `harbor dataset init`
#                              / `harbor add --scan` needed).
#   - tasks.csv                maps the real task_id -> hashed package_name.
#   - used_snapshots.json, task_snapshot_map.json
uv run python build_harbor_tasks.py -od OUTPUT_DIR -dd data-0418 \
  --templates-dir harbor-template --force \
  --dataset-name orca-bench/orca-bench \
  --dataset-description "An agent benchmark for root cause analysis" \
  --dataset-author "Your Name <your@email.com>"

# Upload the task content to the Harbor Hub, then publish the dataset manifest
# (the script prints these two commands when it finishes):
uv run harbor add OUTPUT_DIR/harbor/tasks --scan
uv run harbor publish OUTPUT_DIR/harbor/dataset.toml
```

8. Build and push the Harbor Docker image with the snapshot
   data baked in:

```bash
uv run python build_and_push_snapshots.py --data-dir DATA_DIR

# Example:
#   uv run python build_and_push_snapshots.py --data-dir data-0329-2
```

### Curating ground-truth symptoms

1. Load the latest snapshot from `DATA_DIR`:

> \[!IMPORTANT\]
> It takes 1-2 minutes for all the services to launch. To check that everything is set up property, please run `./check_health.sh`.

```bash
# Loads the latest snapshot (lexicographic sort works because names are timestamp-prefixed)
./load_snapshot.sh DATA_DIR "$(ls -1 DATA_DIR/prometheus/snapshots | sort | tail -n 1)"
```

2. Please following the instructions below for each telemetry type to find golden signals:

> \[!NOTE\]
> Steps marked by :robot: involve LLM assistance.

<details>
  <summary>Instructions for traces</summary>

- Programmatically obtain all traces with the tag `error=true`:

```bash
uv run python find_traces.py -dd DATA_DIR -od OUTPUT_DIR
```

- Below we provide an example prompt for the event with `event_id="dev-1-productCatalogFailure"` and `feature_flag="productCatalogFailure"`:

```
You are given a list of traces: @OUTPUT_DIR/traces/traces-dev-1-productCatalogFailure-error.json

The trace data contains both incident signal and background noise. You MUST classify each group of traces and only include signal in the Timeline:

- **Signal**: Errors directly caused by the feature flag described in the rubric.
- **Noise (background)**: `flagd.evaluation.v1.Service/EventStream` spans with DEADLINE_EXCEEDED — these are normal ~10-minute long-poll reconnection cycles for feature flag streaming. They appear in EVERY observation window regardless of incidents. Always exclude these.
- **Noise (co-occurring)**: Errors caused by OTHER feature flags active in the same window. Identify these by looking for `feature_flag.evaluation` span events where a different flag key has variant `on`. Mention these briefly in the Timeline as "co-occurring incidents" but do not attribute them to the incident under analysis.
- **Noise (pre-incident)**: Errors with timestamps before the incident time. These cannot be caused by the feature flag and should be noted as pre-existing.

If there are NO signal traces (e.g., the incident causes latency degradation, not errors), explicitly state this and explain why the error-based trace query found no relevant signal.

Use the code base at @opentelemetry-demo/, the official documentation at @opentelemetry.io and the Grafana API at localhost:8080/grafana/
```

After several back and forth, please instruct the agent to save the output to `OUTPUT_DIR/json/{event_id}-{feature_flag}.json` following the schema at [incident_schema_2w.json](incident_schema_2w.json).

</details>

<details>
  <summary>Instructions for logs</summary>

- Programmatically obtain all logs matching "error" keywords:

```bash
uv run python find_logs.py -dd DATA_DIR -od OUTPUT_DIR
```

- Below we provide an example prompt for the event with `event_id="dev-1-productCatalogFailure"` and `feature_flag="productCatalogFailure"`:

```
You are given a list of logs: @OUTPUT_DIR/logs/unique-logs-dev-1-productCatalogFailure-error.json
                                                                              
Are there any logs that indicate the productCatalogFailure feature flag is turned on?
                                                                              
Use the code base at @opentelemetry-demo/, the official documenation at @opentelemetry.io and the Grafana API at localhost:8080/grafana/
```

After several back and forth, please instruct the agent to save the output to `OUTPUT_DIR/json/{event_id}-{feature_flag}.json` following the schema at [incident_schema_2w.json](incident_schema_2w.json).

</details>

<details>
  <summary>Instructions for metrics</summary>

- :robot: Find metrics using multiple Claude calls and plot them as a PDF for verification. Reads the JSON rubric for the given feature flag from `rubrics/json/` to obtain the incident description, mechanism, and golden-signal log/trace entries, which are injected into each sub-agent prompt so the agent can cross-reference log and trace evidence when identifying relevant metrics:

```bash
uv run python find_metrics_family.py -f FEATURE_FLAG -t UTC_TIME -od OUTPUT_DIR -n PARALLEL_CALLS

# Example:
#   uv run python find_metrics_family.py -f productCatalogFailure \
#     -t "2026-04-02 06:25:59 UTC" -od OUTPUT_DIR -n 3
```

- Review the per-run JSON files and PDF plots manually. Fix misclassified types (e.g., `"created (fixed after human verification)"`). Mark dubious entries as `inconclusive`.

- :robot: Detect onset timing for each metric using Grafana/Prometheus (requires snapshot loaded via `./load_snapshot.sh`):

```bash
uv run python onset_metrics.py -f FEATURE_FLAG -t UTC_TIME -od OUTPUT_DIR \
  -dir INPUT_DIR -m MODEL -e EFFORT

# Example:
#   uv run python onset_metrics.py -f productCatalogFailure \
#     -t "2026-04-02 06:25:59 UTC" -od OUTPUT_DIR \
#     -dir metrics-0401/metrics -m opus -e max
```

- Review onset JSON files. Verify `onset_utc` aligns with PDF plots and that `created` metrics truly didn't exist pre-incident.

- :robot: Group into metric families and annotate signal layers (`root_cause`, `propagation`, `symptom`, `meta`):

```bash
uv run python label_metrics.py -f FEATURE_FLAG -od OUTPUT_DIR \
  -dir INPUT_DIR -m MODEL -e EFFORT

# Example:
#   uv run python label_metrics.py -f productCatalogFailure \
#     -od OUTPUT_DIR -dir metrics-0401/metrics_onset -m opus -e max
```

</details>

<details>
  <summary>Instructions for frontend symptoms</summary>

1. Repeat steps 1 and 2 from [Constructing the dataset](#constructing-the-dataset).

2. Open the web store at http://localhost:8080/.

3. Use the flagd configurator UI at http://localhost:8080/feature to fire feature flags. Then observe the effect on the web store.

</details>

## Reproducing the evaluation experiments (on the public set)

> [!IMPORTANT]
> To prevent data contamination, we only release 755 out of the 1,079 tasks used in the paper's evaluation experiments.

1. Format accuracy using the scores in `<trial>/verifier/reward-details.json` and task's `task.toml` `[metadata]` — section, granularity, phrasing, flag — downloaded from the Harbor Hub and cached under `~/.cache/harbor/tasks`.

```bash
# Generate Figure 1: RCA accuracy and hallucination
uv run python plot_rca_and_hallucination.py -jd JOBS_DIR -od OUTPUT_DIR -e high

# Generate Figure 4: RCA accuracy and depth by difficulty
uv run python plot_rca_difficulty_bars.py -jd JOBS_DIR -od OUTPUT_DIR -e high

# Generate table of accuracy results for appendix
uv run python format_acc.py -jd JOBS_DIR -od OUTPUT_DIR -e high

# Same table stratified by difficulty, plus per-difficulty bar plots
uv run python format_acc_by_granularity.py -jd JOBS_DIR -od OUTPUT_DIR -e high
```

> [!TIP]
> `-jd/--jobs-dir` is what they **read** and accepts either a single job
> directory (`jobs/2026-05-05__04-10-26`) or a tree of them (`jobs`), in which
> case every job is pooled into one table. `-od/--output-dir` is only where CSVs
> and figures are **written**. `-e/--effort` filters on the judge's reasoning
> effort, not the agent's.

2. Generate token and tool-usage figures:

* Pre-process Terminus-2 trajectories using the following command:

```bash
# Step 1 (recover_outputs) parses asciinema recording.cast to sidestep the
# terminus-2 40-row pane truncation that inflates apparent error rates 4-5x
# in trajectory.json. Steps 2-3 then judge + aggregate per-agent outcomes.
uv run python recover_outputs.py        -od OUTPUT_DIR -jd jobs-sub

# Generate CSV of tool calls per trial
uv run python format_tool_calls.py -od OUTPUT_DIR -jd jobs-sub
```

* Plot per-model command-category Markov-chain transition diagrams. `--models`
   filters panels by substring of the agent-model label and `--fig-name` sets the
   output file, so the two commands below render a 2-model main-paper figure
   (Opus 4.7 + GLM-5) and a 3-model companion appendix figure (the remaining
   models). `--min-prob 0.10` drops edges whose transition probability
   `P(next|prev)` is below 10%. The joint-flow edge-width scale is computed over
   all groups, so the two figures stay comparable.

```bash
# Main-paper figure: Opus 4.7 and GLM-5
uv run python format_markov_chain.py -od OUTPUT_DIR -jd jobs-sub \
    --models opus-4.7 glm-5 --min-prob 0.10 --fig-name markov_chain.pdf

# Appendix figure: Sonnet 4.6, GPT-5.5, DeepSeek-V4-Pro
uv run python format_markov_chain.py -od OUTPUT_DIR -jd jobs-sub \
    --models 4.6-sonnet gpt-5.5 deepseek-v4-pro --min-prob 0.10 --fig-name markov_chain_appendix.pdf
```

* Generate retrieval plot (mean telemetry commands, per-type failure rate, per-type any-match rate):

```bash
# Classify telemetry calls into success, empty, error
uv run python classify_telemetry_calls.py -od OUTPUT_DIR --concurrency 800 -e high -bs 100 -bn 1

# Generate Figure 8: Retrieval diagnostics
uv run python plot_retrieval.py -od OUTPUT_DIR -jd jobs-sub
```
