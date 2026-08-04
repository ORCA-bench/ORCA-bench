# ORCA-bench: How Ready Are Language Model Agents for Oncall?

<!-- [Dataset](https://hub.harborframework.com/datasets/orca-bench/ORCA-bench/latest) · [Website](https://orca-bench.github.io) · Paper (TBD) -->
[Dataset](https://hub.harborframework.com/datasets/orca-bench/ORCA-bench/latest) · [Website](https://orca-bench.github.io)

<!-- This repo contains the instructions to reproduce the dataset construction and evaluation in our paper [ORCA-bench: How Ready Are Language Model Agents for Oncall?]().

```bibtex
``` -->

## Setup Instructions

1. Install Python dependencies:

```bash
pip install uv
uv sync --all-groups
```

### Additional setup instructions for data collection

The telemetry data collection was performed on a DigitalOcean Droplet with 32 GB / 8 CPUs.

We collect telemetry data using the codebase at https://github.com/open-telemetry/opentelemetry-demo with the following modifications:

1. We use OpenSearch as the storage backend for Jaeger as recommended [here](https://www.jaegertracing.io/docs/1.76/faq/#what-is-the-recommended-storage-backend). Jaeger uses date-based indices (e.g. `jaeger-main-jaeger-span-2026-02-17`) and by default only looks back 72 hours (`max_span_age`) when querying. When loading historical snapshots older than 3 days, the `max_span_age` in `jaeger-config-snapshot.yml` must be large enough to cover the age of the snapshot data, otherwise the Jaeger service list will appear empty. It is currently set to `2160h` (90 days). Note: very large values (e.g. 1 year+) cause OpenSearch's 4096-byte HTTP line limit to be exceeded, since Jaeger generates one index name per day in the query URL.

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

## Constructing Dataset

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

7. Create Harbor tasks and publish them to the [Harbor Hub](https://hub.harborframework.com/datasets/orca-bench/ORCA-bench):

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
# NOTE: for the telemetry-only environment, pass `--templates-dir harbor-template-telemetry-only`
uv run python build_harbor_tasks.py -od OUTPUT_DIR -dd data-0418 \
  --templates-dir harbor-template --force \
  --dataset-name orca-bench/ORCA-bench \
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

1. Repeat steps 1 and 2 from [Constructing Dataset](#constructing-dataset).

2. Open the web store at http://localhost:8080/.

3. Use the flagd configurator UI at http://localhost:8080/feature to fire feature flags. Then observe the effect on the web store.

</details>

## Reproducing Harbor evaluation

1. Launch the Harbor tasks using the following command:

```bash
# Launch jobs (note: this script pre-pulls the Docker image with the snapshot to /root/.cache/sre-snapshot-cache so that we can bind-mount it to all tasks).
# Each config pairs one task dataset with the snapshot image whose `/app/` matches its environment.
SNAPSHOT_IMAGE=orcabench/sre-otel-snapshot:data-0418-harbor-template \
    ./run_harbor_cached.sh -c configs/harbor_job_orca_bench.yaml
```

`configs/harbor_job_orca_bench.yaml` sources tasks from the HuggingFace
registry and is the config for local development. For a run you intend to
[submit to the leaderboard](#submitting-to-the-leaderboard), use
[`job-config.yaml`](./job-config.yaml) at the repo root instead — it pins the
dataset as published on the Harbor Hub, which is what CI validates against:

```bash
SNAPSHOT_IMAGE=orcabench/sre-otel-snapshot:data-0418-harbor-template \
    ./run_harbor_cached.sh -c job-config.yaml \
      --ve OPENAI_API_KEY="$OPENAI_API_KEY" \
      --upload --public
```

<details>
<summary>Steps without shell scripts</summary>

> [!NOTE]
> Substitute `job-config.yaml` for a run you intend to submit to the leaderboard.

```bash
# Stage the snapshot's /app/ to a host-side cache keyed by image id.
SNAPSHOT_IMAGE=orcabench/sre-otel-snapshot:data-0418-harbor-template
docker pull "$SNAPSHOT_IMAGE"
CACHE_DIR="$HOME/.cache/sre-snapshot-cache/$(docker image inspect "$SNAPSHOT_IMAGE" -f '{{.Id}}' | cut -d: -f2 | cut -c1-12)"
[ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ] && { mkdir -p "$CACHE_DIR"; CID=$(docker create "$SNAPSHOT_IMAGE"); docker cp "$CID:/app/." "$CACHE_DIR/"; docker rm "$CID"; }

# Bind-mount the cache (read-only) into every trial at the same host path
# (target=source, so the entrypoint can `cp -al` without translating it).
SNAPSHOT_CACHE_HOST_DIR="$CACHE_DIR" uv run harbor run \
    --mounts-json "[{\"type\":\"bind\",\"source\":\"$CACHE_DIR\",\"target\":\"$CACHE_DIR\",\"read_only\":true}]" \
    -c configs/harbor_job_orca_bench.yaml
```

</details>

2. Format accuracy results.

These scripts read scores straight out of the job tree: for every trial they
load `<trial>/verifier/reward-details.json` (written by the verifier),
`<trial>/config.json` for the agent and model, and the task's `task.toml`
`[metadata]` — section, granularity, phrasing, flag — downloaded from the
Harbor Hub and cached under `~/.cache/harbor/tasks`, so the first run needs an
authenticated session (`harbor auth login` or `HARBOR_API_KEY`) and later runs
are offline.

`-jd/--jobs-dir` is what they **read** and accepts either a single job
directory (`jobs/2026-05-05__04-10-26`) or a tree of them (`jobs`), in which
case every job is pooled into one table. `-od/--output-dir` is only where CSVs
and figures are **written**. `-e/--effort` filters on the judge's reasoning
effort, not the agent's.

```bash
# Generate Figure 1: RCA accuracy and hallucination
uv run python plot_rca_and_hallucination.py -jd JOBS_DIR -od OUTPUT_DIR -e high

# Generate Figure 4: PRCA performance by prompt difficulty
uv run python plot_rca_difficulty_bars.py -jd JOBS_DIR -od OUTPUT_DIR -e high

# Generate table of accuracy results for appendix
uv run python format_acc.py -jd JOBS_DIR -od OUTPUT_DIR -e high

# Same table stratified by prompt difficulty, plus per-difficulty bar plots
uv run python format_acc_by_granularity.py -jd JOBS_DIR -od OUTPUT_DIR -e high
```

<details>
<summary>Scoring trials that predate <code>verifier/reward-details.json</code></summary>

Older runs wrote a bare-float `verifier/reward.txt` and a summary
`verifier/details.json` that carries no rubric detail. If you also have the
`judge-*.json` files `run_llm_judge.py` produced for those trials, convert them
in place — the judge verdicts are reused as-is, so this costs no LLM calls:

```bash
uv run python backfill_verifier_outputs.py \
  --jobs-dir JOBS_DIR --scores-dir SCORES_DIR --dry-run   # preview
uv run python backfill_verifier_outputs.py \
  --jobs-dir JOBS_DIR --scores-dir SCORES_DIR
```

Trials are paired with score files by directory name. It writes `reward.json`
and `reward-details.json`, removes the stale `reward.txt` / `details.json`, and
leaves `report.md` and the run logs untouched.

</details>

<details>
<summary>Instructions for generating Figure 5: Code+telemetry vs. telemetry only</summary>

1. Write the accuracy tables to the Overleaf appendix.

2. Update the raw numbers in `plot_ablation_dumbbell.py` in the Overleaf repo and run the script.

</details>

3. Generate token and tool-usage figures:

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

## Submitting to the leaderboard

Results land on the [ORCA-bench leaderboard](https://hub.harborframework.com/datasets/orca-bench/ORCA-bench/latest?tab=leaderboard&leaderboard=orca-bench)
through a pull request. See [`leaderboard/SUBMIT.md`](./leaderboard/SUBMIT.md)
for the full walkthrough; the short version:

1. **Run and upload.** Use [`job-config.yaml`](./job-config.yaml) at the repo
   root — it pins the dataset as published on the Harbor Hub, which is what CI
   validates against. Add `--upload --public` so the trials are readable from
   the hub:

```bash
SNAPSHOT_IMAGE=orcabench/sre-otel-snapshot:data-0418-harbor-template \
    ./run_harbor_cached.sh -c job-config.yaml \
      -a <agent> -m <provider/model> \
      --ve OPENAI_API_KEY="$OPENAI_API_KEY" \
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

Accuracy is the **mean graded reward** over all trials (ORCA-bench verifiers
emit a normalized reward in `[0, 1]`, not pass/fail), reported with a standard
error. Maintainers setting the repo up for the first time should read
[`leaderboard/SETUP.md`](./leaderboard/SETUP.md).