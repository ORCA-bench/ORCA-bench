# OpenTelementry Demo

We collect telemetry data using the codebase at https://github.com/open-telemetry/opentelemetry-demo with the following modifications:

1. We use OpenSearch as the storage backend for Jaeger as recommended [here](https://www.jaegertracing.io/docs/1.76/faq/#what-is-the-recommended-storage-backend). Jaeger uses date-based indices (e.g. `jaeger-main-jaeger-span-2026-02-17`) and by default only looks back 72 hours (`max_span_age`) when querying. When loading historical snapshots older than 3 days, the `max_span_age` in `jaeger-config-snapshot.yml` must be large enough to cover the age of the snapshot data, otherwise the Jaeger service list will appear empty. It is currently set to `2160h` (90 days). Note: very large values (e.g. 1 year+) cause OpenSearch's 4096-byte HTTP line limit to be exceeded, since Jaeger generates one index name per day in the query URL.

2. We also increase the memory resources for certain containers:

| Service | Original | New | Notes |
|---------|----------|-----|-------|
| `opensearch` | 1G | 2G | Prevents OOM kills |
| `llm` | 50M | 100M | Prevents OOM kills |
| `product-catalog` | 20M | 80M | Requires more memory to start |
| `product-catalog` GOMEMLIMIT | 16MiB | 64MiB | Go runtime memory limit |

TODO:
- [ ] Bump `ad` service memory from 300M to 500M in `patches/docker-compose.yml`. The Java service + OTel agent has a steady-state RSS of ~280 MiB (post-GC tenured gen ~60 MB is flat across 48h+ incarnations — no leak), so it sits at ~93% utilization and gets OOM-killed by transient spikes every ~2 days, generating a fresh `service_instance_id` UUID and breaking counter continuity in metrics like `app_ads_ad_requests_total`.
- [ ] Add `?verify=false` to the OpenSearch snapshot repo registration in both `load_snapshot.sh` and `harbor-template/environment/entrypoint.sh`, since we only need read access.

## Setup Instructions

1. Install Python dependencies:

```bash
pip install uv
uv sync --all-groups
# Setup pre-commit hooks to run every time `git commit` is called
uv run pre-commit install
```

<details>
  <summary>Additional setup instructions for data collection</summary>

Setup OpenTelemetry Demo system:

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

</details>

<details>
  <summary>Additional setup instructions for Harbor + GradientAI evaluation</summary>

1. Patch LiteLLM to support `reasoning_effort` and Anthropic `cache_control` passthrough for Gradient AI:

```bash
cp patches/litellm_gradient_ai_chat_transformation.py \
  "$(uv run python -c "import litellm; print(litellm.__path__[0])")/llms/gradient_ai/chat/transformation.py"
```

2. Patch Harbor's `Task.checksum` to ignore `environment/.trials/`. The harbor-template entrypoint stages each trial's OpenTelemetry demo + live OpenSearch data under `environment/.trials/<PROJECT_HASH>/`; without this patch, `dirhash` walks into another in-flight trial's snapshot tree and races OpenSearch's segment renames, raising `FileNotFoundError` mid-scan. The unpatched checksum is also non-deterministic for the same reason.

```bash
cp patches/harbor_models_task_task.py \
  "$(uv run python -c "import harbor.models.task.task as m; print(m.__file__)")"
```

3. Optional: Patch Harbor to support `max_tokens` for the OpenHands SDK agent:

```bash
HARBOR_AGENTS="$(uv run python -c "import harbor.agents.installed; print(harbor.agents.installed.__path__[0])")"
cp patches/harbor_openhands_sdk.py "$HARBOR_AGENTS/openhands_sdk.py"
cp patches/harbor_openhands_sdk_runner.py "$HARBOR_AGENTS/openhands_sdk_runner.py"
```

</details>

### Using DigitalOcean

1. Navigate to the page to create a droplet. (We will only need to use CPU droplets for this project.)

2. Select the region "New York".

3. Choose an image -> Marketplace -> Docker on Ubuntu 22.04.

4. Choose Basic -> Premium Intel -> 32 GB / 8 Intel CPUs (note: as of Feb 27, the pricing is $192/mo.)

5. Configure the ssh-firewall in Droplet -> Networking. This prevents ransomware attacks on the OpenSearch database.

6. Tune kernel limits for high-concurrency Docker trials. The defaults silently drop packets (ARP/conntrack overflow) and can hang SSH/VS Code once you launch many containers across multiple bridges.

```bash
# Add swap (safety net; persist in fstab)
fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Raise ARP/conntrack/inotify/FD/PID/port limits and persist
cat > /etc/sysctl.d/99-docker-scale.conf <<'EOF'
# ARP / neighbor table (prevents silent packet drops with many veth interfaces)
net.ipv4.neigh.default.gc_thresh1 = 8192
net.ipv4.neigh.default.gc_thresh2 = 32768
net.ipv4.neigh.default.gc_thresh3 = 65536
net.ipv6.neigh.default.gc_thresh1 = 8192
net.ipv6.neigh.default.gc_thresh2 = 32768
net.ipv6.neigh.default.gc_thresh3 = 65536

# Conntrack table (same silent-drop failure mode)
net.netfilter.nf_conntrack_max = 524288

# inotify (VS Code remote + file watchers)
fs.inotify.max_user_instances = 8192
fs.inotify.max_user_watches = 1048576

# File descriptors
fs.file-max = 2097152

# PIDs
kernel.pid_max = 4194304

# Ephemeral ports
net.ipv4.ip_local_port_range = 10000 65535
EOF
sysctl -p /etc/sysctl.d/99-docker-scale.conf
```

> \[!TIP\]
> If SSH/VS Code disconnects mid-run, check `dmesg -T | tail` for `neighbour: arp_cache: neighbor table overflow` or `nf_conntrack: table full` — both indicate the limits above need to go higher.

7. Enlarge Docker's IP address pool. Each Harbor environment creates its own bridge network, and Docker's built-in pool can be exhausted after a few dozen concurrent (or stale) networks, producing `Error response from daemon: all predefined address pools have been fully subnetted` on `docker compose up`. The config below provisions 1024 × /24 networks (4 × /16 carved into /24s).

```bash
# Stop running containers first — restarting docker will kill them.
cat > /etc/docker/daemon.json <<'EOF'
{
  "default-address-pools": [
    {"base": "10.200.0.0/14", "size": 24}
  ]
}
EOF
systemctl restart docker
# Verify
docker info | grep -A2 "Default Address Pools"
```

> \[!TIP\]
> If you hit the pool-exhausted error again, sweep stale resources first — Harbor environments that exit abnormally can leave behind networks tied to stopped containers:
> ```bash
> docker container prune -f --filter "label=com.docker.compose.project"
> docker network prune -f
> ```

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
<!-- # If resuming an existing Docker:
docker compose -f opentelemetry-demo/docker-compose.yml up -d
# To pause a Docker, please use the following command:
docker compose -f opentelemetry-demo/docker-compose.yml down -->

> \[!NOTE\]
> Once the images are built and containers are started you can access:
> - Web store: http://localhost:8080/
> - Grafana: http://localhost:8080/grafana/
> - Load Generator UI: http://localhost:8080/loadgen/
> - Jaeger UI: http://localhost:8080/jaeger/ui/
> - Tracetest UI: http://localhost:11633/, only when using make run-tracetesting
> - Flagd configurator UI: http://localhost:8080/feature

> \[!TIP\]
> Helpful Docker monitoring commands:
> - List the running Docker containers: `docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"`
> - Check the memory usage of all containers: `docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"`
> - Stop all containers: `docker stop $(docker ps -q)`

> \[!TIP\]
> To check the health of the telemetry data, please run:
> ```bash
> ./check_health.sh
> ```

3. Run the incident schedule using the following command, saving snapshots at regular intervals:

> \[!TIP\]
> Specifying the TZ environment variable configures the logger to print times in the local timezone.

```bash
# Option 1: Run the incident schedule immediately
TZ="America/New_York" uv run python run_incident_schedule.py \
  --schedule schedules/incident_schedule_dev_quick_1.json \
  --data-dir DATA_DIR \
  --snapshot-interval 1

# Example:
# TZ="America/New_York" uv run python run_incident_schedule.py \
#   -s schedules/incident_schedule_dev_quick.json \
#   --data-dir data-0210s \
#   --snapshot-interval 1

# Option 2: Run the script at a scheduled time using the `atd` tool
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

# Example: uv run python generate_questions.py --schedule schedules/incident_schedule_dev_single_2.json -od out-0402-3
```

7. Create Harbor tasks and share to the HuggingFace repo at https://huggingface.co/datasets/orca-bench/sre-2d-harbor-tasks:

> \[!TIP\]
> For development, you can specify the question ID using the `--qid productCatalogFailure-00` flag.

> \[!TIP\]
> To delete a HF repo from the CLI:
> ```bash
> uv run hf auth login --token HF_TOKEN
> uv run hf repo delete orca-bench/sre-2d-harbor-tasks --repo-type dataset
> ```

> \[!TIP\]
> To clear storage space, please run `docker image prune -af && docker builder prune -af`.


```bash
# NOTE: build_harbor_tasks.py writes `OUTPUT_DIR/harbor/used_snapshots.json` listing the snapshots actually referenced by tasks.
uv run python generate_task_specs.py \
  -od out-test-0503-3 -dd data-0418 -e high \
  --config configs/harbor_tasks_v2_share2w.yaml --force
uv run python generate_answers.py -od out-test-0503-3 -dd data-0418 -e high --concurrency 1600
uv run python build_harbor_tasks.py -od out-test-0503-3 -dd data-0418 --templates-dir harbor-template --force
```

8. Recover wipe-affected OpenSearch indices and build per-snapshot consolidated
   snapshots so the Harbor entrypoint can restore point-in-time-correct state
   (everything queryable before T, nothing after). The OpenSearch instance has
   been hit by `DELETE INDEX` / `DELETE BY QUERY` ransomware events that leave
   gaps in the daily indices captured by each scheduled snapshot — the runbook
   below walks the snapshot size curve to detect every wipe and replay the
   missing data.

> \[!IMPORTANT\]
> Run [**RESTORE_HISTORICAL_INDICES.md**](RESTORE_HISTORICAL_INDICES.md) phases 0–5
> first (load the stack, build `data-XXXX/restore_manifest.json`, restore peak
> snapshots, consolidate into daily indices, and persist the result as
> `post_consolidation_<date>` in the repo). `per_snapshot_consolidate.py` below
> consumes the manifest produced in phase 1.

> \[!IMPORTANT\]
> Run `per_snapshot_consolidate.py` **before** `build_and_push_snapshots.py`
> so the `-consolidated` snapshots end up baked into the Docker image. The
> Harbor entrypoint restores `${SNAPSHOT_NAME}-consolidated`, so without this
> step the image will error on snapshot restore at task startup.

```bash
# First run: build a -consolidated sibling for every entry in used_snapshots.json.
uv run python per_snapshot_consolidate.py DATA_DIR OUTPUT_DIR/harbor/used_snapshots.json

# Incremental: when a new Harbor dataset adds used-snapshot entries, fill in the
# missing -consolidated siblings without rebuilding the existing ones.
uv run python per_snapshot_consolidate.py DATA_DIR OUTPUT_DIR/harbor/used_snapshots.json --fill-missing
```

9. Build and push the Harbor Docker image with the (now consolidated) snapshot
   data baked in:

```bash
uv run python build_and_push_snapshots.py --data-dir DATA_DIR

# Example:
#   uv run python build_and_push_snapshots.py --data-dir data-0329-2
```

<details>
  <summary>Instructions for MCP version</summary>

```bash
uv run python prepare_harbor_tasks.py -od OUTPUT_DIR -dd DATA_DIR \
  --templates-dir harbor-template-mcp \
  --upload-hf --hf-repo-id orca-bench/sre-2d-mcp-harbor-tasks \

uv run python build_and_push_snapshots.py --data-dir DATA_DIR --templates-dir harbor-template-mcp
```

</details>

<details>
  <summary>Instructions for code-only version</summary>

```bash
# Build and push the code-only Docker image (once):
uv run python build_and_push_code.py

# Generate tasks:
uv run python prepare_harbor_tasks.py -od OUTPUT_DIR -dd DATA_DIR \
  --upload-hf --hf-repo-id orca-bench/sre-2d-code-only-harbor-tasks \
  --code-only --templates-dir harbor-template-code-only
```

</details>

<details>
  <summary>Instructions for verification version</summary>

```bash
uv run python prepare_harbor_tasks.py -od OUTPUT_DIR -dd DATA_DIR \
  --upload-hf --hf-repo-id orca-bench/sre-2d-verify-harbor-tasks \
  --templates-dir harbor-template-verify

uv run python build_and_push_snapshots.py --data-dir DATA_DIR
```

</details>

### Curating ground-truth symptoms

1. Load the latest snapshot from `DATA_DIR`:

> \[!IMPORTANT\]
> It takes 1-2 minutes for all the services to launch. To check that everything is set up property, please run `./check_health.sh`.

```bash
# Loads the latest snapshot (lexicographic sort works because names are timestamp-prefixed)
./load_snapshot.sh DATA_DIR "$(ls -1 DATA_DIR/prometheus/snapshots | sort | tail -n 1)"
```

To skip the named-snapshot restore and bring up an empty cluster (so you can
restore a different snapshot manually — e.g. the consolidated post-recovery
state — without hitting `cannot restore index ... because an open index with
same name already exists in the cluster`):

```bash
LATEST=$(ls -1 DATA_DIR/prometheus/snapshots | sort | tail -n 1)
RESTORE_NAMED_SNAPSHOT=false ./load_snapshot.sh DATA_DIR "$LATEST"
# Stack is up with an empty OpenSearch cluster. Restore whichever snapshot you want:
curl -X POST 'http://localhost:9200/_snapshot/scheduled_backups/post_consolidation_2026-04-26/_restore?wait_for_completion=true' \
  -H 'Content-Type: application/json' \
  -d '{"indices": "*", "include_global_state": false}'
```

2. Please following the instructions below for each telemetry type to find golden signals:

> \[!NOTE\]
> Steps marked by :robot: involve LLM assistance.

<details>
  <summary>Instructions for traces</summary>

- Programmatically obtain all traces with the tag `error=true`:

```bash
uv run python find_traces.py -dd DATA_DIR -od OUTPUT_DIR

# Example:
#   uv run python find_traces.py -dd data-0216-copy -od out-0323-2
```

- [ ] Add instructions to prompt Claude and save the output to `OUTPUT_DIR/json/{event_id}-{feature_flag}.json`.

</details>

<details>
  <summary>Instructions for logs</summary>

- Programmatically obtain all logs matching "error" keywords:

```bash
uv run python find_logs.py -dd DATA_DIR -od OUTPUT_DIR

# Example:
#   uv run python find_logs.py -dd data-0216-copy -od out-0323-2
```

- [ ] Add instructions to prompt Claude and save the output to `OUTPUT_DIR/json/{event_id}-{feature_flag}.json`.

</details>

<details>
  <summary>Instructions for metrics</summary>

- :robot: Find metrics using multiple Claude calls and plot them as a PDF for verification. Reads the JSON rubric for the given feature flag from `rubrics/json/` to obtain the incident description, mechanism, and golden-signal log/trace entries, which are injected into each sub-agent prompt so the agent can cross-reference log and trace evidence when identifying relevant metrics:

```bash
uv run python find_metrics_family.py -f FEATURE_FLAG -t UTC_TIME -od OUTPUT_DIR -n PARALLEL_CALLS

# Example:
#   uv run python find_metrics_family.py -f productCatalogFailure \
#     -t "2026-04-02 06:25:59 UTC" -od metrics-0401 -n 3
```

- Review the per-run JSON files and PDF plots manually. Fix misclassified types (e.g., `"created (fixed after human verification)"`). Mark dubious entries as `inconclusive`.

- :robot: Detect onset timing for each metric using Grafana/Prometheus (requires snapshot loaded via `./load_snapshot.sh`):

```bash
uv run python onset_metrics.py -f FEATURE_FLAG -t UTC_TIME -od OUTPUT_DIR \
  -dir INPUT_DIR -m MODEL -e EFFORT

# Example:
#   uv run python onset_metrics.py -f productCatalogFailure \
#     -t "2026-04-02 06:25:59 UTC" -od metrics-0401/metrics_onset \
#     -dir metrics-0401/metrics -m opus -e max
```

- Review onset JSON files. Verify `onset_utc` aligns with PDF plots and that `created` metrics truly didn't exist pre-incident.

- :robot: Group into metric families and annotate signal layers (`root_cause`, `propagation`, `symptom`, `meta`):

```bash
uv run python label_metrics.py -f FEATURE_FLAG -od OUTPUT_DIR \
  -dir INPUT_DIR -m MODEL -e EFFORT

# Example:
#   uv run python label_metrics.py -f productCatalogFailure \
#     -od metrics-0401 -dir metrics-0401/metrics_onset -m opus -e max
```

</details>

<details>
  <summary>Instructions for frontend symptoms</summary>

<!-- See Claude Code conversation `dev-1-questions` -->

</details>

## Reproducing evaluation using Harbor

- [ ] Have the entrypoint write the MCP config file (`~/.claude.json`) at runtime, similar to how it writes `/tmp/env-ports`. This would allow agents with native MCP support (Claude Code, Cline, etc.) to automatically connect to the MCP servers with dynamic ports.

1. Please go to https://cloud.digitalocean.com/gen-ai/model-access-keys -> Create model access key. Then configure the following environment variables:

<details>
  <summary>Instructions for running Codex</summary>

```bash
export OPENAI_API_KEY=$MODEL_ACCESS_KEY
export OPENAI_BASE_URL=https://inference.do-ai.run/v1
```

</details>

<details>
  <summary>Instructions for running Terminus-2 with models on DigitalOcean (e.g., GPT and Claude)</summary>

See https://docs.digitalocean.com/products/gradient-ai-platform/details/models/ for the full list of supported models.

```bash
export GRADIENT_AI_API_KEY=$MODEL_ACCESS_KEY
```

</details>

<details>
  <summary>Instructions for running OpenHands with models on DigitalOcean (e.g., GPT and Claude)</summary>

See https://docs.digitalocean.com/products/gradient-ai-platform/details/models/ for the full list of supported models.

```bash
export LLM_API_KEY=$MODEL_ACCESS_KEY
```

</details>

2. Launch the Harbor tasks using the following command:

<!-- > \[!TIP\]
> If your home volume is small, point Harbor's cache at a mounted disk via a symlink (one-time setup):
> ```bash
> mkdir -p /mnt/volume_nyc2_XXXXX/harbor-cache
> ln -s /mnt/volume_nyc2_XXXXX/harbor-cache ~/.cache/harbor
> ```
> Note: with a symlinked cache, use `find ~/.cache/harbor/ -mindepth 1 -delete` instead of `uv run harbor cache clean`. -->

```bash
# Stop all containers from previous runs to avoid Docker container naming conflicts
docker stop $(docker ps -q)
# Clear harbor cache
uv run harbor cache clean

# Launch jobs (note: this script pre-pulls the Docker image with the snapshot to /root/.cache/sre-snapshot-cache so that we can bind-mount it to all tasks).
# Each config pairs one task dataset with the snapshot image whose `/app/` matches its environment, so run them on separate machines for parallel sweeps:
#   Machine A — full template
SNAPSHOT_IMAGE=orcabench/sre-otel-snapshot:data-0418-harbor-template \
    ./run_harbor_cached.sh -c configs/harbor_job_orca_bench.yaml
#   Machine B — telemetry-only
SNAPSHOT_IMAGE=orcabench/sre-otel-snapshot:data-0418-harbor-template-telemetry-only \
    ./run_harbor_cached.sh -c configs/harbor_job_orca_bench_telemetry_only.yaml
#   Machine C — code-only
SNAPSHOT_IMAGE=orcabench/sre-otel-snapshot:code-only \
    ./run_harbor_cached.sh -c configs/harbor_job_orca_bench_code_only.yaml
```

3. Export predictions from Harbor jobs:

> [!TIP]
> To share traces to HuggingFace:
> ```bash
> uv run harbor traces export --path JOBS_DIR --push --repo HF_REPO_ID --recursive --episodes last
> ```

```bash
uv run harbor-export JOBS_DIR -od OUTPUT_DIR \
  --registry-url https://huggingface.co/datasets/.../registry.json
```

4. Obtain LLM judge scores using GPT 5.4 (high):

> \[!NOTE\]
> You may get errors like `2026-05-18T00:09:35+0000 - ERROR - Trial d6-big-cartfailure-on-univ00-uni__EHeBeZo (d6-big-cartfailure-on-univ00-universal_ttd480m_range30m_off+60m) failed: Unterminated string starting at: line 1 column 2297 (char 2296)`. Please re-run the following command until there are no more errors.

```bash
uv run python run_llm_judge.py -od OUTPUT_DIR -e high -bs 1000 --concurrency 1000
```

<details>
  <summary>Obtain scores on human eval sample only</summary>

```bash
uv run python run_llm_judge.py -od out-0517 -e high -bs 1600 --concurrency 1600 --sampled-tasks-file human-eval-sample/output/sampled_tasks.json
```

</details>

5. Validate the LLM judge against human-verified scores on the human-eval sample. Each annotator (e.g. KC, AG) maintains their own `human_verified_scores*.json` file of independently re-scored task--model pairs:

```bash
# (a) Pairwise inter-annotator agreement + (b) LLM judge vs. average human score,
# broken down by difficulty (Spearman rho, quadratic-weighted Cohen's kappa)
uv run python compute_human_agreement.py \
  --scores-a human_verified_scores.json \
  --scores-b human_verified_scores_ag.json \
  -od out-0522 -e high

# Scatter grid (rows=difficulty, cols=model) of human vs. LLM judge scores,
# one PDF per annotator's score file
uv run python plot_human_vs_llm_scatter.py -od out-0522 -e high --human-scores human_verified_scores_ag.json
uv run python plot_human_vs_llm_scatter.py -od out-0522 -e high --human-scores human_verified_scores.json
```

6. Format accuracy results:

```bash
# Generate Figure 1: RCA accuracy and hallucination
uv run python plot_rca_and_hallucination.py -od out-0619-3 -e high

# Generate Figure 4: PRCA performance by prompt difficulty
uv run python plot_rca_difficulty_bars.py -od out-0619-3 -e high

# Generate table of accuracy results for appendix
uv run python format_acc.py -od out-0510 -e high
```

<details>
<summary>Instructions for generating Figure 5: Code+telemetry vs. telemetry only</summary>

1. Write the accuracy tables to the Overleaf appendix.

2. Update the raw numbers in `plot_ablation_dumbbell.py` in the Overleaf repo and run the script.

</details>

7. Generate token and tool-usage figures:

* Pre-process Terminus-2 trajectories using the following command:

```bash
# Step 1 (recover_outputs) parses asciinema recording.cast to sidestep the
# terminus-2 40-row pane truncation that inflates apparent error rates 4-5x
# in trajectory.json. Steps 2-3 then judge + aggregate per-agent outcomes.
uv run python recover_outputs.py        -od out-0619-3 -jd jobs-sub

# Generate CSV of tool calls per trial
uv run python format_tool_calls.py -od out-0619-3 -jd jobs-sub
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
uv run python format_markov_chain.py -od out-0619-3 -jd jobs-sub \
    --models opus-4.7 glm-5 --min-prob 0.10 --fig-name markov_chain.pdf

# Appendix figure: Sonnet 4.6, GPT-5.5, DeepSeek-V4-Pro
uv run python format_markov_chain.py -od out-0619-3 -jd jobs-sub \
    --models 4.6-sonnet gpt-5.5 deepseek-v4-pro --min-prob 0.10 --fig-name markov_chain_appendix.pdf
```

* Generate plot of token and tool-calling efficiency:

```bash
# Classify telemetry calls into success, empty, error
uv run python classify_telemetry_calls.py -od out-0518-sample100 --concurrency 800 -e high -bs 100 -bn 1
# Classify telemetry calls into unique or redundant
uv run python classify_redundant_calls.py -od out-0518-sample100 --concurrency 800 -e high -bs 100 -bn 1

# Generate Figure 7: Token & tool-calling efficiency
uv run python plot_efficiency.py
```

## Generating Solutions

As a sanity check, `run_solve.py` generates an incident report from the rubric that should achieve close to perfect score using our LLM judge scoring scheme.

1. Please run the following script to generate the solution incident reports:

```bash
# Use format_rubric alone
uv run python run_solve.py -od share-8h

# Use format_rubric and GPT 5.4 (high) to convert the formatted rubric into the style of a postmortem: 1. Summary, 2. Timeline, 3. 5 Whys (with evidence), 4. Remediation
uv run python run_solve.py -od share-8h -m openai-gpt-5.4
```

2. Then export the solutions to a JSON file that mirrors the output of `harbor-export`.

```bash
uv run python export_solutions.py -od share-8h
```

3. Finally, launch the LLM judge script to score using GPT 5.4 (high):

```bash
uv run python run_llm_judge.py -od share-8h
```
