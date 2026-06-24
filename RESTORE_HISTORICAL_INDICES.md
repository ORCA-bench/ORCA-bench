# Restore Historical OpenSearch Indices (Full-Fidelity)

## Background

A ransomware attacker (`bc1q38rjul6gdamfflf6p4ukz0ymtvfgfv2j9saf6r`, contact `wendy.etabw@gmx.com`) repeatedly issued destructive calls against an exposed OpenSearch instance, wiping the contents of daily indices (`otel-logs-YYYY-MM-DD`, `jaeger-main-jaeger-span-YYYY-MM-DD`, `jaeger-main-jaeger-service-YYYY-MM-DD`) every few hours from 2026-04-19 onward. After each wipe, the otel-collector continued writing into the same index — which masked the loss in real time but left earlier hours of that day permanently gone from the live cluster.

Two attack patterns can produce this loss:

1. **`DELETE INDEX`** — the index is dropped entirely; the collector recreates it with a *new* UUID and continues writing. Each (date, type) pair then has multiple UUID "lifetimes" — the periods between successive deletes — and earlier `MetadataDeleteIndexService` events appear in the OpenSearch logs.
2. **`DELETE BY QUERY`** — older docs are deleted in place; the index UUID is *unchanged*, only the contents shrink. No `MetadataDeleteIndexService` event is logged. This is the pattern observed in the `data-0418` capture: snapshot size for each affected daily index grows steadily, then collapses sharply to a few MB, then grows again, sometimes repeatedly within the same day.

In both cases the data is preserved in the OpenSearch fs snapshot repository (`data-XXXX/opensearch-repo-snapshots/`, ~thousands of snapshots taken every 5 minutes since 2026-04-19 by `run_incident_schedule.py`). Each daily index has one or more recoverable "lifetimes," each captured in a peak snapshot just before the next wipe — *regardless* of which attack pattern was used. The recovery script below detects both by walking the per-snapshot size curve.

This document is the runbook for reconstructing full daily coverage on a **local, isolated copy** of the data (not the live cluster). It restores each lifetime as a separate physical index, then unions them under an alias so queries against the original index name see all chunks transparently.

## Prerequisites

- A local copy of `data-XXXX/` containing both `opensearch-data/` (can be empty) and `opensearch-repo-snapshots/` (the source of truth).
- Docker and the `opentelemetry-demo` repo set up locally per the main [README.md](README.md).
- Python 3.11+ for the recon script. No extra packages.
- ~5 GB free disk for restored historical indices on top of any existing data.

## Phase 0 — Start a local stack pointed at the data copy

Use the existing snapshot-loading workflow to bring up an isolated stack with the latest snapshot. This gives you today's index live, and we'll layer historical indices on top.

```bash
cd examples/otel-demo

# Pick the latest snapshot to bootstrap with
LATEST=$(ls -1 data-XXXX/prometheus/snapshots | sort | tail -n 1)

# Brings up the full stack with read-only Prometheus + restored OpenSearch state
./load_snapshot.sh data-XXXX "$LATEST"

# Wait for opensearch healthy (1-2 min)
./check_health.sh
```

After this, `localhost:9200` is your local cluster. It contains today's daily indices restored from the latest snapshot. We'll add the missing historical days next.

If you'd rather start fully empty (skip the latest snapshot), edit `load_snapshot.sh` to skip the `_restore` step in section 6, or use a fresh OpenSearch container with `OPENSEARCH_DATA_MOUNT_DIR=$(mktemp -d)` and `OPENSEARCH_SNAPSHOTS_MOUNT_DIR=$(pwd)/data-XXXX/opensearch-repo-snapshots`.

## Phase 1 — Build the restoration manifest

The script `build_restore_manifest.py` (already in this directory) does no writes — just produces a manifest file describing which snapshots to restore.

The detection strategy: for each daily index, fetch the snapshotted size at every snapshot via `_snapshot/REPO/SNAP/_status`, then walk the curve. Whenever the running peak exceeds the per-type `MIN_PEAK_BYTES` threshold and the next snapshot's size collapses below `DROP_RATIO × peak`, record the peak snapshot as a recoverable lifetime end. This catches both `DELETE INDEX` (size drops to 0 because the index is gone) and `DELETE BY QUERY` (size drops sharply but the index persists). The final snapshot in the curve is also emitted as a lifetime so the post-final-wipe state is preserved.

Per-type thresholds matter: `otel-logs` and `jaeger-main-jaeger-span` are MB-scale (set to 5 MB minimum peak); `jaeger-main-jaeger-service` carries only ~100 service registrations and tops out around 50 KB (set to 5 KB). A single global threshold either misses small wipes on the service indices or over-fires on quiet hours of the log/span indices.

Run:

```bash
uv run python build_restore_manifest.py data-XXXX
```

Inspect `data-XXXX/restore_manifest.json` and the per-index lifetime counts in stdout. The scan fetches `_status` for every snapshot and takes ~3–5 minutes for thousand-snapshot repos with the default `max_workers=8`. Expect 1–5 lifetimes per affected daily index. Tune `DROP_RATIO` and `MIN_PEAK_BYTES_BY_TYPE` in the script if the heuristic over- or under-fires for your data — for `data-0418`, lowering the otel-logs/jaeger-span threshold from 20 MB to 5 MB caught 9 sub-threshold wipes that the original setting silently missed.

## Phase 2 — Disk space pre-flight

```bash
df -h .
du -sh data-XXXX/opensearch-data
# Total restored size will be roughly 100-200% of opensearch-repo-snapshots minus
# duplicated segments. With 542G free and snapshots at ~5G, this is a sanity check.
```

If free space < 5 GB, stop here — restoration would risk filling the disk.

## Phase 3 — Restore loop

`restore_from_manifest.sh` (in this directory) reads the manifest and `_restore`s each peak snapshot under a renamed `*-lifetime{N}` index, leaving the original daily indices alone. The script is idempotent — re-running skips already-restored target indices, so it's safe to interrupt and resume.

Run:

```bash
./restore_from_manifest.sh data-XXXX/restore_manifest.json
```

Each restore takes ~1–3 seconds, so ~50 entries → ~2 minutes. After this phase, queries against `otel-logs-2026-04-21*` (wildcard) hit the live daily index *plus* every restored lifetime — full coverage at the cost of polluting the wildcard pattern.

## Phase 4 — Consolidate lifetimes into the live daily indices

The wildcard option above leaves Jaeger and any other consumer that hard-codes the daily index name (`jaeger-main-jaeger-span-2026-04-21`) blind to the recovered data. To fix that without aliases (which would conflict with the live daily index name — see gotchas), reindex every `*-lifetime{N}` document into its corresponding live daily index. Span/log doc IDs are preserved, so duplicates from overlapping lifetimes overwrite cleanly and re-runs are idempotent.

`consolidate_lifetimes.py` (in this directory) does this for every `*-lifetime*` and `*-extra*` index in the cluster, async via the OpenSearch tasks API. For a 6-day window with ~3.9M spans + ~2.8M logs, the full reindex takes ~25–30 minutes against the local stack with `Xmx400m`.

Run:

```bash
uv run python consolidate_lifetimes.py
```

Now the Jaeger HTTP API (`/api/services`, `/api/traces?service=...`) sees the full recovered union without any alias dance. Queries against the bare daily index name return the right counts.

## Phase 5 — Persist consolidation as a new snapshot

The live `data/opensearch-data/` directory holds the consolidated state, but `load_snapshot.sh` and the Harbor entrypoint both wipe it on bootstrap and re-`_restore` from the snapshot repo. To make consolidation survive restarts, take a new snapshot in the same repo:

```bash
curl -sS -X PUT 'http://localhost:9200/_snapshot/scheduled_backups/post_consolidation_2026-04-26?wait_for_completion=true' \
  -H 'Content-Type: application/json' \
  -d '{
    "indices": "otel-logs-*,jaeger-main-*",
    "include_global_state": false,
    "metadata": {"purpose": "consolidated state after wipe recovery"}
  }'
```

This adds ~3–5 GB to the repo. The snapshot includes both the consolidated daily indices (net-new segments from reindex) *and* the `*-lifetime*` / `*-extra*` indices (which are mostly dedup'd because their segments came from existing snapshots in the same repo). Including the lifetime indices is "belt-and-suspenders" — it preserves them as named entities for later re-derivation work (e.g. per-snapshot consolidation).

After this, anyone who restores `post_consolidation_2026-04-26` instead of the original incident-time snapshot gets the full consolidated state in one shot. The Harbor entrypoint can be patched to do this:

```bash
# In entrypoint.sh, after the existing _restore call:
curl -sS -X POST "http://opensearch:9200/_snapshot/scheduled_backups/post_consolidation_2026-04-26/_restore?wait_for_completion=true" \
    -H 'Content-Type: application/json' \
    -d '{"indices": "*", "include_global_state": false}'
```

(Caveat: this loses point-in-time semantics — agents at any incident time will see all data through the snapshot date. For per-snapshot consolidation that preserves the wall-clock view, see the comments at the top of `consolidate_lifetimes.py` and the lifetime metadata in `restore_manifest.json`.)

## Phase 6 — Verify

```bash
# 1. Lifetime + extra indices exist
curl -s 'localhost:9200/_cat/indices?v' | grep -cE 'lifetime|extra'

# 2. Live daily indices have grown to include recovered data
curl -s 'localhost:9200/_cat/indices/otel-logs-2026-04-21?h=docs.count,store.size'
# Expected: ~500K+ docs / 200+ MB (was ~50K / 20 MB before consolidation)

# 3. Sample a window that fell inside a wipe gap
curl -s -X POST 'http://localhost:9200/otel-logs-2026-04-21/_search' \
  -H 'Content-Type: application/json' \
  -d '{"size": 0, "query": {"range": {"@timestamp": {"gte": "2026-04-21T05:00:00Z", "lte": "2026-04-21T06:00:00Z"}}}}'
# Expected: ~20K hits (was 0 before consolidation)

# 4. Jaeger HTTP API returns the full service set
curl -s 'http://localhost:8080/jaeger/ui/api/services' | python3 -c "import json,sys; print(len(json.load(sys.stdin)['data']))"
# Expected: 18-19 (was 5-8 before consolidation)
```

## Edge cases & gotchas

**Empty lifetimes / wipe-window gaps.** A wipe doesn't happen instantaneously — `DELETE BY QUERY` against a wildcard takes minutes, and any snapshots taken during that window will already show partial loss. The peak snapshot the recon picks is the *last good one before* the drop, so writes that arrive between the peak and the wipe completing are not captured. Expect a small (≤1 hour) gap around each detected wipe, e.g. for `otel-logs-2026-04-21` the lifetime1 snapshot at 12:51 UTC ends just before the 13:00 UTC wipe and lifetime2 doesn't pick up until 14:00 UTC, leaving 13:00–14:00 UTC unrecoverable. Coverage is "best-effort full" rather than "guaranteed every byte."

**Alias-vs-index name collision.** OpenSearch forbids an alias whose name matches an existing index. If you bootstrapped Phase 0 from the latest snapshot, the live cluster already contains daily indices like `otel-logs-2026-04-21` — and `create_aliases.py` will fail to bind the alias name `otel-logs-2026-04-21` over the lifetime indices. Resolve by deleting the live daily indices *before* running Phase 4: their content is identical to the final lifetime entry in the manifest, which has already been restored as e.g. `otel-logs-2026-04-21-lifetime3`. Verify the duplication first (`docs.count` should match), then `curl -X DELETE` each live daily index. Alternatively, skip Phase 4 entirely and query via wildcard (`otel-logs-2026-04-21*`) to get the same union without any destructive step.

**Tuning the wipe detector.** `DROP_RATIO=0.20` and `MIN_PEAK_BYTES=20 MB` are heuristics, not invariants. If the daily traffic is too low to ever cross 20 MB, raise `MIN_PEAK_BYTES` or wipes won't be detected. If a legitimate force-merge drops on-disk size to <20% of pre-merge size, the detector will mistake it for a wipe and produce a redundant lifetime — harmless (you just restore the same data twice under different `-lifetimeN` names) but worth knowing.

**Heap pressure.** OpenSearch is configured with `Xmx400m`. With ~100 extra small indices opened, JVM heap for shard metadata grows, but should stay well under 400 MB since restored historical shards are small (minutes to hours of data each). If you hit heap warnings, close older indices not in active use:
```bash
curl -X POST 'localhost:9200/otel-logs-2026-04-19/_close'
# Reopen on demand:
curl -X POST 'localhost:9200/otel-logs-2026-04-19/_open'
```

**Jaeger UI missing data despite alias.** Jaeger queries OpenSearch with a date-prefix pattern derived from `rollover_frequency: day` ([opensearch-config.yml](opentelemetry-demo/src/jaeger/opensearch-config.yml)). The alias should resolve transparently, but if it doesn't, re-check by hitting OpenSearch directly with the index name Jaeger uses — view Jaeger's logs for the actual query URL. As a fallback, point Jaeger at a wildcard pattern.

**Resuming after interrupt.** Both `restore_from_manifest.sh` and `create_aliases.py` are idempotent. The restore script skips existing target indices; the alias script just reapplies the same `add` actions (no-op if already present).

**Cleanup.** To roll back the entire restoration:
```bash
# Drop all lifetime indices (also removes their alias bindings)
curl -X DELETE 'localhost:9200/*-lifetime*'
```

## Why aliases instead of `-rN` suffixes only

OpenSearch aliases let multiple physical indices appear as one logical index. Queries against the alias name (or a wildcard matching it) hit all members and merge results — exactly what we need for time-range queries that should fan out across all lifetime chunks of a day. Without aliases, every Grafana/Jaeger panel would need its index pattern reconfigured to match `*-lifetime*`, which would also pollute results from days that weren't fragmented. Aliases keep the original-name interface intact while transparently unioning the chunks.
