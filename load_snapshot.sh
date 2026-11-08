#!/usr/bin/env bash
set -euo pipefail

# Load a telemetry snapshot locally and bring up the OpenTelemetry Demo stack.
#
# This is a thin wrapper around harbor-template/environment/entrypoint.sh --
# the same script the Harbor task image runs -- so the local and in-trial code
# paths cannot drift. Everything host-specific is passed as an environment
# override; see the "Host/container switches" block at the top of the
# entrypoint for the full list and their container defaults.
#
# Usage:
#   ./load_snapshot.sh <source> <snapshot_name> [stage_dir]
#
# Arguments:
#   source          Either
#                     - a raw data directory from run_incident_schedule.py
#                       (contains prometheus/ and opensearch-repo-snapshots/),
#                       e.g. data-0418; or
#                     - a staged snapshot cache from stage_snapshot_cache.sh
#                       (contains opentelemetry-demo/ and data/), e.g.
#                       /root/.cache/sre-snapshot-cache/<image-id>.
#                   The mode is detected from the layout.
#   snapshot_name   Prometheus TSDB snapshot name, also used (lowercased) as
#                   the OpenSearch snapshot name, e.g.
#                   20260424T192542Z-64df31d58b702c09
#   stage_dir       Working directory for ephemeral per-run data
#                   (default: "data"). Gitignored; safe to delete.
#
# Examples:
#   # From locally collected data
#   ./load_snapshot.sh data-0418 "$(ls -1 data-0418/prometheus/snapshots | sort | tail -n 1)"
#
#   # From the published image's staged cache (same data the leaderboard uses)
#   IMAGE=orcabench/sre-otel-snapshot:data-0418-harbor-template-v2
#   CACHE="$(./stage_snapshot_cache.sh --image "$IMAGE" --cache-root /root/.cache/sre-snapshot-cache)"
#   ./load_snapshot.sh "$CACHE" 20260424T192542Z-64df31d58b702c09
#
# Useful overrides (all read by the entrypoint):
#   COMPOSE_SERVICES="prometheus opensearch"
#                   Bring up only those services instead of the full stack.
#                   Much faster when you only need to query the backends.
#   RESTORE_SNAPSHOT=false
#                   Leave OpenSearch empty so you can restore a different
#                   snapshot by hand afterwards (e.g. post_consolidation_<date>)
#                   without the overlap-on-existing-index error from a double
#                   restore. Replaces the old RESTORE_NAMED_SNAPSHOT=false.
#   RESTORE_SUFFIX=""
#                   Restore the bare snapshot name rather than the
#                   "-consolidated" variant.
#
# Unlike earlier versions, this script does NOT stop every running container.
# Trials and a local stack can coexist; tear this one down explicitly with
#   COMPOSE_PROJECT_NAME=otel-local docker compose \
#     -f <stage_dir>/context/.trials/*/opentelemetry-demo/docker-compose-base.yml down

if [ $# -lt 2 ]; then
    echo "Usage: $0 <source> <snapshot_name> [stage_dir]" >&2
    echo "Example: $0 data-0418 20260424T192542Z-64df31d58b702c09" >&2
    echo "" >&2
    echo "Available snapshots:" >&2
    echo "  ls <source>/prometheus/snapshots/       # raw data dir" >&2
    echo "  ls <source>/data/prometheus/snapshots/  # staged cache" >&2
    exit 1
fi

SOURCE="$1"
SNAPSHOT_NAME="$2"
STAGE_DIR="${3:-data}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRYPOINT="$REPO_ROOT/harbor-template/environment/entrypoint.sh"
[ -x "$ENTRYPOINT" ] || { echo "Error: entrypoint not found or not executable: $ENTRYPOINT" >&2; exit 1; }

# Resolve to absolute paths -- the entrypoint hands these to the host Docker
# daemon as bind-mount sources, which must be absolute.
[[ "$SOURCE" == /* ]]    || SOURCE="$(pwd)/$SOURCE"
[[ "$STAGE_DIR" == /* ]] || STAGE_DIR="$(pwd)/$STAGE_DIR"
[ -d "$SOURCE" ] || { echo "Error: source directory does not exist: $SOURCE" >&2; exit 1; }

# ── 1. Work out which kind of source this is ────────────────────────────────
if [ -d "$SOURCE/opentelemetry-demo" ] && [ -d "$SOURCE/data" ]; then
    MODE=cache
    CACHE_DIR="$SOURCE"
elif [ -d "$SOURCE/prometheus/snapshots" ]; then
    MODE=raw
    CACHE_DIR="$STAGE_DIR/cache"
else
    echo "Error: $SOURCE looks like neither a raw data dir (needs prometheus/snapshots/)" >&2
    echo "       nor a staged cache (needs opentelemetry-demo/ and data/)." >&2
    exit 1
fi

echo "[load_snapshot] Snapshot: $SNAPSHOT_NAME"
echo "[load_snapshot] Source:   $SOURCE ($MODE)"
echo "[load_snapshot] Stage:    $STAGE_DIR"

# ── 2. In raw mode, assemble a cache-shaped tree ────────────────────────────
# The entrypoint expects $SNAPSHOT_CACHE_HOST_DIR to hold the otel-demo source,
# the four snapshot-mode configs, and a data/ directory. A raw data dir already
# has exactly the data/ layout, so point at it with a symlink rather than
# copying 55 GB -- Docker resolves symlinked components in bind-mount sources,
# and the only bind out of here is opensearch's repo mount.
#
# opentelemetry-demo must be a real directory: the entrypoint does
# `cp -a "$CACHE/opentelemetry-demo" "$OTEL_DIR"`, and a symlink would be
# preserved as a symlink, making the per-trial `mkdir -p "$OTEL_DIR/data"`
# write straight into the submodule. It is only ~13 MB.
if [ "$MODE" = raw ]; then
    echo "[load_snapshot] Assembling cache-shaped tree at $CACHE_DIR ..."
    rm -rf "$CACHE_DIR"
    mkdir -p "$CACHE_DIR"
    ln -s "$SOURCE" "$CACHE_DIR/data"
    cp -a "$REPO_ROOT/opentelemetry-demo" "$CACHE_DIR/opentelemetry-demo"
    for f in docker-compose-base.yml docker-compose.snapshot.yml \
             otelcol-config-snapshot.yml jaeger-config-snapshot.yml; do
        [ -f "$REPO_ROOT/$f" ] || { echo "Error: missing $REPO_ROOT/$f" >&2; exit 1; }
        cp -f "$REPO_ROOT/$f" "$CACHE_DIR/$f"
    done
fi

[ -d "$CACHE_DIR/data/prometheus/snapshots/$SNAPSHOT_NAME" ] || {
    echo "Error: snapshot not found: $CACHE_DIR/data/prometheus/snapshots/$SNAPSHOT_NAME" >&2
    exit 1
}

# ── 3. Clear the previous run's per-trial tree ──────────────────────────────
# The entrypoint reuses $OTEL_DIR when it already exists and skips the
# Prometheus copy when the destination is non-empty. Locally CONTEXT_DIR and
# the hostname are both stable, so PROJECT_HASH is too -- without this, a
# second run with a different snapshot name would silently keep serving the
# first snapshot's TSDB.
CONTEXT_DIR="$STAGE_DIR/context"
mkdir -p "$CONTEXT_DIR"
if [ -d "$CONTEXT_DIR/.trials" ]; then
    echo "[load_snapshot] Clearing previous per-trial tree under $CONTEXT_DIR/.trials ..."
    rm -rf "$CONTEXT_DIR/.trials"
fi

# ── 4. Hand off to the shared entrypoint ────────────────────────────────────
# Fixed host ports so the documented localhost URLs keep working: the README,
# check_health.sh, find_traces.py (:8080) and find_logs.py (:9200) all assume
# them. The entrypoint skips its own port allocation when all four are set.
export ENVOY_HOST_PORT="${ENVOY_HOST_PORT:-8080}"
export ENVOY_ADMIN_HOST_PORT="${ENVOY_ADMIN_HOST_PORT:-8081}"
export PROMETHEUS_HOST_PORT="${PROMETHEUS_HOST_PORT:-9090}"
export OPENSEARCH_HOST_PORT="${OPENSEARCH_HOST_PORT:-9200}"

# Pick a readiness probe that is actually reachable. Grafana is the right
# signal for the full stack, but a COMPOSE_SERVICES subset without
# frontend-proxy would just burn the entrypoint's 120s timeout, so fall back
# to OpenSearch. An explicitly set READY_URL always wins, including an empty
# one, which skips the wait.
if [ -z "${READY_URL+x}" ]; then
    case " ${COMPOSE_SERVICES:-} " in
        "  ") READY_URL="http://localhost:${ENVOY_HOST_PORT}/grafana/api/health" ;;
        *" frontend-proxy "*) READY_URL="http://localhost:${ENVOY_HOST_PORT}/grafana/api/health" ;;
        *) READY_URL="http://localhost:${OPENSEARCH_HOST_PORT}/_cluster/health" ;;
    esac
fi
export READY_URL
echo "[load_snapshot] Readiness probe: ${READY_URL:-<skipped>}"

exec env \
    SNAPSHOT_NAME="$SNAPSHOT_NAME" \
    CONTEXT_DIR="$CONTEXT_DIR" \
    SNAPSHOT_CACHE_HOST_DIR="$CACHE_DIR" \
    COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-otel-local}" \
    ENTRYPOINT_LOG="${ENTRYPOINT_LOG:-$STAGE_DIR/entrypoint.log}" \
    PREP_AGENT_FILES=false \
    JOIN_COMPOSE_NETWORK=false \
    CLEANUP_ON_EXIT=false \
    REQUIRE_RESTORE="${REQUIRE_RESTORE:-true}" \
    GRAFANA_URL="http://localhost:${ENVOY_HOST_PORT}/grafana" \
    JAEGER_URL="http://localhost:${ENVOY_HOST_PORT}/jaeger/ui" \
    OPENSEARCH_URL="http://localhost:${OPENSEARCH_HOST_PORT}" \
    PROMETHEUS_URL="http://localhost:${PROMETHEUS_HOST_PORT}" \
    "$ENTRYPOINT"
