#!/usr/bin/env bash
set -euo pipefail

# Load a telemetry snapshot and restart the OpenTelemetry Demo services.
# Builds a ready-to-mount data/ directory from raw snapshot data, matching
# the layout expected by docker-compose volume mounts.
#
# Usage:
#   ./load_snapshot.sh <data_dir> <snapshot_name> [output_dir]
#
# Arguments:
#   data_dir       Directory containing raw snapshot data (created by run_incident_schedule.py)
#   snapshot_name   Prometheus TSDB snapshot name (e.g. 20260210T190036Z-324b21f5fffc57d1),
#                   also used as the OpenSearch snapshot name (lowercased)
#   output_dir      Working directory for ephemeral data (default: "data")
#
# Example:
#   ./load_snapshot.sh data-0216 20260217T165634Z-21ead764d59c9486
#
# Snapshot data layout (input):
#   data_dir/prometheus/snapshots/<name>/    Prometheus TSDB snapshot (mounted directly)
#   data_dir/opensearch-repo-snapshots/      OpenSearch snapshot repo (mounted directly)
#
# The script:
#   1. Stops any running containers
#   2. Mounts the Prometheus snapshot and read-only config
#   3. Creates an empty opensearch-data dir in output_dir (populated by snapshot restore)
#   4. Starts services with snapshot-mode configs (read-only Prometheus, OTel Collector, Jaeger)
#   5. Restores the named OpenSearch snapshot (Jaeger traces + OTel logs).
#      Set RESTORE_NAMED_SNAPSHOT=false to skip this — useful when you intend to
#      restore a different snapshot manually (e.g. post_consolidation_<date> or
#      ${SNAPSHOT_NAME}-consolidated) into the empty cluster afterwards, and want
#      to avoid the overlap-on-existing-index error from a double restore.
#
# Note: Jaeger traces are stored in OpenSearch, so they are captured
# automatically by the OpenSearch snapshot — no separate Jaeger data needed.

if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_dir> <snapshot_name> [output_dir]"
    echo "Example: $0 data-0216 20260217T165634Z-21ead764d59c9486"
    echo ""
    echo "Available snapshots:"
    echo "  ls <data_dir>/prometheus/snapshots/"
    exit 1
fi

DATA_DIR="$1"
SNAPSHOT_NAME="$2"
DATA_LINK="${3:-data}"

# Resolve to absolute paths
if [[ "$DATA_DIR" != /* ]]; then
    DATA_DIR="$(pwd)/$DATA_DIR"
fi

# Validate data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory does not exist: $DATA_DIR"
    exit 1
fi

echo "[load_snapshot] Preparing snapshot: $SNAPSHOT_NAME"
echo "  Data dir:    $DATA_DIR"
echo "  Output dir:  $DATA_LINK"
echo ""

# Ensure output directory exists
rm -rf "$DATA_LINK"
mkdir -p "$DATA_LINK"

# ── 0. Stop services if running (in case of re-load) ────────────────────────
echo "[load_snapshot] Stopping OpenTelemetry Demo services (if running)..."
docker ps -q | xargs -r docker stop

# ── 1. Prometheus: copy snapshot data directly ──────────────────────────────
# Symlinks break when Docker creates bind mounts (Docker daemon mkdir -p
# replaces symlinks with empty dirs). Use cp -a instead.
export PROMETHEUS_MOUNT_DIR="$DATA_DIR/prometheus/snapshots/$SNAPSHOT_NAME"
# Prometheus runs as nobody (uid 65534); fix permissions so it can write.
chmod -R a+rwX "$PROMETHEUS_MOUNT_DIR"
echo "  $PROMETHEUS_MOUNT_DIR (Prometheus snapshot data)"

# ── 2. Prometheus config: read-only config (no scraping) ────────────────────
export PROMETHEUS_CONFIG_PATH="$(pwd)/prometheus-config-readonly.yaml"
echo "  $PROMETHEUS_CONFIG_PATH (Prometheus config, read-only, no scraping)"

# ── 3. OpenSearch: copy snapshot repo, create empty data dir ─────────────────
# We intentionally do NOT copy opensearch-data/ — OpenSearch starts with an
# empty node and the snapshot restore (step 6) recreates the indices.
# This avoids conflicts from pre-existing indices blocking the restore.
rm -rf "$DATA_LINK/opensearch-data"
mkdir -p "$DATA_LINK/opensearch-data"
chmod -R a+rwX "$DATA_LINK/opensearch-data"
echo "  $DATA_LINK/opensearch-data (empty — will be populated by snapshot restore)"

export OPENSEARCH_DATA_MOUNT_DIR="$(pwd)/$DATA_LINK/opensearch-data"
export OPENSEARCH_SNAPSHOTS_MOUNT_DIR="$DATA_DIR/opensearch-repo-snapshots"
chmod -R a+rwX "$DATA_DIR/opensearch-repo-snapshots"
echo "  $OPENSEARCH_SNAPSHOTS_MOUNT_DIR (OpenSearch snapshot repo)"

# ── 4. Start the OpenTelemetry Demo stack ───────────────────────────────────
echo ""
echo "[load_snapshot] Starting OpenTelemetry Demo services..."
export OPENSEARCH_READ_ONLY=true
export OTEL_COLLECTOR_CONFIG="$(pwd)/otelcol-config-snapshot.yml"
export JAEGER_CONFIG="$(pwd)/jaeger-config-snapshot.yml"
docker compose -f opentelemetry-demo/docker-compose.yml -f docker-compose.snapshot.yml up --force-recreate --remove-orphans --detach

# ── 5. Wait for services to initialise ──────────────────────────────────────
echo "[load_snapshot] Waiting for services to initialise..."
sleep 30

# ── 6. Restore OpenSearch snapshot (includes Jaeger traces + OTel logs) ────
# Always register the snapshot repository so subsequent manual restores
# (e.g. post_consolidation_<date>) can find it even when we skip the named
# snapshot restore below.
echo "[load_snapshot] Registering OpenSearch snapshot repository..."
curl -s -X PUT 'http://localhost:9200/_snapshot/scheduled_backups' \
    -H 'Content-Type: application/json' \
    -d '{"type": "fs", "settings": {"location": "/usr/share/opensearch/snapshots"}}' \
    || echo "[load_snapshot] Warning: could not register OpenSearch snapshot repo"

sleep 2

if [ "${RESTORE_NAMED_SNAPSHOT:-true}" = "true" ]; then
    # OpenSearch requires lowercase snapshot names
    OS_SNAPSHOT_NAME="${SNAPSHOT_NAME,,}"
    echo "[load_snapshot] Restoring OpenSearch snapshot: $OS_SNAPSHOT_NAME"
    # Restore the snapshot (no need to close indices — opensearch-data starts empty)
    curl -s -X POST "http://localhost:9200/_snapshot/scheduled_backups/${OS_SNAPSHOT_NAME}/_restore?wait_for_completion=true" \
        -H 'Content-Type: application/json' \
        -d '{"indices": "*", "include_global_state": false}' \
        || echo "[load_snapshot] Warning: OpenSearch snapshot restore failed (may not exist for this snapshot)"
else
    echo "[load_snapshot] RESTORE_NAMED_SNAPSHOT=false — skipping named snapshot restore."
    echo "[load_snapshot] Cluster left empty; restore your chosen snapshot manually, e.g.:"
    echo "[load_snapshot]   curl -X POST 'http://localhost:9200/_snapshot/scheduled_backups/post_consolidation_<date>/_restore?wait_for_completion=true' \\"
    echo "[load_snapshot]     -H 'Content-Type: application/json' -d '{\"indices\": \"*\", \"include_global_state\": false}'"
fi

echo ""
echo "[load_snapshot] Done. Snapshot $SNAPSHOT_NAME loaded."
