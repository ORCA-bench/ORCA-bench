#!/usr/bin/env bash
set -euo pipefail

# Harbor container entrypoint: start the OpenTelemetry Demo services
# with pre-built snapshot data and hand off to the agent.
#
# The base Docker image (built by build_and_push_snapshots.py) bakes in
# the otel-demo source, snapshot data, and docker-compose-base.yml at
# /app/.  At startup this entrypoint copies them to CONTEXT_DIR (a
# host-visible bind mount) so the host Docker daemon can access them
# for nested compose volume mounts and builds.
#
# Environment variables:
#   SNAPSHOT_NAME             - Prometheus hash used as canonical snapshot name for all backends
#   CONTEXT_DIR               - host path to the environment directory (bind-mounted)
#   SNAPSHOT_CACHE_HOST_DIR   - host path to the staged snapshot cache (run_harbor_cached.sh
#                               populates and bind-mounts this); per-trial trees are hardlinked
#                               from here so it must share a filesystem with CONTEXT_DIR

SNAPSHOT_NAME="${SNAPSHOT_NAME:?SNAPSHOT_NAME must be set}"
CONTEXT_DIR="${CONTEXT_DIR:?CONTEXT_DIR must be set}"
# Re-export so the inner `docker compose` invocation (and the otel-demo services
# it starts) see SNAPSHOT_CACHE_HOST_DIR for ${SNAPSHOT_CACHE_HOST_DIR} substitution
# in docker-compose-base.yml (specifically opensearch's repo bind-mount source).
export SNAPSHOT_CACHE_HOST_DIR="${SNAPSHOT_CACHE_HOST_DIR:?SNAPSHOT_CACHE_HOST_DIR must be set; run via run_harbor_cached.sh}"

# ── 0. Tee all output to a persistent log file ──────────────────────────────
ENTRYPOINT_LOG="/logs/agent/entrypoint.log"
mkdir -p "$(dirname "$ENTRYPOINT_LOG")"
exec > >(tee -a "$ENTRYPOINT_LOG") 2>&1

# Pre-create agent-writable files so the non-root `agent` user can write them.
# Harbor's OracleAgent pre-touches log files from the host as root (mode 644),
# which blocks the agent from writing; touch(existing) only bumps mtime, so
# the 666 mode sticks. /app is root-owned; pre-create report.md likewise.
install -m 666 /dev/null /logs/agent/oracle.txt
install -m 666 /dev/null /logs/agent/exit-code.txt
install -m 666 /dev/null /app/report.md

# ── 0b. Derive a unique project name for Docker Compose isolation ────────────
# CONTEXT_DIR looks like: .../tasks/<taskID>/<taskName>/environment
# Hash the (taskID, taskName) tuple so the resulting Docker project, network,
# and container names do not leak task semantics (e.g. lag/availability) into
# anything the evaluating agent might inspect.
TASK_NAME="$(basename "$(dirname "$CONTEXT_DIR")")"
TASK_ID="$(basename "$(dirname "$(dirname "$CONTEXT_DIR")")")"
# HOST_AGENT_LOGS_PATH is per-trial (Harbor's trial_paths.agent_dir); fall
# back to the container hostname (also per-trial) if Harbor stops setting it.
# Both inputs are hashed so neither task scenario nor trial name leaks.
TRIAL_RAW="${HOST_AGENT_LOGS_PATH:-$(cat /etc/hostname)}"
PROJECT_HASH="$(printf '%s' "${TASK_ID}-${TASK_NAME}-${TRIAL_RAW}" | sha256sum | cut -c1-12)"
PROJECT_NAME="otel-${PROJECT_HASH}"
export COMPOSE_PROJECT_NAME="$PROJECT_NAME"

# ── 0c. Assign unique host ports for services that need host-port bindings ───
# Reserve ports atomically and hold them open until Docker binds them.
# This avoids the TOCTOU race where a port is freed after detection but
# before Docker binds it, causing "port is already allocated" errors.
PORT_HOLDER_PIDFILE="/tmp/port-holder.pid"

allocate_ports() {
    # Kill any existing holder from a previous attempt
    release_ports
    read -r ENVOY_HOST_PORT ENVOY_ADMIN_HOST_PORT PROMETHEUS_HOST_PORT OPENSEARCH_HOST_PORT < <(
        python3 -c "
import socket, os, sys, signal, time

sockets = []
ports = []
for _ in range(4):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', 0))
    ports.append(s.getsockname()[1])
    sockets.append(s)

# Print ports for the parent shell to capture
print(' '.join(str(p) for p in ports))
sys.stdout.flush()

# Fork a holder process that keeps sockets open until signalled
pid = os.fork()
if pid == 0:
    # Child: hold sockets open until SIGTERM
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)
else:
    # Parent: write holder PID and exit
    with open('${PORT_HOLDER_PIDFILE}', 'w') as f:
        f.write(str(pid))
    for s in sockets:
        s.close()
"
    )
    export ENVOY_HOST_PORT ENVOY_ADMIN_HOST_PORT PROMETHEUS_HOST_PORT OPENSEARCH_HOST_PORT
}

release_ports() {
    if [ -f "${PORT_HOLDER_PIDFILE}" ]; then
        kill "$(cat "$PORT_HOLDER_PIDFILE")" 2>/dev/null || true
        rm -f "$PORT_HOLDER_PIDFILE"
    fi
}

if [ -z "${ENVOY_HOST_PORT:-}" ] || [ -z "${ENVOY_ADMIN_HOST_PORT:-}" ] \
   || [ -z "${PROMETHEUS_HOST_PORT:-}" ] || [ -z "${OPENSEARCH_HOST_PORT:-}" ]; then
    allocate_ports
fi

echo "[entrypoint] Project: $PROJECT_NAME"
echo "[entrypoint] Ports: envoy=$ENVOY_HOST_PORT, envoy-admin=$ENVOY_ADMIN_HOST_PORT, prometheus=$PROMETHEUS_HOST_PORT, opensearch=$OPENSEARCH_HOST_PORT"

# ── 1. Materialize per-trial tree from host snapshot cache ───────────────────
# The host Docker daemon can only bind-mount host-visible paths.
# CONTEXT_DIR is bind-mounted at the same path on host and container, so
# materializing here makes data accessible to nested compose services.
# docker-compose-base.yml uses relative paths (./data/...) so data must
# be a sibling of the compose file inside the opentelemetry-demo dir.
#
# Per-trial copy strategy: deep copy the small parts; bind the heavy bits
# directly from the cache via the inner compose:
#   - opentelemetry-demo source       (~12 MB):    cp -a from cache
#   - data/ (configs, opensearch-data) (small):    cp -a from cache
#   - data/prometheus/snapshots/*     (~10 GB):    excluded — only the one
#                                                  selected by SNAPSHOT_NAME
#                                                  is copied below
#   - data/opensearch-repo-snapshots/ (~41 GB):    excluded — opensearch
#                                                  bind-mounts it :ro directly
#                                                  from ${SNAPSHOT_CACHE_HOST_DIR}
#                                                  per docker-compose-base.yml
#   - data/prometheus/snapshots/${SNAPSHOT_NAME}/ (~few hundred MB): cp -a
#
# Hardlinks (cp -al) would let us skip the prometheus snapshot copy too, but
# Linux's link(2) refuses links across distinct mount points (`man 2 link`,
# EXDEV note) — and Docker bind-mounts the cache and CONTEXT_DIR as separate
# vfsmounts even when they're on the same host filesystem.
OTEL_DIR="${CONTEXT_DIR}/.trials/${PROJECT_HASH}/opentelemetry-demo"
mkdir -p "$(dirname "$OTEL_DIR")"
if [ ! -d "${OTEL_DIR}" ]; then
    echo "[entrypoint] Copying otel-demo from ${SNAPSHOT_CACHE_HOST_DIR}..."
    cp -a "${SNAPSHOT_CACHE_HOST_DIR}/opentelemetry-demo" "${OTEL_DIR}"
fi
if [ ! -d "${OTEL_DIR}/data" ]; then
    echo "[entrypoint] Copying snapshot data (excluding heavy snapshot trees)..."
    mkdir -p "${OTEL_DIR}/data"
    tar -C "${SNAPSHOT_CACHE_HOST_DIR}/data" \
        --exclude='./prometheus/snapshots' \
        --exclude='./opensearch-repo-snapshots' \
        -cf - . | tar -C "${OTEL_DIR}/data" -xf -
fi

# Populate the Prometheus data dir with only the selected snapshot's TSDB block.
# docker-compose-base.yml mounts $DATA_DIR/prometheus -> /prometheus, so the
# TSDB block must be a direct child of that directory.
PROM_SRC="${SNAPSHOT_CACHE_HOST_DIR}/data/prometheus/snapshots/${SNAPSHOT_NAME}"
PROM_DEST="${OTEL_DIR}/data/prometheus"
if [ ! -d "${PROM_DEST}" ] || [ -z "$(ls -A "${PROM_DEST}" 2>/dev/null)" ]; then
    if [ -d "${PROM_SRC}" ]; then
        echo "[entrypoint] Copying Prometheus snapshot ${SNAPSHOT_NAME}..."
        mkdir -p "${PROM_DEST}"
        cp -a "${PROM_SRC}/." "${PROM_DEST}/"
    else
        echo "[entrypoint] WARNING: Prometheus snapshot not found: ${PROM_SRC}"
    fi
fi
# Place docker-compose-base.yml alongside the otel-demo source
cp -f "${SNAPSHOT_CACHE_HOST_DIR}/docker-compose-base.yml"     "${OTEL_DIR}/docker-compose-base.yml"
cp -f "${SNAPSHOT_CACHE_HOST_DIR}/docker-compose.snapshot.yml" "${OTEL_DIR}/docker-compose.snapshot.yml"
cp -f "${SNAPSHOT_CACHE_HOST_DIR}/otelcol-config-snapshot.yml" "${OTEL_DIR}/otelcol-config-snapshot.yml"
# Jaeger's max_span_age must still reach back to the snapshot's trace data,
# which is frozen while wall-clock time advances, so size it at start-up
# rather than copying the baked-in value.
"${SNAPSHOT_CACHE_HOST_DIR}/set_max_span_age.sh" \
    "${SNAPSHOT_CACHE_HOST_DIR}/jaeger-config-snapshot.yml" \
    "${OTEL_DIR}/jaeger-config-snapshot.yml"
# Ensure the .env file is present — docker compose reads it automatically
# from the compose file's directory for variable substitution.
if [ -f "${SNAPSHOT_CACHE_HOST_DIR}/opentelemetry-demo/.env" ] && [ ! -f "${OTEL_DIR}/.env" ]; then
    echo "[entrypoint] Copying .env to ${OTEL_DIR}/.env..."
    cp -a "${SNAPSHOT_CACHE_HOST_DIR}/opentelemetry-demo/.env" "${OTEL_DIR}/.env"
fi

DATA_DIR="${OTEL_DIR}/data"
COMPOSE_FILE="${OTEL_DIR}/docker-compose-base.yml"
SNAPSHOT_COMPOSE="${OTEL_DIR}/docker-compose.snapshot.yml"

# ── 1b. Register cleanup trap ────────────────────────────────────────────────
# The inner otel-demo containers are siblings on the host Docker daemon (not
# children of this container), so they won't be cleaned up when Harbor tears
# down the outer compose.  Trap signals to tear them down explicitly.
# We cannot use `exec "$@"` at the end because that replaces this shell and
# loses the trap.  Instead we run the command in the background and wait.
cleanup() {
    local t0=$(date +%s)
    echo "[entrypoint] [$(date '+%H:%M:%S')] Cleaning up otel-demo resources for $PROJECT_NAME..."
    # Disconnect first so `docker compose down` can remove the network.  An
    # external endpoint on the network (this container) would otherwise leave
    # an orphaned network behind.
    docker network disconnect -f "${COMPOSE_PROJECT_NAME}_default" "$(cat /etc/hostname)" 2>/dev/null || true
    # SIGKILL inner services up-front so we don't wait per-service stop_grace_period.
    # Graceful shutdown (especially opensearch flushing indices) regularly pushed
    # cleanup past the outer container's 120s stop_grace_period, getting the trap
    # SIGKILL'd before `down` finished — leaving orphan containers, the per-trial
    # opensearch image, and ~1.5 GB of opensearch-data on disk per leaked trial.
    docker compose -f "$COMPOSE_FILE" -f "$SNAPSHOT_COMPOSE" \
        kill 2>&1 \
        | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' || true
    docker compose -f "$COMPOSE_FILE" -f "$SNAPSHOT_COMPOSE" \
        down --volumes --remove-orphans --rmi local 2>&1 \
        | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' || true
    # Reclaim the bind-mounted payload on the host.  CONTEXT_DIR is a host
    # bind mount, so `docker compose down --volumes` never touches it and the
    # ~6 GB of copied snapshot data per trial would leak indefinitely.
    rm -rf "$OTEL_DIR" 2>/dev/null || true
    echo "[entrypoint] [$(date '+%H:%M:%S')] Cleanup took $(( $(date +%s) - t0 ))s"
}
trap cleanup EXIT TERM INT

# ── 2. Ensure opensearch-data is empty ───────────────────────────────────────
# OpenSearch must start with an empty data dir so the snapshot restore can
# recreate indices from scratch without conflicts from pre-existing indices.
rm -rf "$DATA_DIR/opensearch-data"
mkdir -p "$DATA_DIR/opensearch-data"

# ── 3. Fix permissions on data/ subdirectories ──────────────────────────────
# Prometheus runs as nobody (uid 65534) and OpenSearch as uid 1000;
# both need write access to their data directories. opensearch-repo-snapshots
# is bound :ro from the shared cache (see docker-compose-base.yml) and is not
# in this per-trial dir, so no chmod is needed for it.
echo "[entrypoint] Fixing data directory permissions..."
chmod -R a+rwX "$DATA_DIR/prometheus" 2>/dev/null || true
chmod -R a+rwX "$DATA_DIR/opensearch-data" 2>/dev/null || true

# ── 4. Start the OpenTelemetry Demo stack ───────────────────────────────────
echo "[entrypoint] Starting OpenTelemetry Demo services..."
export OPENSEARCH_READ_ONLY=true
export OTEL_COLLECTOR_CONFIG="${OTEL_DIR}/otelcol-config-snapshot.yml"
export JAEGER_CONFIG="${OTEL_DIR}/jaeger-config-snapshot.yml"
# Pull only images not already cached locally (--policy missing) to avoid
# ghcr.io rate-limit timeouts when many Harbor tasks start concurrently.
docker compose --progress=plain --parallel 3 -f "$COMPOSE_FILE" pull --policy missing --include-deps 2>&1 | cat
MAX_RETRIES=3
for attempt in $(seq 1 $MAX_RETRIES); do
    release_ports
    if docker compose -f "$COMPOSE_FILE" -f "$SNAPSHOT_COMPOSE" up --pull never --force-recreate --remove-orphans --detach 2>&1; then
        break
    fi
    echo "[entrypoint] docker compose up failed (attempt $attempt/$MAX_RETRIES)"
    docker compose -f "$COMPOSE_FILE" -f "$SNAPSHOT_COMPOSE" down --remove-orphans --timeout 30 2>/dev/null || true
    if [ "$attempt" -eq "$MAX_RETRIES" ]; then
        echo "[entrypoint] ERROR: docker compose up failed after $MAX_RETRIES attempts"
        exit 1
    fi
    allocate_ports
    echo "[entrypoint] Retrying with ports: envoy=$ENVOY_HOST_PORT, envoy-admin=$ENVOY_ADMIN_HOST_PORT, prometheus=$PROMETHEUS_HOST_PORT, opensearch=$OPENSEARCH_HOST_PORT"
done

# ── 4b. Join the otel-demo compose network ─────────────────────────────────
# Attach this (agent) container to the otel-demo compose network so we can
# reach services by DNS name on a per-trial private network.  This replaces
# the previous host-networking design, which leaked every other trial's
# published ports into every agent container.
SELF_CID="$(cat /etc/hostname)"
NET_NAME="${COMPOSE_PROJECT_NAME}_default"
if ! docker network connect "$NET_NAME" "$SELF_CID" 2>/tmp/netconn.err; then
    # Re-entry is fine: if we're already attached Docker prints
    # "endpoint with name ... already exists".  Anything else is fatal.
    if ! grep -q "already exists" /tmp/netconn.err; then
        echo "[entrypoint] ERROR: could not join network '$NET_NAME'"
        cat /tmp/netconn.err
        docker network ls
        exit 1
    fi
fi
echo "[entrypoint] Joined network $NET_NAME as $SELF_CID"

# ── 5. Wait for services to become healthy ──────────────────────────────────
echo "[entrypoint] Waiting for services to initialise..."
GRAFANA_URL="http://frontend-proxy:8080/grafana"
MAX_WAIT=120
ELAPSED=0
while ! curl -s "${GRAFANA_URL}/api/health" > /dev/null 2>&1; do
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "[entrypoint] Warning: Grafana not reachable after ${MAX_WAIT}s, continuing anyway"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "[entrypoint] Grafana reachable after ~${ELAPSED}s"

# ── 6. Restore OpenSearch snapshot ──────────────────────────────────────────
# OpenSearch requires lowercase snapshot names
OS_SNAPSHOT_NAME="${SNAPSHOT_NAME,,}"
echo "[entrypoint] Restoring OpenSearch snapshot: $OS_SNAPSHOT_NAME"
# Register the snapshot repository as read-only — the repo dir is bind-mounted
# :ro from the shared cache, and `readonly: true` tells OpenSearch to skip any
# repo-side writes (e.g. master.dat refresh) that would otherwise hit EROFS.
# verify=false skips the post-register verification round-trip, which also
# attempts a write.
curl -s -X PUT "http://opensearch:9200/_snapshot/scheduled_backups?verify=false" \
    -H 'Content-Type: application/json' \
    -d '{"type": "fs", "settings": {"location": "/usr/share/opensearch/snapshots", "readonly": true}}' \
    || echo "[entrypoint] Warning: could not register OpenSearch snapshot repo"

sleep 2

# Restore the snapshot (no need to close indices — opensearch-data starts empty)
curl -s -X POST "http://opensearch:9200/_snapshot/scheduled_backups/${OS_SNAPSHOT_NAME}/_restore?wait_for_completion=true" \
    -H 'Content-Type: application/json' \
    -d '{"indices": "*", "include_global_state": false}' \
    || echo "[entrypoint] Warning: OpenSearch snapshot restore failed (may not exist for this snapshot)"

echo "[entrypoint] Environment ready."

# ── 6b. Export service URLs so agent scripts can discover them ──────────────
# Agent-facing URLs resolve by Docker DNS on the per-trial compose network.
# HOST_PORT vars are kept for host-side tools (check_health.sh, find_metrics.py)
# that run outside the container and still hit the published host ports.
cat > /tmp/env-ports <<EOF
export GRAFANA_URL=http://frontend-proxy:8080/grafana
export JAEGER_URL=http://frontend-proxy:8080/jaeger/ui
export OPENSEARCH_URL=http://opensearch:9200
export PROMETHEUS_URL=http://prometheus:9090
export ENVOY_HOST_PORT=${ENVOY_HOST_PORT}
export ENVOY_ADMIN_HOST_PORT=${ENVOY_ADMIN_HOST_PORT}
export PROMETHEUS_HOST_PORT=${PROMETHEUS_HOST_PORT}
export OPENSEARCH_HOST_PORT=${OPENSEARCH_HOST_PORT}
# Hide Harbor-injected paths that contain the task name (e.g. "lag60m-avail0m")
# from agent shells, which would otherwise leak the task setup.
unset CONTEXT_DIR TEST_DIR
EOF

touch /tmp/env-ready

# ── 7. Hand off to the agent command (if any) ──────────────────────────────
# Run in foreground (not exec) so the EXIT trap can fire for cleanup.
"$@" &
CHILD_PID=$!
wait "$CHILD_PID" 2>/dev/null || true
