#!/usr/bin/env bash
# check_health.sh — Run from the host to verify the OpenTelemetry Demo
#                   environment is healthy and queryable.
#
# Usage:
#   ./check_health.sh
#
# Override defaults with environment variables:
#   GRAFANA_URL=http://admin:admin@localhost:8080/grafana ./check_health.sh
set -euo pipefail

ENVOY_HOST_PORT="${ENVOY_HOST_PORT:-8080}"
OPENSEARCH_HOST_PORT="${OPENSEARCH_HOST_PORT:-9200}"
PROMETHEUS_HOST_PORT="${PROMETHEUS_HOST_PORT:-9090}"
GRAFANA_URL="${GRAFANA_URL:-http://admin:admin@localhost:${ENVOY_HOST_PORT}/grafana}"
echo "GRAFANA_URL=${GRAFANA_URL}"
DATASOURCE_UID="${DATASOURCE_UID:-webstore-metrics}"
JAEGER_URL="${JAEGER_URL:-http://localhost:${ENVOY_HOST_PORT}/jaeger/ui}"
OPENSEARCH_URL="${OPENSEARCH_URL:-http://localhost:${OPENSEARCH_HOST_PORT}}"

TIMEOUT="${TIMEOUT:-10}"
CURL="curl -sf --max-time ${TIMEOUT}"
PASS=0
FAIL=0

check() {
    local label="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "  [PASS] $label"
        (( ++PASS ))
    else
        echo "  [FAIL] $label"
        (( ++FAIL ))
    fi
}

# ── 1. Grafana ───────────────────────────────────────────────────────────────
echo ""
echo "== Grafana =="
check "Grafana API reachable" $CURL "${GRAFANA_URL}/api/health"

echo -n "  Datasource ${DATASOURCE_UID}: "
DS_RESP=$($CURL "${GRAFANA_URL}/api/datasources/uid/${DATASOURCE_UID}" 2>/dev/null) && {
    DS_NAME=$(echo "$DS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('name','unknown'))" 2>/dev/null || echo "unknown")
    echo "[PASS] found (name=${DS_NAME})"
    (( ++PASS ))
} || {
    echo "[FAIL] not reachable"
    (( ++FAIL ))
}

# ── 2. Prometheus ─────────────────────────────────────────────────────────────
echo ""
echo "== Prometheus =="
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:${PROMETHEUS_HOST_PORT}}"
check "Prometheus API reachable" $CURL "${PROMETHEUS_URL}/api/v1/status/config"

# Use label names endpoint to verify snapshot data is loaded (read-only mode
# has no scrape targets so headStats.numSeries is always 0, but snapshot data
# is stored in blocks and still queryable).
METRIC_COUNT=$($CURL "${PROMETHEUS_URL}/api/v1/label/__name__/values" 2>/dev/null \
    | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))" 2>/dev/null || echo "?")
if [ "$METRIC_COUNT" != "0" ] && [ "$METRIC_COUNT" != "?" ]; then
    echo "  [PASS] Prometheus has ${METRIC_COUNT} metrics loaded"
    (( ++PASS ))
else
    echo "  [FAIL] Prometheus has no metrics (snapshot may not have loaded)"
    (( ++FAIL ))
fi

# Also verify Grafana can proxy to Prometheus
check "Grafana->Prometheus proxy" $CURL "${GRAFANA_URL}/api/datasources/proxy/uid/${DATASOURCE_UID}/api/v1/status/config"

# ── 3. Jaeger ────────────────────────────────────────────────────────────────
echo ""
echo "== Jaeger =="
check "Jaeger UI reachable" $CURL "${JAEGER_URL}/"

SERVICES_RESP=$($CURL "${JAEGER_URL}/api/services" 2>/dev/null) && {
    SVC_LIST=$(echo "$SERVICES_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('\n'.join(d.get('data',[])))" 2>/dev/null || echo "")
    SVC_COUNT=$(echo "$SVC_LIST" | grep -c . || echo "0")
    echo "  [PASS] Jaeger has ${SVC_COUNT} service(s)"
    (( ++PASS ))

    # Show an example trace from each service.
    # Determine the query window from the span index dates so that snapshot
    # data (which may be days/weeks old) is discoverable.
    SPAN_DATES=$(curl -sf --max-time 10 "${OPENSEARCH_URL}/_cat/indices/jaeger-main-jaeger-span-*?h=index" 2>/dev/null \
        | sed 's/.*jaeger-main-jaeger-span-//' | sort)
    if [ -n "$SPAN_DATES" ]; then
        EARLIEST_DATE=$(echo "$SPAN_DATES" | head -1)
        # start at midnight of the earliest span-index date
        START_TS=$(python3 -c "import datetime; print(int(datetime.datetime.strptime('${EARLIEST_DATE}','%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp() * 1e6))")
    else
        START_TS=$(( $(date +%s%6N) - 24 * 3600 * 1000000 ))
    fi
    END_TS=$(date +%s%6N)  # microseconds since epoch (Jaeger API format)
    SVC_WITH_TRACES=0
    while IFS= read -r svc; do
        [ -z "$svc" ] && continue
        TRACE_RESP=$(curl -sf --max-time 30 \
            "${JAEGER_URL}/api/traces?service=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${svc}'))")&limit=1&start=${START_TS}&end=${END_TS}" 2>/dev/null || echo "")
        EXAMPLE=$(echo "$TRACE_RESP" | python3 -c "
import sys, json, datetime
try:
    d = json.load(sys.stdin).get('data', [])
except Exception:
    d = []
if not d:
    print('no traces')
else:
    span = d[0].get('spans', [{}])[0]
    op = span.get('operationName', '?')
    ts_us = span.get('startTime', 0)
    ts = datetime.datetime.fromtimestamp(ts_us / 1e6).strftime('%H:%M:%S') if ts_us else '?'
    dur = span.get('duration', 0)
    if dur >= 1000:
        dur_str = f'{dur/1000:.1f}ms'
    else:
        dur_str = f'{dur}us'
    print(f'{op} at {ts} ({dur_str})')
" 2>/dev/null || echo "error querying")
        if [ "$EXAMPLE" != "no traces" ] && [ "$EXAMPLE" != "error querying" ]; then
            SVC_WITH_TRACES=$((SVC_WITH_TRACES + 1))
        fi
        echo "    ${svc}: ${EXAMPLE}"
    done <<< "$SVC_LIST"

    if [ "$SVC_WITH_TRACES" -gt 0 ]; then
        echo "  [PASS] ${SVC_WITH_TRACES}/${SVC_COUNT} services have traces"
        (( ++PASS ))
    else
        echo "  [FAIL] No traces found in Jaeger"
        (( ++FAIL ))
    fi
} || {
    echo "  [FAIL] Cannot list Jaeger services"
    (( ++FAIL ))
}

# ── 4. OpenSearch ────────────────────────────────────────────────────────────
echo ""
echo "== OpenSearch =="
check "OpenSearch cluster reachable" $CURL "${OPENSEARCH_URL}/_cluster/health"

OS_HEALTH=$($CURL "${OPENSEARCH_URL}/_cluster/health" 2>/dev/null) && {
    CLUSTER_STATUS=$(echo "$OS_HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
    echo "  Cluster status: ${CLUSTER_STATUS}"
    INDICES=$($CURL "${OPENSEARCH_URL}/_cat/indices?v&h=index,docs.count,store.size" 2>/dev/null || true)
    if [ -n "$INDICES" ]; then
        echo "  Indices:"
        echo "$INDICES" | sed 's/^/    /'
    fi
} || true

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "==============================="
echo "  PASS: ${PASS}   FAIL: ${FAIL}"
echo "==============================="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
