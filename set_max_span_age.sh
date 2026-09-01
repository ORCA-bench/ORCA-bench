#!/usr/bin/env bash
# Write a Jaeger snapshot config with max_span_age sized to the snapshot data.
#
# Snapshot trace data is frozen at SNAPSHOT_DATA_DATE while wall-clock time
# keeps advancing, so a hardcoded max_span_age eventually stops reaching back
# far enough.  Jaeger builds its index-name list from `now - max_span_age`, so
# once the window closes it never names the snapshot's dated indices and the
# service list silently comes back empty (see README.md).  Compute the window
# at start-up instead of baking it in.
#
# Usage:
#   set_max_span_age.sh <src-config> <dst-config>
#
# Environment:
#   SNAPSHOT_DATA_DATE         Date of the snapshot's trace data (default 2026-04-18)
#   MAX_SPAN_AGE_MARGIN_DAYS   Slack added to the computed window (default 7)
#   OPENSEARCH_MAX_LINE_BYTES  Keep in sync with http.max_initial_line_length
#                              on the opensearch service (default 65536)

set -euo pipefail

SRC="${1:?usage: set_max_span_age.sh <src-config> <dst-config>}"
DST="${2:?usage: set_max_span_age.sh <src-config> <dst-config>}"

SNAPSHOT_DATA_DATE="${SNAPSHOT_DATA_DATE:-2026-04-18}"
MARGIN_DAYS="${MAX_SPAN_AGE_MARGIN_DAYS:-7}"
LIMIT_BYTES="${OPENSEARCH_MAX_LINE_BYTES:-65536}"

if [ -e "$DST" ] && [ "$SRC" -ef "$DST" ]; then
    echo "[max-span-age] ERROR: src and dst are the same file: $SRC" >&2
    exit 1
fi

now_s=$(date -u +%s)
if ! snap_s=$(date -u -d "$SNAPSHOT_DATA_DATE" +%s 2>/dev/null); then
    echo "[max-span-age] ERROR: unparseable SNAPSHOT_DATA_DATE: $SNAPSHOT_DATA_DATE" >&2
    exit 1
fi

age_days=$(( (now_s - snap_s) / 86400 ))
if [ "$age_days" -lt 0 ]; then
    age_days=0
fi
window_days=$(( age_days + MARGIN_DAYS ))
hours=$(( window_days * 24 ))

# Jaeger puts one index name per day in the query URL, and OpenSearch rejects a
# request line longer than http.max_initial_line_length.  Of the four index
# families the service name is longest, at 38 bytes/day including its comma.
est_bytes=$(( window_days * 38 ))
if [ "$est_bytes" -gt $(( LIMIT_BYTES * 3 / 4 )) ]; then
    echo "[max-span-age] WARNING: ${window_days}d needs ~${est_bytes}B of index names," \
         "against a ${LIMIT_BYTES}B OpenSearch line limit. Raise" \
         "http.max_initial_line_length on the opensearch service." >&2
fi

mkdir -p "$(dirname "$DST")"
sed -E "s|^([[:space:]]*max_span_age:[[:space:]]*).*|\1${hours}h|" "$SRC" > "$DST"

if ! grep -qE "^[[:space:]]*max_span_age:[[:space:]]*${hours}h[[:space:]]*$" "$DST"; then
    echo "[max-span-age] ERROR: no max_span_age key found in $SRC" >&2
    exit 1
fi

echo "[max-span-age] snapshot ${SNAPSHOT_DATA_DATE} is ${age_days}d old;" \
     "set max_span_age=${hours}h (${window_days}d) -> ${DST}"
