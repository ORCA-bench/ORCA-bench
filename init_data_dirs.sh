#!/usr/bin/env bash
# Creates the data subdirectories required as volume mounts in docker-compose.yml.
# Usage: ./init_data_dirs.sh [DATA_DIR]
#   DATA_DIR defaults to "data" (relative to the current directory).
#
# After running, export the printed environment variables
# (PROMETHEUS_MOUNT_DIR, OPENSEARCH_DATA_MOUNT_DIR, OPENSEARCH_SNAPSHOTS_MOUNT_DIR)
# before running docker compose.

set -euo pipefail

DATA_DIR="${1:-data}"

mkdir -p \
  "$DATA_DIR/jaeger" \
  "$DATA_DIR/jaeger-badger-snapshots" \
  "$DATA_DIR/prometheus" \
  "$DATA_DIR/opensearch-data" \
  "$DATA_DIR/opensearch-repo-snapshots"

# Match the ownership expected by docker-compose init containers:
#   jaeger-init:     chown 10001:10001
#   prometheus-init: chown 65534:65534
#   opensearch-init: chown 1000:1000
chown -R 10001:10001 "$DATA_DIR/jaeger"
chmod -R u+rwX        "$DATA_DIR/jaeger"
chown -R 10001:10001 "$DATA_DIR/jaeger-badger-snapshots"
chmod -R u+rwX        "$DATA_DIR/jaeger-badger-snapshots"
chown -R 65534:65534 "$DATA_DIR/prometheus"
chmod -R u+rwX        "$DATA_DIR/prometheus"
chown -R 1000:1000   "$DATA_DIR/opensearch-data"
chmod -R u+rwX        "$DATA_DIR/opensearch-data"
chown -R 1000:1000   "$DATA_DIR/opensearch-repo-snapshots"
chmod -R u+rwX        "$DATA_DIR/opensearch-repo-snapshots"

echo "Created data directories under $DATA_DIR"
echo ""
echo "Set the following environment variables before running docker compose:"
echo ""
echo "  export PROMETHEUS_MOUNT_DIR=\"$(cd "$DATA_DIR" && pwd)/prometheus\""
echo "  export OPENSEARCH_DATA_MOUNT_DIR=\"$(cd "$DATA_DIR" && pwd)/opensearch-data\""
echo "  export OPENSEARCH_SNAPSHOTS_MOUNT_DIR=\"$(cd "$DATA_DIR" && pwd)/opensearch-repo-snapshots\""
echo ""
