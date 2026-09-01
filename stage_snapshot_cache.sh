#!/usr/bin/env bash
# Extract /app/ from the snapshot Docker image to a host-side cache, once.
#
# The cache is keyed by the image's docker image-id, so any rebuild of the
# snapshot image produces a new cache directory and old caches stay valid for
# any in-flight trial that pulled the older digest.
#
# Each Harbor trial then materializes its working tree by hardlinking from
# this cache (cp -al), which requires source and destination to share a host
# filesystem. The wrapper (run_harbor_cached.sh) enforces that invariant.
#
# Usage:
#   stage_snapshot_cache.sh --image <ref> --cache-root <path>
#
# Prints the populated cache directory to stdout for callers to capture.

set -euo pipefail

IMAGE=""
CACHE_ROOT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --cache-root) CACHE_ROOT="$2"; shift 2 ;;
        *) echo "stage_snapshot_cache.sh: unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$IMAGE" ] || [ -z "$CACHE_ROOT" ]; then
    echo "Usage: stage_snapshot_cache.sh --image <ref> --cache-root <path>" >&2
    exit 2
fi

mkdir -p "$CACHE_ROOT"

# Pull (no-op if already present locally and tag hasn't changed remotely).
docker pull "$IMAGE" >&2

IMG_ID="$(docker image inspect "$IMAGE" -f '{{.Id}}' | cut -d: -f2 | cut -c1-12)"
CACHE_DIR="$CACHE_ROOT/$IMG_ID"

# Serialize concurrent first-time invocations against the same cache root.
exec 9>"$CACHE_ROOT/.stage.lock"
flock 9

if [ -d "$CACHE_DIR" ] && [ -n "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    echo "$CACHE_DIR"
    exit 0
fi

echo "[stage_snapshot_cache] Extracting $IMAGE -> $CACHE_DIR" >&2
mkdir -p "$CACHE_DIR"
CID="$(docker create "$IMAGE")"
trap 'docker rm "$CID" >/dev/null 2>&1 || true' EXIT
docker cp "$CID:/app/." "$CACHE_DIR/"
echo "[stage_snapshot_cache] Done" >&2

echo "$CACHE_DIR"
