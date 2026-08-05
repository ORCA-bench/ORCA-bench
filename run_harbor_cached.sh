#!/usr/bin/env bash
# Run `harbor run` with a one-time-staged snapshot cache.
#
# Stages /app/ from the snapshot image to a host-side cache (idempotent),
# then invokes `harbor run` with --mounts-json bind-mounting the cache into
# every trial's agent container at the same host path. The patched
# harbor-template entrypoint hardlinks from there into each per-trial dir.
#
# Usage:
#   run_harbor_cached.sh <harbor-run-args...>
#
# Examples:
#   ./run_harbor_cached.sh -c configs/harbor_job_share2w_base.yaml
#   SNAPSHOT_IMAGE=orcabench/sre-otel-snapshot:data-0418-harbor-template \
#       ./run_harbor_cached.sh -c configs/foo.yaml --n-trials 1

set -euo pipefail

IMAGE="${SNAPSHOT_IMAGE:-orcabench/sre-otel-snapshot:data-0418-harbor-template}"

# Run this script from examples/otel-demo so ./stage_snapshot_cache.sh resolves.

CACHE_ROOT="/root/.cache/sre-snapshot-cache"
CACHE_DIR="$(./stage_snapshot_cache.sh --image "$IMAGE" --cache-root "$CACHE_ROOT")"
echo "[run_harbor_cached] Snapshot cache: $CACHE_DIR" >&2

# `mounts_json.source` is taken literally by Harbor (no ${VAR} interpolation),
# so we expand the host path here. target=source so paths are identical inside
# and outside the container — the entrypoint can pass the same value to cp -al
# without any in-container/host translation.
MOUNTS=$(printf '[{"type":"bind","source":"%s","target":"%s","read_only":true}]' \
         "$CACHE_DIR" "$CACHE_DIR")

# SNAPSHOT_CACHE_HOST_DIR rides through Harbor's resolve_env_vars() into the
# docker-compose subprocess env, then into the main container via the
# harbor-template docker-compose.yaml's `environment:` passthrough.
SNAPSHOT_CACHE_HOST_DIR="$CACHE_DIR" \
    exec uv run harbor run --mounts-json "$MOUNTS" "$@"
