#!/usr/bin/env python3
"""Per-snapshot consolidation: build point-in-time-correct consolidated snapshots.

For each snapshot S in the *used* set (the snapshots Harbor tasks restore), this
emits a sibling snapshot ${S.lower()}-consolidated whose contents include all
recovered lifetime data with timestamp <= T (where T is parsed from S's name)
and *no* data from after T.

Workflow uses the cumulative trick: process used snapshots in time order against
a single long-lived cluster whose state grows monotonically. For each S at T:

  1. Apply any manifest peak with T_peak < T not yet applied (restore peak as
     temp, reindex into live daily index, delete temp). Each peak is processed
     exactly once across the whole run.
  2. Restore S as overlay (rename to *_temp, reindex into live daily index,
     delete *_temp). This adds the post-last-peak fresh data that exists only
     in the live snapshot, not in any peak.
  3. Snapshot ${S.lower()}-consolidated covering otel-logs-* and jaeger-main-*.

Because the cluster only ever holds data <= T at iteration T, the consolidated
snapshot is automatically point-in-time correct without per-doc timestamp filters.

Side effects:
  - Wipes all matching indices from the cluster on startup. To restore the
    pre-run state, restore post_consolidation_YYYY-MM-DD afterward.
  - Adds ~91 consolidated snapshots to the repo (~3 GB additional storage,
    courtesy of BlobStore segment dedup across the small per-iteration deltas).

Usage:
    uv run python per_snapshot_consolidate.py <data_dir> <used_snapshots.json>
    uv run python per_snapshot_consolidate.py <data_dir> <used_snapshots.json> --start <idx> --stop <idx>

The optional --start/--stop bounds let you smoke-test on a slice (e.g. 0..3) before
committing to the full 91-iteration run.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

OS_URL = "http://localhost:9200"
REPO = "scheduled_backups"
TARGET_INDEX_PATTERN = "otel-logs-*,jaeger-main-*"


def http(method: str, path: str, body: dict | None = None) -> dict:
    """Issue a JSON HTTP request to OpenSearch and return the parsed response body."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{OS_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req) as r:
                text = r.read().decode()
            return json.loads(text) if text else {}
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 5:
                time.sleep(0.5 * (2**attempt))
                continue
            raise


def parse_snap_time(snap_name: str) -> datetime:
    """Parse the wall-clock timestamp from a snapshot name like `20260421t125152z-...`."""
    return datetime.strptime(snap_name[:16].lower(), "%Y%m%dt%H%M%Sz")


def os_name(prom_name: str) -> str:
    """Convert a prometheus snapshot name (uppercase T/Z) to OpenSearch convention (lowercase)."""
    return prom_name.lower()


def list_indices_matching(prefix_or_pattern: str) -> list[str]:
    """Return the names of cluster indices matching the given _cat/indices pattern."""
    try:
        return [
            i["index"] for i in http("GET", f"/_cat/{prefix_or_pattern}?format=json")
        ]
    except urllib.error.HTTPError:
        return []


def index_exists(name: str) -> bool:
    """Return True if an index with the given name exists in the cluster."""
    try:
        http("GET", f"/{name}/_stats")
        return True
    except urllib.error.HTTPError:
        return False


def _delta_query_for(idx: str, t_prev: datetime | None) -> dict | None:
    """Build a `source.query` filter that selects only docs newer than `t_prev`.

    Returns None if no filter applies (first iteration, or jaeger-main-jaeger-service
    indices which carry no useful timestamp field).
    """
    if t_prev is None:
        return None
    if idx.startswith("otel-logs-"):
        return {"range": {"@timestamp": {"gt": t_prev.strftime("%Y-%m-%dT%H:%M:%SZ")}}}
    if idx.startswith("jaeger-main-jaeger-span-"):
        return {"range": {"startTimeMillis": {"gt": int(t_prev.timestamp() * 1000)}}}
    return None  # services: no filter (small + no clean timestamp)


def reindex_into(src: str, dst: str, query: dict | None = None) -> dict:
    """Reindex `src` into `dst` async-via-tasks API, polling until completion.

    `query` is an optional source-side filter so we only reindex docs matching
    it. Used to skip already-merged docs on overlay reindexes.
    """
    if not index_exists(dst):
        # Bootstrap dst with the same mapping as src so the first reindex doesn't
        # land in a dynamic-mapping default that drops fields.
        meta = http("GET", f"/{src}")[src]
        body = {
            "settings": {
                k: v
                for k, v in meta["settings"]["index"].items()
                if k in ("number_of_shards", "number_of_replicas")
            },
            "mappings": meta["mappings"],
        }
        http("PUT", f"/{dst}", body)

    source_body: dict = {"index": src, "size": 1000}
    if query is not None:
        source_body["query"] = query
    submit = http(
        "POST",
        "/_reindex?wait_for_completion=false",
        {
            "source": source_body,
            "dest": {"index": dst, "op_type": "index"},
        },
    )
    task_id = submit["task"]
    while True:
        time.sleep(2)
        st = http("GET", f"/_tasks/{task_id}")
        if st.get("completed"):
            break
    http("POST", f"/{dst}/_refresh")
    return st.get("response", {})


def restore_with_rename(snap: str, indices: str, rename_to: str) -> None:
    """Restore the named snapshot's `indices` into the cluster as `rename_to`."""
    http(
        "POST",
        f"/_snapshot/{REPO}/{snap}/_restore?wait_for_completion=true",
        {
            "indices": indices,
            "rename_pattern": ".+",
            "rename_replacement": rename_to,
            "include_global_state": False,
            "include_aliases": False,
        },
    )


def apply_peak(entry: dict) -> dict:
    """Restore a manifest peak as temp, reindex into the live daily, delete temp."""
    snap = entry["snapshot"]
    daily = entry["index"]
    tmp = f"{daily}-tmp"
    if index_exists(tmp):
        http("DELETE", f"/{tmp}")
    restore_with_rename(snap, daily, tmp)
    res = reindex_into(tmp, daily)
    http("DELETE", f"/{tmp}")
    return res


def apply_overlay(snap: str, t_prev: datetime | None = None) -> dict:
    """Restore a used snapshot S into temp namespace, reindex each into its daily counterpart.

    When `t_prev` is provided, the per-overlay reindex filters by timestamp > t_prev
    so we only ship the delta since the previous iteration — for late-day or
    next-day snapshots this is 5 min of data instead of 1 GB of cluster state.
    """
    # Clean up any leftover overlay indices from a prior interrupted run
    leftovers = [n for n in list_indices_matching("indices") if n.endswith("_overlay")]
    for ov in leftovers:
        http("DELETE", f"/{ov}")
    # Restore everything in S to a *_overlay suffix so we can iterate
    http(
        "POST",
        f"/_snapshot/{REPO}/{snap}/_restore?wait_for_completion=true",
        {
            "indices": TARGET_INDEX_PATTERN,
            "rename_pattern": "(.+)",
            "rename_replacement": "$1_overlay",
            "include_global_state": False,
            "include_aliases": False,
        },
    )
    overlays = [n for n in list_indices_matching("indices") if n.endswith("_overlay")]
    summary = {"reindexed": 0, "indices": []}
    for ov in overlays:
        daily = ov[: -len("_overlay")]
        query = _delta_query_for(daily, t_prev)
        res = reindex_into(ov, daily, query=query)
        summary["reindexed"] += res.get("created", 0) + res.get("updated", 0)
        summary["indices"].append(daily)
        http("DELETE", f"/{ov}")
    return summary


def wipe_cluster() -> None:
    """Delete all otel-logs-* and jaeger-main-* indices from the cluster (caller-confirmed)."""
    indices = [
        i["index"]
        for i in http("GET", "/_cat/indices?format=json")
        if i["index"].startswith(("otel-logs-", "jaeger-main-"))
    ]
    if not indices:
        return
    print(f"  Wiping {len(indices)} existing matching indices...", flush=True)
    # Batch in groups of 50 to avoid URL-too-long
    for i in range(0, len(indices), 50):
        batch = indices[i : i + 50]
        http("DELETE", "/" + ",".join(batch))


def take_consolidated_snapshot(snap_name: str) -> dict:
    """Snapshot all otel-logs-* and jaeger-main-* indices under the given name."""
    return http(
        "PUT",
        f"/_snapshot/{REPO}/{snap_name}?wait_for_completion=true",
        {
            "indices": TARGET_INDEX_PATTERN,
            "include_global_state": False,
            "metadata": {
                "purpose": f"point-in-time consolidated state at T={snap_name[:15]}"
            },
        },
    )


def _fill_missing(used: list[str], manifest: list[dict], existing: set[str]) -> None:
    """Build only the entries whose -consolidated snapshot is missing from the repo.

    Each missing entry is built independently:
      1. Wipe the cluster.
      2. Restore the closest existing -consolidated snapshot with T_prev < T as a base.
      3. Apply any manifest peaks with T_peak in (T_prev, T) (typically zero — peaks
         are sparse, and any peak that ever crossed a previously-built timeline has
         already been baked into an existing -consolidated).
      4. Apply S itself as an overlay, with delta filter `timestamp > T_prev`.
      5. Snapshot ${S.lower()}-consolidated.
    """
    # Sort existing -consolidated snapshots by their wall-clock T so we can
    # binary-search the closest predecessor for each missing entry.
    consolidated_by_t: list[tuple[datetime, str]] = []
    for snap in existing:
        if snap.endswith("-consolidated"):
            try:
                consolidated_by_t.append((parse_snap_time(snap), snap))
            except ValueError:
                continue
    consolidated_by_t.sort()

    missing = [S for S in used if f"{os_name(S)}-consolidated" not in existing]
    print(
        f"[fill] {len(missing)} missing entries to build (out of {len(used)} total).",
        flush=True,
    )

    for i, S in enumerate(missing):
        T = parse_snap_time(S)
        target_snap = f"{os_name(S)}-consolidated"

        # Find the latest existing -consolidated with T_prev < T
        base = None
        t_prev: datetime | None = None
        for tp, snap in consolidated_by_t:
            if tp < T:
                base, t_prev = snap, tp
            else:
                break

        t0 = time.time()
        print(f"[fill {i + 1}/{len(missing)}] {S}  T={T}  base={base}", flush=True)

        # 1. Wipe and restore base (or stay empty if no predecessor)
        wipe_cluster()
        if base is not None:
            http(
                "POST",
                f"/_snapshot/{REPO}/{base}/_restore?wait_for_completion=true",
                {
                    "indices": TARGET_INDEX_PATTERN,
                    "include_global_state": False,
                    "include_aliases": False,
                },
            )

        # 2. Apply peaks crossed since base's T (typically zero in practice)
        peaks_applied = 0
        for entry in manifest:
            tp = parse_snap_time(entry["snapshot"])
            if (t_prev is None or tp >= t_prev) and tp < T:
                res = apply_peak(entry)
                peaks_applied += 1
                print(
                    f"    peak {peaks_applied}: {entry['index']}-lifetime{entry['lifetime']} "
                    f"({entry['snapshot_time']}) +{res.get('created', 0):,} created",
                    flush=True,
                )

        # 3. Overlay S with delta filter (so we add only docs newer than T_prev)
        overlay = apply_overlay(os_name(S), t_prev=t_prev)
        print(
            f"    overlay: {len(overlay['indices'])} indices, {overlay['reindexed']:,} docs touched",
            flush=True,
        )

        # 4. Snapshot
        res = take_consolidated_snapshot(target_snap)
        snap_state = res.get("snapshot", {}).get("state", "?")
        elapsed = time.time() - t0
        print(f"    snapshot {target_snap}: {snap_state} in {elapsed:.1f}s", flush=True)

        # Future missing entries can use this newly-built one as a base
        consolidated_by_t.append((T, target_snap))
        consolidated_by_t.sort()


def main() -> None:
    """Drive the per-snapshot consolidation loop in time order."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("used_snapshots", type=Path)
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start at this index in the used list (default: 0)",
    )
    parser.add_argument(
        "--stop", type=int, default=None, help="Stop before this index (default: end)"
    )
    parser.add_argument(
        "--no-wipe",
        action="store_true",
        help="Skip the cluster wipe (use only when resuming)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip iterations whose -consolidated snapshot already exists",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help=(
            "Build only entries whose -consolidated snapshot is missing. "
            "Each missing entry is built independently: wipe cluster, restore "
            "the closest existing -consolidated with T_prev < T as a base, "
            "apply any peaks crossed since T_prev, overlay S, snapshot. "
            "Use this to extend an existing consolidation set with new used "
            "snapshots without rebuilding the whole timeline."
        ),
    )
    args = parser.parse_args()

    manifest = json.loads((args.data_dir / "restore_manifest.json").read_text())
    used = json.loads(args.used_snapshots.read_text())
    used = used[args.start : args.stop]
    used.sort(key=parse_snap_time)
    manifest.sort(key=lambda e: parse_snap_time(e["snapshot"]))

    print(
        f"Will process {len(used)} used snapshots and merge in {len(manifest)} manifest peaks.",
        flush=True,
    )

    # Check existing snapshots so we can skip when resuming
    if args.skip_existing or args.fill_missing:
        existing = {
            s["snapshot"] for s in http("GET", f"/_snapshot/{REPO}/_all")["snapshots"]
        }
    else:
        existing = set()

    if args.fill_missing:
        _fill_missing(used, manifest, existing)
        return

    if not args.no_wipe:
        print(
            "[wipe] Resetting cluster to empty otel-logs-*/jaeger-main-* state...",
            flush=True,
        )
        wipe_cluster()

    peak_idx = 0
    t_prev: datetime | None = None
    for i, S in enumerate(used):
        T = parse_snap_time(S)
        target_snap = f"{os_name(S)}-consolidated"
        if target_snap in existing:
            print(f"[{i + 1}/{len(used)}] skip {S} (snapshot exists)", flush=True)
            # Advance peak_idx and t_prev so the next iteration's delta filter
            # is anchored to the correct prior wall-clock time
            while (
                peak_idx < len(manifest)
                and parse_snap_time(manifest[peak_idx]["snapshot"]) < T
            ):
                peak_idx += 1
            t_prev = T
            continue

        t0 = time.time()
        print(f"[{i + 1}/{len(used)}] {S}  T={T}", flush=True)

        # 1. Apply any peaks crossed since last S
        peaks_applied = 0
        while (
            peak_idx < len(manifest)
            and parse_snap_time(manifest[peak_idx]["snapshot"]) < T
        ):
            entry = manifest[peak_idx]
            res = apply_peak(entry)
            peaks_applied += 1
            print(
                f"    peak {peaks_applied}: {entry['index']}-lifetime{entry['lifetime']} "
                f"({entry['snapshot_time']}) +{res.get('created', 0):,} created",
                flush=True,
            )
            peak_idx += 1

        # 2. Apply S as overlay (delta-only when we have a prior anchor)
        overlay = apply_overlay(os_name(S), t_prev=t_prev)
        print(
            f"    overlay: {len(overlay['indices'])} indices, {overlay['reindexed']:,} docs touched",
            flush=True,
        )

        # 3. Snapshot
        res = take_consolidated_snapshot(target_snap)
        snap_state = res.get("snapshot", {}).get("state", "?")
        elapsed = time.time() - t0
        print(f"    snapshot {target_snap}: {snap_state} in {elapsed:.1f}s", flush=True)
        t_prev = T


if __name__ == "__main__":
    main()
