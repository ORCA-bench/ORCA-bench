#!/usr/bin/env python3
"""Build a restoration manifest by detecting in-place data wipes from snapshot size curves.

Two ransomware attack patterns can cause data loss:
  (1) DELETE INDEX (full drop, recreates with new UUID)
  (2) DELETE BY QUERY (in-place wipe of docs; UUID unchanged)

The original version of this script mined the OpenSearch container logs for
MetadataDeleteIndexService events, which only catches pattern (1). This version
also catches pattern (2) by walking each daily index's snapshot size curve and
flagging snapshots that immediately precede a sharp size drop. Each such peak
snapshot becomes a recoverable "lifetime" of that index.

Usage: python build_restore_manifest.py <data_dir> [opensearch_url]
"""

import json
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(sys.argv[1])
OS_URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9200"
REPO = "scheduled_backups"

INDEX_TYPES = ["otel-logs", "jaeger-main-jaeger-span", "jaeger-main-jaeger-service"]
DATES = [
    "2026-04-19",
    "2026-04-20",
    "2026-04-21",
    "2026-04-22",
    "2026-04-23",
    "2026-04-24",
    "2026-04-25",
]
TARGETS = [f"{t}-{d}" for t in INDEX_TYPES for d in DATES]

# A wipe is detected when size drops below DROP_RATIO * recent_peak AND the peak
# was at least MIN_PEAK_BYTES (filters noise from indices that never grew).
# Threshold is per index type because traffic volumes differ by 3+ orders of
# magnitude (otel-logs/jaeger-span are MB-scale; jaeger-service is KB-scale,
# tracking the small set of services seen on a given day).
DROP_RATIO = 0.20
# Threshold = "minimum peak size to count as a recoverable lifetime end."
# Set just above one snapshot interval's worth of writes (5 minutes) so the
# detector catches partial-recovery-then-wiped-again sequences, not just the
# big peaks. Setting it too high silently misses small wipes that fell into
# the slow-traffic windows of a day (e.g. otel-logs-2026-04-24 at 17:35 UTC,
# peak 15 MB, was missed at the original 20 MB threshold).
MIN_PEAK_BYTES_BY_TYPE = {
    "otel-logs": 5 * 1024 * 1024,  # 5 MB
    "jaeger-main-jaeger-span": 5 * 1024 * 1024,  # 5 MB
    "jaeger-main-jaeger-service": 5 * 1024,  # 5 KB
}

# 1. Pull all snapshot metadata
with urllib.request.urlopen(f"{OS_URL}/_snapshot/{REPO}/_all") as r:
    all_snaps = json.load(r)["snapshots"]
for s in all_snaps:
    s["_dt"] = datetime.strptime(s["start_time"][:19], "%Y-%m-%dT%H:%M:%S")
all_snaps.sort(key=lambda s: s["_dt"])


def fetch_size(snap_name: str, idx: str) -> int:
    """Return the size of `idx` in bytes within the named snapshot, retrying on 429/503."""
    url = f"{OS_URL}/_snapshot/{REPO}/{snap_name}/_status"
    import time

    for attempt in range(6):
        try:
            with urllib.request.urlopen(url) as r:
                sd = json.load(r)["snapshots"][0]
            return (
                sd["indices"]
                .get(idx, {})
                .get("stats", {})
                .get("total", {})
                .get("size_in_bytes", 0)
            )
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 5:
                time.sleep(0.5 * (2**attempt))
                continue
            raise


# 2. For each target index, fetch sizes for every snapshot and detect peaks
manifest = []
for idx in TARGETS:
    snaps_for_idx = [s for s in all_snaps if idx in s["indices"]]
    if not snaps_for_idx:
        print(f"[skip] {idx}: not present in any snapshot")
        continue

    print(f"[scan] {idx}: {len(snaps_for_idx)} snapshots", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        sizes = list(ex.map(lambda s: fetch_size(s["snapshot"], idx), snaps_for_idx))

    min_peak = next(
        (v for prefix, v in MIN_PEAK_BYTES_BY_TYPE.items() if idx.startswith(prefix)),
        20 * 1024 * 1024,
    )

    # Walk the curve: track running peak, flag whenever size collapses
    peaks = []  # list of (peak_index_in_snaps_for_idx, peak_size)
    peak_i, peak_size = 0, sizes[0]
    for i in range(1, len(sizes)):
        if sizes[i] >= peak_size:
            peak_i, peak_size = i, sizes[i]
        elif peak_size >= min_peak and sizes[i] < peak_size * DROP_RATIO:
            peaks.append((peak_i, peak_size))
            # Reset tracking from this point
            peak_i, peak_size = i, sizes[i]

    # Final snapshot is always emitted as a lifetime (post-final-wipe state)
    peaks.append((len(sizes) - 1, sizes[-1]))

    if max(sizes) < min_peak:
        print(
            f"  no recoverable lifetimes (max size {max(sizes)} bytes below threshold {min_peak})"
        )
        continue

    for life_num, (pi, psize) in enumerate(peaks, start=1):
        snap = snaps_for_idx[pi]
        manifest.append(
            {
                "snapshot": snap["snapshot"],
                "index": idx,
                "lifetime": life_num,
                "snapshot_time": snap["start_time"],
                "size_mb": round(psize / 1024 / 1024, 1),
            }
        )
        print(
            f"  lifetime{life_num}: {snap['start_time']}  {psize / 1024 / 1024:7.1f} MB  {snap['snapshot']}"
        )

out = DATA_DIR / "restore_manifest.json"
out.write_text(json.dumps(manifest, indent=2))
print(f"\nWrote {len(manifest)} entries to {out}")

by_index = Counter(e["index"] for e in manifest)
print("\nLifetimes per index:")
for idx, n in sorted(by_index.items()):
    print(f"  {n:3d}  {idx}")
