#!/usr/bin/env python3
"""Reindex all *-lifetime* indices into their live daily counterpart.

After RESTORE_HISTORICAL_INDICES.md Phase 3 has restored peak snapshots as
*-lifetimeN indices, this script copies those docs into the live daily index
(e.g. jaeger-main-jaeger-span-2026-04-21-lifetime3 -> jaeger-main-jaeger-span-2026-04-21).

Once consolidated, the Jaeger HTTP API and any other consumer that queries by
exact daily index name will see the full pre-wipe + post-wipe union without
needing aliases or wildcard config. Span/log doc IDs are preserved, so re-runs
are idempotent (overlapping docs from adjacent lifetimes overwrite each other).

Usage: python consolidate_lifetimes.py [opensearch_url]
"""

import json
import sys
import time
import urllib.request
from collections import defaultdict

OS_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9200"


def http_json(method: str, path: str, body: dict | None = None) -> dict:
    """Issue a JSON HTTP request to OpenSearch and return the parsed response body."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{OS_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req) as r:
        text = r.read().decode()
    return json.loads(text) if text else {}


# Discover lifetime indices and group by their original daily name
all_indices = [i["index"] for i in http_json("GET", "/_cat/indices?format=json")]
groups: dict[str, list[str]] = defaultdict(list)
for idx in all_indices:
    if "-lifetime" in idx:
        original = idx.rsplit("-lifetime", 1)[0]
        groups[original].append(idx)

if not groups:
    print("No -lifetime indices found; nothing to consolidate.")
    sys.exit(0)


def clone_index(src: str, dst: str) -> None:
    """Create dst as an empty index with the same mapping/settings as src."""
    src_meta = http_json("GET", f"/{src}")[src]
    body = {
        "settings": {
            k: v
            for k, v in src_meta["settings"]["index"].items()
            if k in ("number_of_shards", "number_of_replicas")
        },
        "mappings": src_meta["mappings"],
    }
    http_json("PUT", f"/{dst}", body)


for original in sorted(groups):
    lifetimes = sorted(groups[original])
    if original not in all_indices:
        print(f"[create] {original}: live missing — bootstrap from {lifetimes[0]}")
        clone_index(lifetimes[0], original)

    before = http_json("GET", f"/{original}/_count")["count"]
    print(
        f"[reindex] {len(lifetimes)} lifetimes -> {original} (currently {before:,} docs)"
    )
    t0 = time.time()
    # Submit async; long reindexes break urllib's 100-header limit when waiting inline
    submit = http_json(
        "POST",
        "/_reindex?wait_for_completion=false",
        {
            "source": {"index": lifetimes, "size": 1000},
            "dest": {"index": original, "op_type": "index"},
        },
    )
    task_id = submit["task"]
    while True:
        time.sleep(2)
        st = http_json("GET", f"/_tasks/{task_id}")
        if st.get("completed"):
            break
        s = st["task"]["status"]
        print(
            f"    progress: {s['created']:,} created / {s['total']:,} total ({100 * s['created'] / max(1, s['total']):.0f}%)",
            flush=True,
        )
    elapsed = time.time() - t0
    resp = st.get("response", {})
    http_json("POST", f"/{original}/_refresh")
    after = http_json("GET", f"/{original}/_count")["count"]
    print(
        f"  done in {elapsed:.1f}s: total={resp.get('total'):,} "
        f"created={resp.get('created'):,} updated={resp.get('updated'):,} "
        f"failures={len(resp.get('failures', []))} -> {after:,} docs"
    )
