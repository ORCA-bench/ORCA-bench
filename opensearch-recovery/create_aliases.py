#!/usr/bin/env python3
"""Create aliases that union all -lifetimeN indices under the original daily index name."""

import json
import sys
import urllib.request
from collections import defaultdict

OS_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9200"

# Discover all indices matching *-lifetimeN pattern
with urllib.request.urlopen(f"{OS_URL}/_cat/indices?format=json") as r:
    all_indices = [i["index"] for i in json.load(r)]

groups = defaultdict(list)  # original_name -> [lifetime_indices]
for idx in all_indices:
    if "-lifetime" in idx:
        original = idx.rsplit("-lifetime", 1)[0]
        groups[original].append(idx)

actions = []
for original, members in groups.items():
    for m in sorted(members):
        actions.append({"add": {"index": m, "alias": original}})

if not actions:
    print("No -lifetimeN indices found; nothing to alias.")
    sys.exit(0)

req = urllib.request.Request(
    f"{OS_URL}/_aliases",
    data=json.dumps({"actions": actions}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as r:
    print(json.load(r))
print(f"Created/updated {len(actions)} alias bindings across {len(groups)} aliases.")
