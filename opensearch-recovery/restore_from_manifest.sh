#!/usr/bin/env bash
set -euo pipefail
MANIFEST="${1:?usage: $0 <restore_manifest.json>}"
OS_URL="${OS_URL:-http://localhost:9200}"
REPO="${REPO:-scheduled_backups}"

count=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
i=0
python3 -c "import json; [print(json.dumps(e)) for e in json.load(open('$MANIFEST'))]" | \
while read -r entry; do
    i=$((i+1))
    snap=$(echo "$entry" | python3 -c "import json,sys; print(json.load(sys.stdin)['snapshot'])")
    idx=$(echo  "$entry" | python3 -c "import json,sys; print(json.load(sys.stdin)['index'])")
    life=$(echo "$entry" | python3 -c "import json,sys; print(json.load(sys.stdin)['lifetime'])")
    target="${idx}-lifetime${life}"

    # Skip if already restored (idempotent)
    if curl -sf "${OS_URL}/${target}/_stats" >/dev/null 2>&1; then
        echo "[$i/$count] skip ${target} (already exists)"
        continue
    fi

    echo "[$i/$count] restore ${snap} → ${target}"
    curl -sf -X POST "${OS_URL}/_snapshot/${REPO}/${snap}/_restore?wait_for_completion=true" \
        -H 'Content-Type: application/json' \
        -d "$(cat <<EOF
{
  "indices": "${idx}",
  "rename_pattern": ".+",
  "rename_replacement": "${target}",
  "include_global_state": false,
  "include_aliases": false
}
EOF
)" >/dev/null
done

echo "Done. Verify: curl -s ${OS_URL}/_cat/indices?v | wc -l"
