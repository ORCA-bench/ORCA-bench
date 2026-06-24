#!/bin/bash
set -euo pipefail

set +e
python /tests/check_prediction.py 2>&1 | tee /logs/verifier/log.txt
status=${PIPESTATUS[0]}
set -e

# Preserve agent predictions even on verifier failure (Harbor deletes containers).
if [[ -f /app/report.md ]]; then
  mkdir -p /logs/verifier
  cp /app/report.md /logs/verifier/report.md 2>/dev/null || true
fi

exit $status
