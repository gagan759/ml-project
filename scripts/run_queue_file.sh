#!/usr/bin/env bash
# GPU job queue read from a file, one job per line: "<task> <recipe> [extra args]" (e.g. "5class dinov2_s_robust").
# Jobs are taken from the top one at a time, so the file can be edited while the queue runs.
# Usage: bash scripts/run_queue_file.sh [queue-file] [pid-to-wait-for]
set -u
cd "$(dirname "$0")/.."
Q="${1:-experiments/queue.txt}"
WAIT_PID="${2:-}"
PY=.venv/Scripts/python
if [ -n "$WAIT_PID" ]; then
  echo "=== $(date +%H:%M:%S) waiting for PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
fi
while true; do
  line=$(grep -v '^[[:space:]]*$' "$Q" 2>/dev/null | head -n1)
  [ -z "$line" ] && break
  # remove exactly that first non-empty line
  awk -v l="$line" 'BEGIN{d=0} { if (!d && $0==l) { d=1; next } print }' "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
  read -r task recipe extra <<<"$line"
  echo "=== $(date +%H:%M:%S) $task $recipe ${extra:-}"
  "$PY" scripts/21_train_cv.py --recipe "$recipe" --task "$task" ${extra:-} || echo "!!! $task $recipe failed"
done
echo "=== $(date +%H:%M:%S) queue done"
