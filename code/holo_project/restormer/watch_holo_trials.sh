#!/usr/bin/env bash
# Run the two 25,500-iteration decision trials in order and monitor the
# actual runner process.  Do not infer process state by grepping log text:
# messages such as "End of training" contain spaces and are not a reliable
# completion signal.
set -Eeuo pipefail

PROJECT_ROOT="${HOLO_PROJECT_ROOT:?Set HOLO_PROJECT_ROOT first}"
DATA_ROOT="${HOLO_DATA_ROOT:?Set HOLO_DATA_ROOT first}"
REPO_ROOT="$PROJECT_ROOT/restormer"
RUN_DIR="$PROJECT_ROOT/experiments"
RUN_LOG="$RUN_DIR/serial_run.log"
PID_FILE="$RUN_DIR/serial_run.pid"
STATUS_FILE="$RUN_DIR/serial_run.status"
LOCK_FILE="$RUN_DIR/serial_run.lock"

SEEDS="${HOLO_SEEDS:-10}"
MICRO_BATCH_SIZE="${HOLO_MICRO_BATCH_SIZE:-2}"
ITERS="${HOLO_ITERS:-25500}"

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"

# Prevent accidentally starting a second pair of trials.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "A holo trial runner is already active: $RUN_DIR/serial_run.pid" >&2
  exit 2
fi

if [[ -s "$PID_FILE" ]]; then
  old_pid="$(<"$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "A holo trial runner is already active (PID $old_pid)." >&2
    exit 2
  fi
fi

rm -f "$STATUS_FILE"
printf 'STARTED %s\n' "$(date --iso-8601=seconds)" >"$STATUS_FILE"

# run_holo_trials.py itself is the sequence controller: it runs MDTA first,
# and starts Band only after MDTA exits successfully (subprocess check=True).
set +e
PYTHONUNBUFFERED=1 python -u run_holo_trials.py \
  --run \
  --seeds "$SEEDS" \
  --models both \
  --iters "$ITERS" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  >"$RUN_LOG" 2>&1 &
runner_pid=$!
set -e
printf '%s\n' "$runner_pid" >"$PID_FILE"

set +e
wait "$runner_pid"
runner_rc=$?
set -e

if (( runner_rc == 0 )); then
  printf 'COMPLETED %s\n' "$(date --iso-8601=seconds)" >"$STATUS_FILE"
else
  printf 'FAILED %s exit_code=%s\n' "$(date --iso-8601=seconds)" "$runner_rc" >"$STATUS_FILE"
fi
rm -f "$PID_FILE"
exit "$runner_rc"
