#!/usr/bin/env bash
# After the active MDTA -> Band run exits successfully, use each selected best
# checkpoint to export only configured visual_samples as LR/SR/HR BMPs. This
# visual-only pass does not calculate test metrics.
set -Eeuo pipefail

RUNNER_PID="${1:?Usage: export_holo_paper_images.sh RUNNER_PID}"
if [[ ! "$RUNNER_PID" =~ ^[0-9]+$ ]]; then
  echo "Invalid runner PID: $RUNNER_PID" >&2
  exit 2
fi

PROJECT_ROOT="${HOLO_PROJECT_ROOT:-/root/autodl-tmp/holo_project}"
DATA_ROOT="${HOLO_DATA_ROOT:-/root/autodl-tmp}"
REPO_ROOT="$PROJECT_ROOT/restormer"
EXPERIMENTS_ROOT="$PROJECT_ROOT/experiments"
STATUS_FILE="$EXPERIMENTS_ROOT/paper_image_export.status"
LOG_FILE="$EXPERIMENTS_ROOT/paper_image_export_$(date +%Y%m%d_%H%M%S).log"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

mkdir -p "$EXPERIMENTS_ROOT"
printf 'WAITING %s runner_pid=%s\n' "$(date --iso-8601=seconds)" "$RUNNER_PID" >"$STATUS_FILE"

while kill -0 "$RUNNER_PID" 2>/dev/null; do
  sleep 60
done
sleep 5

export HOLO_PROJECT_ROOT="$PROJECT_ROOT"
export HOLO_DATA_ROOT="$DATA_ROOT"
cd "$REPO_ROOT"

if ! grep -Fq 'Prepared trials.' "$EXPERIMENTS_ROOT/serial_run.log"; then
  printf 'FAILED %s runner_exited_without_success_marker\n' "$(date --iso-8601=seconds)" >"$STATUS_FILE"
  echo "Training runner exited without success marker; no visual export started." >"$LOG_FILE"
  exit 1
fi

for model_name in mdta band; do
  exp_dir="$EXPERIMENTS_ROOT/holo_restormer_${model_name}_25500_seed10_mb2"
  best_record="$exp_dir/best_checkpoint.json"
  if [[ ! -f "$best_record" ]]; then
    printf 'FAILED %s missing_best_checkpoint_%s\n' "$(date --iso-8601=seconds)" "$model_name" >"$STATUS_FILE"
    exit 1
  fi
  best_iter="$("$PYTHON_BIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["iter"]))' "$best_record")"
  checkpoint="$exp_dir/models/net_g_best_${best_iter}.pth"
  if [[ ! -f "$checkpoint" ]]; then
    printf 'FAILED %s missing_checkpoint_%s\n' "$(date --iso-8601=seconds)" "$checkpoint" >"$STATUS_FILE"
    exit 1
  fi
  model_log="$(find "$exp_dir" -maxdepth 1 -type f -name 'train_*.log' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -z "$model_log" ]] || ! grep -Fq 'End of training.' "$model_log"; then
    printf 'FAILED %s incomplete_training_log_%s\n' "$(date --iso-8601=seconds)" "$model_name" >"$STATUS_FILE"
    exit 1
  fi
done

printf 'EXPORTING %s\n' "$(date --iso-8601=seconds)" >"$STATUS_FILE"
for model_name in mdta band; do
  exp_dir="$EXPERIMENTS_ROOT/holo_restormer_${model_name}_25500_seed10_mb2"
  best_iter="$("$PYTHON_BIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["iter"]))' "$exp_dir/best_checkpoint.json")"
  "$PYTHON_BIN" evaluate_holo.py \
    --option "$exp_dir/options.yml" \
    --checkpoint "$exp_dir/models/net_g_best_${best_iter}.pth" \
    --split test --visual-only >>"$LOG_FILE" 2>&1
  sr_count="$(find "$exp_dir/results/test" -maxdepth 1 -type f -name '*_SR.bmp' | wc -l)"
  if [[ "$sr_count" -ne 10 ]]; then
    printf 'FAILED %s %s_expected_10_SR_got_%s\n' "$(date --iso-8601=seconds)" "$model_name" "$sr_count" >"$STATUS_FILE"
    exit 1
  fi
done

printf 'COMPLETED %s\n' "$(date --iso-8601=seconds)" >"$STATUS_FILE"
