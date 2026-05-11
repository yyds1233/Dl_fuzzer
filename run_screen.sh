#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash /root/run_screen_with_timeout.sh 6h
#   MAX_RUNTIME=90m bash /root/run_screen_with_timeout.sh
#   LOG_DIR=/root/screen_logs bash /root/run_screen_with_timeout.sh 3600s
#
# Notes:
# - Runs from /root
# - Stops automatically after the requested duration using `timeout`
# - Writes stdout/stderr to a timestamped log file
# - Forces Python/stdout to be as unbuffered as possible

ROOT_DIR="/root"
LOG_DIR="${LOG_DIR:-/root/screen_logs}"
MAX_RUNTIME="${1:-${MAX_RUNTIME:-6h}}"

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/screen_run_${TIMESTAMP}.log"

if ! command -v timeout >/dev/null 2>&1; then
  echo "[ERROR] 'timeout' command not found. Install coreutils first." >&2
  exit 127
fi

if ! command -v stdbuf >/dev/null 2>&1; then
  echo "[ERROR] 'stdbuf' command not found. Install coreutils first." >&2
  exit 127
fi

cleanup() {
  local exit_code=$?
  if [[ $exit_code -eq 124 ]]; then
    echo "[$(date '+%F %T')] Reached time limit: $MAX_RUNTIME" | tee -a "$LOG_FILE"
  elif [[ $exit_code -ne 0 ]]; then
    echo "[$(date '+%F %T')] Exited with code: $exit_code" | tee -a "$LOG_FILE"
  else
    echo "[$(date '+%F %T')] Finished successfully." | tee -a "$LOG_FILE"
  fi
}
trap cleanup EXIT

cd "$ROOT_DIR"

echo "[$(date '+%F %T')] Starting run from $ROOT_DIR" | tee -a "$LOG_FILE"
echo "[$(date '+%F %T')] Max runtime: $MAX_RUNTIME" | tee -a "$LOG_FILE"
echo "[$(date '+%F %T')] Log file: $LOG_FILE" | tee -a "$LOG_FILE"

CMD=(
  python3 -m screen.cli.main 
  --harnesses_json /root/screen/auto_harness_all.json 
  --groups_map /root/screen/groups_map.json 
  --root /root/fuzz_output_result
  --epoch 30 
  --steps 0 
  --audit_every 3
  --audit_min_delta_files 200
  --audit_max_inputs 0 
  --slow_metric LH
  --cov_audit_script /root/screen/cov_global_union_audit.py 
  --cov_venv_activate /root/pytorch_cov/bin/activate 
  --primary_object /root/pytorch-cov-src/torch/lib/libtorch_cpu.so 
  --extra_object /root/pytorch-cov-src/torch/lib/libc10.so 
  --extra_object /root/pytorch-cov-src/torch/_C.cpython-310-x86_64-linux-gnu.so 
  --ignore_filename_regex ".*(site-packages|third_party|build).*"
  --disable_profiles
  --fuzz_flags "-ignore_timeouts=1 -rss_limit_mb=8192 -malloc_limit_mb=8192 -use_value_profile=1 -entropic=1"
)

echo "[$(date '+%F %T')] Command: ${CMD[*]}" | tee -a "$LOG_FILE"

set -x
PYTHONUNBUFFERED=1 stdbuf -oL -eL \
timeout --foreground --signal=TERM --kill-after=30s "$MAX_RUNTIME" "${CMD[@]}" \
2>&1 | tee -a "$LOG_FILE"