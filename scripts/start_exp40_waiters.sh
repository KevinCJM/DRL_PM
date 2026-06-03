#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/background_logs

process_active() {
  local pattern="$1"
  pgrep -f "$pattern" >/dev/null 2>&1
}

start_if_needed() {
  local name="$1"
  local pattern="$2"
  local cmd="$3"
  local log_path="$4"
  if process_active "$pattern"; then
    echo "[skip] ${name} already_running"
    return 0
  fi
  nohup bash -lc "$cmd" > "$log_path" 2>&1 &
  echo "[start] ${name}"
}

start_if_needed \
  "EXP40C_p14_early" \
  'scripts/exp40c_p14_early.sh' \
  'bash scripts/exp40c_p14_early.sh' \
  'results/background_logs/EXP40C_p14_early.log'

start_if_needed \
  "EXP40D_p16_final_early" \
  'scripts/exp40d_p16_final_early.sh' \
  'bash scripts/exp40d_p16_final_early.sh' \
  'results/background_logs/EXP40D_p16_final_early.log'

start_if_needed \
  "EXP40E_bundle_readiness" \
  'scripts/exp40e_bundle_readiness.sh' \
  'bash scripts/exp40e_bundle_readiness.sh' \
  'results/background_logs/EXP40E_bundle_readiness.log'

echo "started"
