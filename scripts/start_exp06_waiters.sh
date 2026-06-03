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
  "EXP06A_main_plus_p9" \
  'scripts/exp06a_main_plus_p9.sh' \
  'bash scripts/exp06a_main_plus_p9.sh' \
  'results/background_logs/EXP06A_main_plus_p9_only.log'

start_if_needed \
  "EXP06C_p2_p8_diagnostics" \
  'scripts/exp06c_p2_p8_diagnostics_flexible.sh' \
  'bash scripts/exp06c_p2_p8_diagnostics_flexible.sh' \
  'results/background_logs/EXP06C_p2_p8_diagnostics_flexible.log'

echo "started"
