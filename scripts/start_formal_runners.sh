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
  "EXP05_wrapper" \
  'scripts/run_exp05_p7_formal_hpo_main_native.sh|scripts/recover_exp05_p7_formal_seed.py --config configs/paper/hpo_equal_budget_main_native_seed_runner.yaml' \
  'bash scripts/run_exp05_p7_formal_hpo_main_native.sh' \
  'results/background_logs/EXP05_wrapper.log'

start_if_needed \
  "EXP09_wrapper" \
  'scripts/run_exp09_p9_formal_hpo_related_work.sh|--config configs/paper/hpo_equal_budget_related_work_seed_runner.yaml --seed ' \
  'bash scripts/run_exp09_p9_formal_hpo_related_work.sh' \
  'results/background_logs/EXP09_wrapper.log'

start_if_needed \
  "EXP35B_wrapper" \
  'scripts/run_exp35b_p16_gate_and_formal.sh|scripts/recover_exp35_p16_formal_seed.py --config configs/paper/p16_ra_gt_rcpo_formal_seed_runner.yaml' \
  'bash scripts/run_exp35b_p16_gate_and_formal.sh' \
  'results/background_logs/EXP35B_wrapper.log'

echo "started"
