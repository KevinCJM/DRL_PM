#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/background_logs

LOCK_ROOT="results/background_locks"
LOCK_DIR="${LOCK_ROOT}/run_exp35b_p16_formal_wrapper.lock"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

release_lock() {
  if [ -d "$LOCK_DIR" ] && [ -f "${LOCK_DIR}/pid" ] && [ "$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)" = "$$" ]; then
    rm -rf "$LOCK_DIR"
  fi
}

acquire_lock() {
  mkdir -p "$LOCK_ROOT"
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    if [ -f "${LOCK_DIR}/pid" ]; then
      owner_pid="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
      if [ -n "${owner_pid}" ] && ps -p "${owner_pid}" >/dev/null 2>&1; then
        echo "[wait] $(timestamp) p16_wrapper_lock_held_by=${owner_pid}"
        sleep 20
        continue
      fi
    fi
    rm -rf "$LOCK_DIR"
  done
  echo "$$" > "${LOCK_DIR}/pid"
  trap release_lock EXIT
}

process_active() {
  local pattern="$1"
  pgrep -f "$pattern" >/dev/null 2>&1
}

formal_seed_ready() {
  local run_name="$1"
  ./.venv/bin/python - <<'PY' "$run_name"
import json
import sys
from pathlib import Path

run_dir = Path("results") / sys.argv[1]
manifest_path = run_dir / "logs" / "run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
ok = (
    manifest.get("status") == "success"
    and manifest.get("diagnostic_status") == "formal"
    and manifest.get("rankable_in_unified_table") is True
    and manifest.get("protocol_id") == "core13_v2_full_reset_20260522"
    and str(manifest.get("data_cutoff_date")) == "2026-05-20"
)
raise SystemExit(0 if ok else 1)
PY
}

export_ready() {
  ./.venv/bin/python - <<'PY'
import json
from pathlib import Path

manifest_path = Path("results/EXP36_P1_fixed_deterministic_formal_export/logs/run_manifest.json")
if not manifest_path.exists():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
ok = (
    manifest.get("status") == "success"
    and manifest.get("diagnostic_status") == "formal"
    and manifest.get("rankable_in_unified_table") is True
    and manifest.get("protocol_id") == "core13_v2_full_reset_20260522"
    and str(manifest.get("data_cutoff_date")) == "2026-05-20"
)
raise SystemExit(0 if ok else 1)
PY
}

start_formal_seed() {
  local seed="$1"
  local run_name="EXP35_P16_formal_ra_gt_rcpo_s${seed}"
  local seed_log="results/background_logs/${run_name}.log"
  echo "[start] $(timestamp) ${run_name} log=${seed_log}" >&2
  ./.venv/bin/python scripts/recover_exp35_p16_formal_seed.py \
    --config configs/paper/p16_ra_gt_rcpo_formal_seed_runner.yaml \
    --seed "${seed}" \
    --run-name "${run_name}" \
    >"${seed_log}" 2>&1 &
  launched_pids+=("$!")
}

start_export() {
  local export_log="results/background_logs/EXP36_P1_fixed_deterministic_formal_export.log"
  echo "[run] $(timestamp) p16_deterministic_export" >&2
  ./.venv/bin/python -m src.experiments.run_experiment \
    --config configs/paper/p16_p1_fixed_deterministic_formal_export.yaml \
    --seed 42 \
    --run-name EXP36_P1_fixed_deterministic_formal_export \
    >"${export_log}" 2>&1 &
  launched_pids+=("$!")
}

ref_csv="results/paper_tables/p16_validation_references/validation_reference_comparison.csv"
pilot_cmp="results/EXP35_P16_ra_gt_rcpo_pilot_s42/metrics/hpo_model_final_comparison.csv"
pilot_trials="results/EXP35_P16_ra_gt_rcpo_pilot_s42/logs/hpo_trials.csv"
export_manifest="results/EXP36_P1_fixed_deterministic_formal_export/logs/run_manifest.json"

acquire_lock

echo "[wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) waiting_for_p16_refs_and_pilot"
until [ -f "$ref_csv" ] && [ -f "$pilot_cmp" ] && [ -f "$pilot_trials" ]; do
  sleep 20
done

echo "[run] $(date -u +%Y-%m-%dT%H:%M:%SZ) p16_promotion_gate"
./.venv/bin/python scripts/evaluate_p16_promotion_gate.py \
  --pilot-run-dir results/EXP35_P16_ra_gt_rcpo_pilot_s42 \
  --reference-dir results/paper_tables/p16_validation_references \
  --output-dir results/paper_tables/p16_promotion_gate \
  --average-cost-per-step-budget 0.001 \
  --cost-budget-tolerance 1.0e-6

passed=$(
  ./.venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("results/paper_tables/p16_promotion_gate/promotion_gate_report.csv")
if not path.exists():
    print("0")
else:
    frame = pd.read_csv(path)
    ok = False
    if not frame.empty and "promotion_gate_passed" in frame.columns:
        ok = frame["promotion_gate_passed"].fillna(False).astype(bool).any()
    print("1" if ok else "0")
PY
)

echo "[gate] $(date -u +%Y-%m-%dT%H:%M:%SZ) p16_passed=${passed}"
if [ "$passed" != "1" ]; then
  echo "[skip] $(date -u +%Y-%m-%dT%H:%M:%SZ) p16_formal_not_started"
  exit 0
fi

echo "[run] $(date -u +%Y-%m-%dT%H:%M:%SZ) p16_formal_seed_runs"
seeds=(42 123 2024 3407 9999)
launched_pids=()
active_patterns=()
for seed in "${seeds[@]}"; do
  run_name="EXP35_P16_formal_ra_gt_rcpo_s${seed}"
  active_pattern="EXP35_P16_formal_ra_gt_rcpo_s${seed}"
  if formal_seed_ready "${run_name}"; then
    echo "[skip] $(timestamp) ${run_name} already_ready"
    continue
  fi
  if process_active "${active_pattern}"; then
    echo "[wait] $(timestamp) ${run_name} already_running"
    active_patterns+=("${active_pattern}")
    continue
  fi
  start_formal_seed "${seed}"
done

export_pattern='EXP36_P1_fixed_deterministic_formal_export'
if export_ready; then
  echo "[skip] $(timestamp) p16_deterministic_export_already_ready"
elif process_active "${export_pattern}"; then
  echo "[wait] $(timestamp) p16_deterministic_export_already_running"
  active_patterns+=("${export_pattern}")
else
  start_export
fi

fail=0
for pid in "${launched_pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

for pattern in "${active_patterns[@]:-}"; do
  [ -n "${pattern}" ] || continue
  while process_active "${pattern}"; do
    sleep 20
  done
done

for seed in "${seeds[@]}"; do
  run_name="EXP35_P16_formal_ra_gt_rcpo_s${seed}"
  if ! formal_seed_ready "${run_name}"; then
    echo "[error] $(timestamp) ${run_name} not_ready_after_wait"
    fail=1
  fi
done

if [ -f "$export_manifest" ] && ! export_ready; then
  echo "[error] $(timestamp) p16_deterministic_export_manifest_not_ready"
  fail=1
fi
if [ ! -f "$export_manifest" ]; then
  echo "[error] $(timestamp) p16_deterministic_export_manifest_missing"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "[error] $(date -u +%Y-%m-%dT%H:%M:%SZ) p16_formal_or_export_failed"
  exit 1
fi

echo "[done] $(date -u +%Y-%m-%dT%H:%M:%SZ) p16_gate_formal_and_export"
