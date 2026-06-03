#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/background_logs

LOCK_ROOT="results/background_locks"
LOCK_DIR="${LOCK_ROOT}/run_exp05_p7_formal_wrapper.lock"

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
        echo "[wait] $(timestamp) p7_wrapper_lock_held_by=${owner_pid}"
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

seed_ready() {
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

start_seed() {
  local seed="$1"
  local run_name="EXP05_P7_formal_hpo_main_native_s${seed}"
  local log_path="results/background_logs/${run_name}.log"
  echo "[start] $(timestamp) ${run_name} log=${log_path}" >&2
  ./.venv/bin/python scripts/recover_exp05_p7_formal_seed.py \
    --config configs/paper/hpo_equal_budget_main_native_seed_runner.yaml \
    --seed "${seed}" \
    --run-name "${run_name}" \
    >"${log_path}" 2>&1 &
  launched_pids+=("$!")
}

seeds=(42 123 2024 3407 9999)
launched_pids=()
active_patterns=()

acquire_lock

for seed in "${seeds[@]}"; do
  run_name="EXP05_P7_formal_hpo_main_native_s${seed}"
  active_pattern="scripts/recover_exp05_p7_formal_seed.py --config configs/paper/hpo_equal_budget_main_native_seed_runner.yaml --seed ${seed} --run-name ${run_name}"
  if seed_ready "${run_name}"; then
    echo "[skip] $(timestamp) ${run_name} already_ready"
    continue
  fi
  if process_active "${active_pattern}"; then
    echo "[wait] $(timestamp) ${run_name} already_running"
    active_patterns+=("${active_pattern}")
    continue
  fi
  start_seed "${seed}"
done

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
  run_name="EXP05_P7_formal_hpo_main_native_s${seed}"
  if ! seed_ready "${run_name}"; then
    echo "[error] $(timestamp) ${run_name} not_ready_after_wait"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "[error] $(timestamp) one_or_more_seeds_failed"
  exit 1
fi

while process_active 'scripts/run_exp11_p1_hpo_final_main_native_from_hpo.sh'; do
  echo "[wait] $(timestamp) p1_from_hpo_already_running"
  sleep 20
done

bash scripts/run_exp11_p1_hpo_final_main_native_from_hpo.sh

echo "[done] $(timestamp) EXP05_P7_formal_hpo_main_native"
