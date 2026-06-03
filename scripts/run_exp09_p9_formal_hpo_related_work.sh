#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/background_logs

LOCK_ROOT="results/background_locks"
LOCK_DIR="${LOCK_ROOT}/run_exp09_p9_formal_wrapper.lock"

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
        echo "[wait] $(timestamp) p9_wrapper_lock_held_by=${owner_pid}"
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

p9_formal_ready() {
  ./.venv/bin/python - <<'PY'
import json
from pathlib import Path

for seed in (42, 123, 2024, 3407, 9999):
    run_dir = Path("results") / f"EXP09_P9_formal_hpo_related_work_s{seed}"
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
    if not ok:
        raise SystemExit(1)
print("ready")
PY
}

group_ready() {
  ./.venv/bin/python - <<'PY'
import json
from pathlib import Path
import pandas as pd

base = Path("results/paper_tables/p9_related_work_hpo")
manifest = base / "paper_aggregate_manifest.json"
main = base / "paper_main_comparison.csv"
seed = base / "paper_seed_summary.csv"
paired = base / "paper_paired_statistics.csv"
source_list = base / "source_run_dirs.txt"
if not (manifest.exists() and main.exists() and seed.exists() and paired.exists() and source_list.exists()):
    raise SystemExit(1)
payload = json.loads(manifest.read_text(encoding="utf-8"))
formal = payload.get("formal_filter") or {}
rows = int((((payload.get("row_counts") or {}).get("paper_main_comparison")) or 0))
if not (
    rows > 0
    and formal.get("required_protocol_id") == "core13_v2_full_reset_20260522"
    and formal.get("required_data_cutoff_date") == "2026-05-20"
    and formal.get("require_formal_manifest") is True
    and formal.get("require_availability_mask_contract") is True
):
    raise SystemExit(1)
frame = pd.read_csv(main)
source_runs = set(frame.get("source_run", pd.Series(dtype="object")).dropna().astype(str).unique())
source_files = set(frame.get("source_file", pd.Series(dtype="object")).dropna().astype(str).unique())
expected_runs = {f"EXP09_P9_formal_hpo_related_work_s{seed}" for seed in (42, 123, 2024, 3407, 9999)}
if source_runs != expected_runs or source_files != {"hpo_model_final_comparison.csv"}:
    raise SystemExit(1)
actual_run_dirs = [
    (Path(line.strip()) if Path(line.strip()).is_absolute() else (Path.cwd() / line.strip())).resolve()
    for line in source_list.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
expected_run_dirs = [(Path("results") / f"EXP09_P9_formal_hpo_related_work_s{seed}").resolve() for seed in (42, 123, 2024, 3407, 9999)]
expected_run_dirs = list(expected_run_dirs)
if set(actual_run_dirs) != set(expected_run_dirs):
    raise SystemExit(1)
aggregate_mtime = max(
    manifest.stat().st_mtime,
    main.stat().st_mtime,
    seed.stat().st_mtime,
    paired.stat().st_mtime,
    source_list.stat().st_mtime,
)
for run_dir in actual_run_dirs:
    source_manifest = run_dir / "logs" / "run_manifest.json"
    if not source_manifest.exists() or source_manifest.stat().st_mtime > aggregate_mtime:
        raise SystemExit(1)
print("ready")
PY
}

seeds=(42 123 2024 3407 9999)
launched_pids=()
active_patterns=()

acquire_lock

if group_ready >/dev/null 2>&1; then
  echo "[skip] $(timestamp) p9_related_work_hpo_already_ready"
  exit 0
fi

pids=()
for seed in "${seeds[@]}"; do
  run_name="EXP09_P9_formal_hpo_related_work_s${seed}"
  active_pattern="--config configs/paper/hpo_equal_budget_related_work_seed_runner.yaml --seed ${seed} --run-name ${run_name}"
  if p9_formal_ready >/dev/null 2>&1; then
    break
  fi
  if process_active "${active_pattern}"; then
    echo "[wait] $(timestamp) ${run_name} already_running"
    active_patterns+=("${active_pattern}")
    continue
  fi
  log_path="results/background_logs/${run_name}.log"
  echo "[start] $(timestamp) ${run_name} log=${log_path}"
  ./.venv/bin/python -m src.experiments.run_experiment \
    --config configs/paper/hpo_equal_budget_related_work_seed_runner.yaml \
    --seed "${seed}" \
    --run-name "${run_name}" \
    >"${log_path}" 2>&1 &
  launched_pids+=("$!")
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

if ! p9_formal_ready >/dev/null 2>&1; then
  echo "[error] $(timestamp) p9_formal_not_ready_after_wait"
  exit 1
fi

if [ "$fail" -ne 0 ]; then
  echo "[error] $(timestamp) one_or_more_seeds_failed"
  exit 1
fi

if group_ready >/dev/null 2>&1; then
  echo "[skip] $(timestamp) p9_related_work_hpo_already_ready"
  exit 0
fi

./.venv/bin/python -m src.experiments.paper_aggregate \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s42 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s123 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s2024 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s3407 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s9999 \
  --output-dir results/paper_tables/p9_related_work_hpo \
  --benchmark-model ppo_dqn_hierarchical_reimplementation \
  --benchmark-model hybrid_dqn_optimizer_sharpe_maximization \
  --paper-group-id p9_related_work_hpo \
  --protocol-id core13_v2_full_reset_20260522 \
  --data-cutoff-date 2026-05-20 \
  --require-formal-manifest \
  --require-availability-mask-contract

echo "[done] $(timestamp) EXP09_P9_formal_hpo_related_work"
