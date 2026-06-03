#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

LOCK_ROOT="results/background_locks"
LOCK_DIR="${LOCK_ROOT}/run_exp11_p1_from_hpo.lock"

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
        echo "[wait] $(timestamp) p1_from_hpo_lock_held_by=${owner_pid}"
        sleep 20
        continue
      fi
    fi
    rm -rf "$LOCK_DIR"
  done
  echo "$$" > "${LOCK_DIR}/pid"
  trap release_lock EXIT
}

group_ready() {
  ./.venv/bin/python - <<'PY'
import json
from pathlib import Path

SOURCE_FILES = [
    "logs/run_manifest.json",
    "metrics/hpo_model_final_comparison.csv",
    "metrics/hpo_model_final_daily_returns.csv",
    "metrics/hpo_model_final_daily_weights.csv",
    "metrics/hpo_model_final_daily_turnover.csv",
    "metrics/hpo_model_final_daily_rebalance.csv",
    "metrics/hpo_model_final_daily_costs.csv",
]

def p1_source_fresh(run_dir: Path) -> bool:
    meta_path = run_dir / "logs" / "p1_from_hpo_source.json"
    manifest_path = run_dir / "logs" / "run_manifest.json"
    comparison_path = run_dir / "metrics" / "baseline_comparison.csv"
    if not (meta_path.exists() and manifest_path.exists() and comparison_path.exists()):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    seed = int(meta.get("seed"))
    expected_source = (Path("results") / f"EXP05_P7_formal_hpo_main_native_s{seed}").resolve()
    source_dir_raw = str(meta.get("source_run_dir") or "")
    if not source_dir_raw:
        return False
    source_dir = (Path(source_dir_raw) if Path(source_dir_raw).is_absolute() else (Path.cwd() / source_dir_raw)).resolve()
    if source_dir != expected_source:
        return False
    source_paths = [source_dir / rel for rel in SOURCE_FILES]
    if not all(path.exists() for path in source_paths):
        return False
    source_mtime = max(path.stat().st_mtime for path in source_paths)
    output_anchor = min(manifest_path.stat().st_mtime, comparison_path.stat().st_mtime, meta_path.stat().st_mtime)
    return source_mtime <= output_anchor

manifest_path = Path("results/paper_tables/main_hpo_5seed/paper_aggregate_manifest.json")
main_path = Path("results/paper_tables/main_hpo_5seed/paper_main_comparison.csv")
seed_path = Path("results/paper_tables/main_hpo_5seed/paper_seed_summary.csv")
paired_path = Path("results/paper_tables/main_hpo_5seed/paper_paired_statistics.csv")
source_list = Path("results/paper_tables/main_hpo_5seed/source_run_dirs.txt")
if not (manifest_path.exists() and main_path.exists() and seed_path.exists() and paired_path.exists() and source_list.exists()):
    raise SystemExit(1)
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
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
import pandas as pd
main = pd.read_csv(main_path)
expected = {f"EXP11_P1_hpo_final_main_native_from_hpo_s{seed}" for seed in (42, 123, 2024, 3407, 9999)}
actual = set(main.get("source_run", pd.Series(dtype="object")).dropna().astype(str).unique())
source_files = set(main.get("source_file", pd.Series(dtype="object")).dropna().astype(str).unique())
ok = actual == expected and source_files == {"baseline_comparison.csv"}
actual_run_dirs = [
    (Path(line.strip()) if Path(line.strip()).is_absolute() else (Path.cwd() / line.strip())).resolve()
    for line in source_list.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
expected_run_dirs = [
    (Path("results") / f"EXP11_P1_hpo_final_main_native_from_hpo_s{seed}").resolve()
    for seed in (42, 123, 2024, 3407, 9999)
]
if set(actual_run_dirs) != set(expected_run_dirs):
    ok = False
aggregate_mtime = max(
    manifest_path.stat().st_mtime,
    main_path.stat().st_mtime,
    seed_path.stat().st_mtime,
    paired_path.stat().st_mtime,
    source_list.stat().st_mtime,
)
for run_dir in actual_run_dirs:
    source_manifest = run_dir / "logs" / "run_manifest.json"
    if not source_manifest.exists() or source_manifest.stat().st_mtime > aggregate_mtime:
        ok = False
        break
    if not p1_source_fresh(run_dir):
        ok = False
        break
raise SystemExit(0 if ok else 1)
PY
}

seed_ready() {
  local seed="$1"
  ./.venv/bin/python - <<'PY' "$seed"
import json
import sys
from pathlib import Path

SOURCE_FILES = [
    "logs/run_manifest.json",
    "metrics/hpo_model_final_comparison.csv",
    "metrics/hpo_model_final_daily_returns.csv",
    "metrics/hpo_model_final_daily_weights.csv",
    "metrics/hpo_model_final_daily_turnover.csv",
    "metrics/hpo_model_final_daily_rebalance.csv",
    "metrics/hpo_model_final_daily_costs.csv",
]

seed = int(sys.argv[1])
run_dir = Path("results") / f"EXP11_P1_hpo_final_main_native_from_hpo_s{seed}"
manifest_path = run_dir / "logs" / "run_manifest.json"
comparison_path = run_dir / "metrics" / "baseline_comparison.csv"
source_meta = run_dir / "logs" / "p1_from_hpo_source.json"
if not (manifest_path.exists() and comparison_path.exists() and source_meta.exists()):
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
ok = (
    manifest.get("status") == "success"
    and manifest.get("diagnostic_status") == "formal"
    and manifest.get("rankable_in_unified_table") is True
    and manifest.get("protocol_id") == "core13_v2_full_reset_20260522"
    and str(manifest.get("data_cutoff_date")) == "2026-05-20"
    and manifest.get("source_hpo_comparison_file") == "hpo_model_final_comparison.csv"
)
meta = json.loads(source_meta.read_text(encoding="utf-8"))
expected_source = (Path("results") / f"EXP05_P7_formal_hpo_main_native_s{seed}").resolve()
source_dir_raw = str(meta.get("source_run_dir") or "")
if not source_dir_raw:
    ok = False
else:
    source_dir = (Path(source_dir_raw) if Path(source_dir_raw).is_absolute() else (Path.cwd() / source_dir_raw)).resolve()
    if source_dir != expected_source:
        ok = False
    else:
        source_paths = [source_dir / rel for rel in SOURCE_FILES]
        if not all(path.exists() for path in source_paths):
            ok = False
        else:
            source_mtime = max(path.stat().st_mtime for path in source_paths)
            output_anchor = min(manifest_path.stat().st_mtime, comparison_path.stat().st_mtime, source_meta.stat().st_mtime)
            ok = ok and source_mtime <= output_anchor
raise SystemExit(0 if ok else 1)
PY
}

source_ready() {
  local seed="$1"
  ./.venv/bin/python - <<'PY' "$seed"
import json
import sys
from pathlib import Path

seed = int(sys.argv[1])
run_dir = Path("results") / f"EXP05_P7_formal_hpo_main_native_s{seed}"
manifest_path = run_dir / "logs" / "run_manifest.json"
comparison = run_dir / "metrics" / "hpo_model_final_comparison.csv"
returns = run_dir / "metrics" / "hpo_model_final_daily_returns.csv"
if not (manifest_path.exists() and comparison.exists() and returns.exists()):
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
ok = (
    manifest.get("status") == "success"
    and manifest.get("diagnostic_status") == "formal"
    and manifest.get("rankable_in_unified_table") is True
    and manifest.get("protocol_id") == "core13_v2_full_reset_20260522"
    and str(manifest.get("data_cutoff_date")) == "2026-05-20"
    and manifest.get("execution_activity_protocol") == "daily_gate_with_cost_constraint"
    and manifest.get("turnover_optimization_protocol_id") == "turnover_active_v1"
    and manifest.get("scheduler_blocks_model_actions") is False
    and manifest.get("activity_gate_enforced") is True
)
raise SystemExit(0 if ok else 1)
PY
}

echo "[start] $(timestamp) p1_from_hpo_main_native"

acquire_lock

if group_ready >/dev/null 2>&1; then
  echo "[skip] $(timestamp) main_hpo_5seed_already_ready_from_p1"
  exit 0
fi

for seed in 42 123 2024 3407 9999; do
  if ! source_ready "$seed" >/dev/null 2>&1; then
    echo "[wait] $(timestamp) waiting_for_p7_source_seed=${seed}"
    exit 1
  fi
done

for seed in 42 123 2024 3407 9999; do
  if seed_ready "$seed" >/dev/null 2>&1; then
    echo "[skip] $(timestamp) p1_seed_ready=${seed}"
    continue
  fi
  echo "[run] $(timestamp) p1_seed=${seed}"
  ./.venv/bin/python scripts/export_p1_main_native_from_hpo.py \
    --config configs/paper/baseline_comparison_main_native_from_hpo.yaml \
    --source-run-dir "results/EXP05_P7_formal_hpo_main_native_s${seed}" \
    --output-run-dir "results/EXP11_P1_hpo_final_main_native_from_hpo_s${seed}"
done

echo "[run] $(timestamp) main_hpo_5seed"
./.venv/bin/python -m src.experiments.paper_aggregate \
  --run-dir results/EXP11_P1_hpo_final_main_native_from_hpo_s42 \
  --run-dir results/EXP11_P1_hpo_final_main_native_from_hpo_s123 \
  --run-dir results/EXP11_P1_hpo_final_main_native_from_hpo_s2024 \
  --run-dir results/EXP11_P1_hpo_final_main_native_from_hpo_s3407 \
  --run-dir results/EXP11_P1_hpo_final_main_native_from_hpo_s9999 \
  --output-dir results/paper_tables/main_hpo_5seed \
  --benchmark-model full_dqn_gated_multitask_cnn_ppo \
  --benchmark-model cnn_ppo_native \
  --benchmark-model pgportfolio_eiie_native \
  --paper-group-id main_hpo_5seed \
  --protocol-id core13_v2_full_reset_20260522 \
  --data-cutoff-date 2026-05-20 \
  --require-formal-manifest \
  --require-availability-mask-contract

echo "[done] $(timestamp) p1_from_hpo_main_native"
