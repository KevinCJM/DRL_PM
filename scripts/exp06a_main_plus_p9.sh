#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/background_logs

LOCK_ROOT="results/background_locks"
LOCK_DIR="${LOCK_ROOT}/exp06a_main_plus_p9_waiter.lock"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

release_lock() {
  if [ -d "$LOCK_DIR" ] && [ -f "${LOCK_DIR}/pid" ] && [ "$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)" = "$$" ]; then
    rm -rf "$LOCK_DIR"
  fi
}

acquire_lock_or_exit() {
  mkdir -p "$LOCK_ROOT"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$$" > "${LOCK_DIR}/pid"
    trap release_lock EXIT
    return 0
  fi
  if [ -f "${LOCK_DIR}/pid" ]; then
    owner_pid="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
    if [ -n "${owner_pid}" ] && ps -p "${owner_pid}" >/dev/null 2>&1; then
      echo "[skip] $(timestamp) exp06a_waiter_already_running pid=${owner_pid}"
      exit 0
    fi
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
  echo "$$" > "${LOCK_DIR}/pid"
  trap release_lock EXIT
}

process_active() {
  local pattern="$1"
  pgrep -f "$pattern" >/dev/null 2>&1
}

ensure_p7_wrapper() {
  if p7_formal_ready >/dev/null 2>&1; then
    return 0
  fi
  if process_active 'scripts/run_exp05_p7_formal_hpo_main_native.sh'; then
    return 0
  fi
  if process_active 'scripts/recover_exp05_p7_formal_seed.py --config configs/paper/hpo_equal_budget_main_native_seed_runner.yaml'; then
    return 0
  fi
  echo "[run] $(timestamp) restart_p7_formal_wrapper"
  nohup bash scripts/run_exp05_p7_formal_hpo_main_native.sh \
    >> results/background_logs/EXP05_P7_formal_wrapper.supervisor.log 2>&1 &
}

ensure_p9_wrapper() {
  if group_ready "p9_related_work_hpo" >/dev/null 2>&1; then
    return 0
  fi
  if p9_formal_ready >/dev/null 2>&1; then
    if process_active 'paper_aggregate.*results/paper_tables/p9_related_work_hpo'; then
      return 0
    fi
    echo "[run] $(timestamp) rebuild_p9_related_work_hpo"
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
      --require-availability-mask-contract || true
    return 0
  fi
  if process_active 'scripts/run_exp09_p9_formal_hpo_related_work.sh'; then
    return 0
  fi
  if process_active 'hpo_equal_budget_related_work_seed_runner.yaml'; then
    return 0
  fi
  echo "[run] $(timestamp) restart_p9_formal_wrapper"
  nohup bash scripts/run_exp09_p9_formal_hpo_related_work.sh \
    >> results/background_logs/EXP09_P9_formal_wrapper.supervisor.log 2>&1 &
}

group_ready() {
  local group="$1"
  ./.venv/bin/python - <<'PY' "$group"
import json
import sys
from pathlib import Path
import pandas as pd

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
    if not run_dir.name.startswith("EXP11_P1_hpo_final_main_native_from_hpo_s"):
        return True
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

group = sys.argv[1]
base = Path("results/paper_tables") / group
path = base / "paper_aggregate_manifest.json"
main_path = base / "paper_main_comparison.csv"
seed_path = base / "paper_seed_summary.csv"
paired_path = base / "paper_paired_statistics.csv"
source_list = base / "source_run_dirs.txt"
if not (path.exists() and main_path.exists() and seed_path.exists() and paired_path.exists() and source_list.exists()):
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
formal = payload.get("formal_filter") or {}
rows = int((((payload.get("row_counts") or {}).get("paper_main_comparison")) or 0))
ok = (
    rows > 0
    and formal.get("required_protocol_id") == "core13_v2_full_reset_20260522"
    and formal.get("required_data_cutoff_date") == "2026-05-20"
    and formal.get("require_formal_manifest") is True
    and formal.get("require_availability_mask_contract") is True
)
expected_run_dirs = []
if ok and group == "main_hpo_5seed":
    main = pd.read_csv(main_path)
    expected = {f"EXP11_P1_hpo_final_main_native_from_hpo_s{seed}" for seed in (42, 123, 2024, 3407, 9999)}
    actual = set(main.get("source_run", pd.Series(dtype="object")).dropna().astype(str).unique())
    source_files = set(main.get("source_file", pd.Series(dtype="object")).dropna().astype(str).unique())
    ok = actual == expected and source_files == {"baseline_comparison.csv"}
    expected_run_dirs = [Path("results") / f"EXP11_P1_hpo_final_main_native_from_hpo_s{seed}" for seed in (42, 123, 2024, 3407, 9999)]
if ok and group == "main_hpo_plus_p9":
    main = pd.read_csv(main_path)
    source_runs = set(main.get("source_run", pd.Series(dtype="object")).dropna().astype(str).unique())
    has_p1 = {f"EXP11_P1_hpo_final_main_native_from_hpo_s{seed}" for seed in (42, 123, 2024, 3407, 9999)}.issubset(source_runs)
    has_p9 = {f"EXP09_P9_formal_hpo_related_work_s{seed}" for seed in (42, 123, 2024, 3407, 9999)}.issubset(source_runs)
    source_files = set(main.get("source_file", pd.Series(dtype="object")).dropna().astype(str).unique())
    ok = has_p1 and has_p9 and "baseline_comparison.csv" in source_files and "hpo_model_final_comparison.csv" in source_files
    expected_run_dirs = [Path("results") / f"EXP11_P1_hpo_final_main_native_from_hpo_s{seed}" for seed in (42, 123, 2024, 3407, 9999)]
    expected_run_dirs += [Path("results") / f"EXP09_P9_formal_hpo_related_work_s{seed}" for seed in (42, 123, 2024, 3407, 9999)]
if ok and group == "p9_related_work_hpo":
    expected_run_dirs = [Path("results") / f"EXP09_P9_formal_hpo_related_work_s{seed}" for seed in (42, 123, 2024, 3407, 9999)]

if ok:
    actual_run_dirs = [
        (Path(line.strip()) if Path(line.strip()).is_absolute() else (Path.cwd() / line.strip())).resolve()
        for line in source_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resolved_expected = [run_dir.resolve() for run_dir in expected_run_dirs]
    if expected_run_dirs and set(actual_run_dirs) != set(resolved_expected):
        ok = False
    aggregate_mtime = max(
        path.stat().st_mtime,
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

output_ready() {
  group_ready "main_hpo_plus_p9"
}

p7_formal_ready() {
  ./.venv/bin/python - <<'PY'
import json
from pathlib import Path

for seed in (42, 123, 2024, 3407, 9999):
    run_dir = Path("results") / f"EXP05_P7_formal_hpo_main_native_s{seed}"
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

echo "[start] $(timestamp) main_plus_p9_waiter"

acquire_lock_or_exit

while true; do
  if output_ready >/dev/null 2>&1; then
    echo "[skip] $(timestamp) main_hpo_plus_p9_already_ready"
    exit 0
  fi
  ensure_p7_wrapper
  ensure_p9_wrapper
  if ! group_ready "main_hpo_5seed" >/dev/null 2>&1 && p7_formal_ready >/dev/null 2>&1; then
    if process_active 'scripts/run_exp11_p1_hpo_final_main_native_from_hpo.sh'; then
      echo "[wait] $(timestamp) p1_from_hpo_already_running"
    else
      echo "[run] $(timestamp) build_main_hpo_5seed_via_p1"
      bash scripts/run_exp11_p1_hpo_final_main_native_from_hpo.sh || true
    fi
  fi
  if group_ready "main_hpo_5seed" >/dev/null 2>&1 && group_ready "p9_related_work_hpo" >/dev/null 2>&1; then
    break
  fi
  echo "[wait] $(timestamp) waiting_for_main_hpo_and_p9"
  sleep 300
done

echo "[run] $(timestamp) main_hpo_plus_p9"
./.venv/bin/python -m src.experiments.paper_aggregate \
  --run-dir results/EXP11_P1_hpo_final_main_native_from_hpo_s42 \
  --run-dir results/EXP11_P1_hpo_final_main_native_from_hpo_s123 \
  --run-dir results/EXP11_P1_hpo_final_main_native_from_hpo_s2024 \
  --run-dir results/EXP11_P1_hpo_final_main_native_from_hpo_s3407 \
  --run-dir results/EXP11_P1_hpo_final_main_native_from_hpo_s9999 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s42 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s123 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s2024 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s3407 \
  --run-dir results/EXP09_P9_formal_hpo_related_work_s9999 \
  --output-dir results/paper_tables/main_hpo_plus_p9 \
  --benchmark-model full_dqn_gated_multitask_cnn_ppo \
  --benchmark-model ppo_dqn_hierarchical_reimplementation \
  --benchmark-model hybrid_dqn_optimizer_sharpe_maximization \
  --paper-group-id main_hpo_plus_p9 \
  --protocol-id core13_v2_full_reset_20260522 \
  --data-cutoff-date 2026-05-20 \
  --require-formal-manifest \
  --require-availability-mask-contract
echo "[done] $(timestamp) main_plus_p9_only"
