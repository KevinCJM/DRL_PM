#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOCK_ROOT="results/background_locks"
LOCK_DIR="${LOCK_ROOT}/exp37b_resume_waiter.lock"

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
      echo "[skip] $(timestamp) exp37b_waiter_already_running pid=${owner_pid}"
      exit 0
    fi
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
  echo "$$" > "${LOCK_DIR}/pid"
  trap release_lock EXIT
}

start_exp37() {
  local log_path="results/background_logs/EXP37_P16_ra_gt_rcpo_ablation_s42.log"
  echo "[run] $(timestamp) starting_exp37"
  mkdir -p "$(dirname "$log_path")"
  if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t drl_exp37 >/dev/null 2>&1; then
      echo "[skip] $(timestamp) exp37_tmux_already_running"
      return 0
    fi
    tmux new-session -d -s drl_exp37 \
      "cd \"$ROOT\" && ./.venv/bin/python -m src.experiments.run_experiment --config configs/paper/p16_ra_gt_rcpo_ablation.yaml --seed 42 --run-name EXP37_P16_ra_gt_rcpo_ablation_s42 >\"$log_path\" 2>&1"
    return 0
  fi
  nohup ./.venv/bin/python -m src.experiments.run_experiment \
    --config configs/paper/p16_ra_gt_rcpo_ablation.yaml \
    --seed 42 \
    --run-name EXP37_P16_ra_gt_rcpo_ablation_s42 \
    >"$log_path" 2>&1 &
}

group_ready() {
  ./.venv/bin/python - <<'PY'
import json
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

path = Path("results/paper_tables/main_hpo_plus_p9/paper_aggregate_manifest.json")
main_path = Path("results/paper_tables/main_hpo_plus_p9/paper_main_comparison.csv")
seed_path = Path("results/paper_tables/main_hpo_plus_p9/paper_seed_summary.csv")
paired_path = Path("results/paper_tables/main_hpo_plus_p9/paper_paired_statistics.csv")
source_list = Path("results/paper_tables/main_hpo_plus_p9/source_run_dirs.txt")
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
if ok:
    main = pd.read_csv(main_path)
    source_runs = set(main.get("source_run", pd.Series(dtype="object")).dropna().astype(str).unique())
    has_p1 = {f"EXP11_P1_hpo_final_main_native_from_hpo_s{seed}" for seed in (42, 123, 2024, 3407, 9999)}.issubset(source_runs)
    has_p9 = {f"EXP09_P9_formal_hpo_related_work_s{seed}" for seed in (42, 123, 2024, 3407, 9999)}.issubset(source_runs)
    source_files = set(main.get("source_file", pd.Series(dtype="object")).dropna().astype(str).unique())
    ok = has_p1 and has_p9 and "baseline_comparison.csv" in source_files and "hpo_model_final_comparison.csv" in source_files
if ok:
    actual_run_dirs = [
        (Path(line.strip()) if Path(line.strip()).is_absolute() else (Path.cwd() / line.strip())).resolve()
        for line in source_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_run_dirs = [
        (Path("results") / f"EXP11_P1_hpo_final_main_native_from_hpo_s{seed}").resolve()
        for seed in (42, 123, 2024, 3407, 9999)
    ] + [
        (Path("results") / f"EXP09_P9_formal_hpo_related_work_s{seed}").resolve()
        for seed in (42, 123, 2024, 3407, 9999)
    ]
    if set(actual_run_dirs) != set(expected_run_dirs):
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

echo "[start] $(timestamp) exp37_resume_waiter"

acquire_lock_or_exit

while true; do
  if group_ready >/dev/null 2>&1; then
    break
  fi
  echo "[wait] $(timestamp) waiting_for_main_hpo_plus_p9_before_resuming_exp37"
  sleep 300
done

echo "[run] $(timestamp) resuming_exp37"
exp37_pids="$(pgrep -f 'EXP37_P16_ra_gt_rcpo_ablation_s42' || true)"
if [[ -z "$exp37_pids" ]]; then
  start_exp37
  echo "[done] $(timestamp) exp37_started"
  exit 0
fi
while IFS= read -r pid; do
  [[ -n "$pid" ]] || continue
  kill -CONT "$pid"
done <<<"$exp37_pids"
echo "[done] $(timestamp) exp37_resumed"
