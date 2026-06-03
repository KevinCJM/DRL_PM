#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOCK_ROOT="results/background_locks"
LOCK_DIR="${LOCK_ROOT}/exp06c_p2_p8_waiter.lock"

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
      echo "[skip] $(timestamp) exp06c_waiter_already_running pid=${owner_pid}"
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

wait_for_pattern_clear() {
  local pattern="$1"
  while process_active "$pattern"; do
    sleep 60
  done
}

group_ready() {
  local group="$1"
  shift
  ./.venv/bin/python - <<'PY' "$group" "$@"
import json
import sys
from pathlib import Path

group = sys.argv[1]
expected_run_dirs = [Path(item).resolve() for item in sys.argv[2:]]
manifest = Path("results/paper_tables") / group / "paper_aggregate_manifest.json"
main = Path("results/paper_tables") / group / "paper_main_comparison.csv"
seed = Path("results/paper_tables") / group / "paper_seed_summary.csv"
paired = Path("results/paper_tables") / group / "paper_paired_statistics.csv"
source_list = Path("results/paper_tables") / group / "source_run_dirs.txt"
if not (manifest.exists() and main.exists() and seed.exists() and paired.exists() and source_list.exists()):
    raise SystemExit(1)
payload = json.loads(manifest.read_text(encoding="utf-8"))
rows = int((((payload.get("row_counts") or {}).get("paper_main_comparison")) or 0))
if rows <= 0:
    raise SystemExit(1)

actual_run_dirs = [
    (Path(line.strip()) if Path(line.strip()).is_absolute() else (Path.cwd() / line.strip())).resolve()
    for line in source_list.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if expected_run_dirs and set(actual_run_dirs) != set(expected_run_dirs):
    raise SystemExit(1)

aggregate_mtime = manifest.stat().st_mtime
for run_dir in actual_run_dirs:
    source_manifest = run_dir / "logs" / "run_manifest.json"
    if not source_manifest.exists():
        raise SystemExit(1)
    if source_manifest.stat().st_mtime > aggregate_mtime:
        raise SystemExit(1)

raise SystemExit(0)
PY
}

main_hpo_ready() {
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

path = Path("results/paper_tables/main_hpo_5seed/paper_aggregate_manifest.json")
main_path = Path("results/paper_tables/main_hpo_5seed/paper_main_comparison.csv")
seed_path = Path("results/paper_tables/main_hpo_5seed/paper_seed_summary.csv")
paired_path = Path("results/paper_tables/main_hpo_5seed/paper_paired_statistics.csv")
source_list = Path("results/paper_tables/main_hpo_5seed/source_run_dirs.txt")
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
    expected = {f"EXP11_P1_hpo_final_main_native_from_hpo_s{seed}" for seed in (42, 123, 2024, 3407, 9999)}
    actual = set(main.get("source_run", pd.Series(dtype="object")).dropna().astype(str).unique())
    source_files = set(main.get("source_file", pd.Series(dtype="object")).dropna().astype(str).unique())
    ok = actual == expected and source_files == {"baseline_comparison.csv"}
if ok:
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

run_completed() {
  local run_name="$1"
  ./.venv/bin/python - <<'PY' "$run_name"
import json
import sys
from pathlib import Path

run_name = sys.argv[1]
manifest = Path("results") / run_name / "logs" / "run_manifest.json"
if not manifest.exists():
    raise SystemExit(1)
payload = json.loads(manifest.read_text(encoding="utf-8"))
ok = (
    payload.get("status") == "success"
    and payload.get("protocol_id") == "core13_v2_full_reset_20260522"
    and str(payload.get("data_cutoff_date")) == "2026-05-20"
)
raise SystemExit(0 if ok else 1)
PY
}

ensure_run_finished() {
  local run_name="$1"
  shift
  local active_pattern="--run-name ${run_name}"
  if run_completed "$run_name" >/dev/null 2>&1; then
    echo "[skip] $(timestamp) run_ready=${run_name}"
    return 0
  fi
  if process_active "$active_pattern"; then
    echo "[wait] $(timestamp) active_run=${run_name}"
    wait_for_pattern_clear "$active_pattern"
    if run_completed "$run_name" >/dev/null 2>&1; then
      echo "[skip] $(timestamp) run_ready_after_wait=${run_name}"
      return 0
    fi
  fi
  echo "[run] $(timestamp) run_name=${run_name}"
  ./.venv/bin/python -m src.experiments.run_experiment "$@"
}

ensure_group_aggregate() {
  local group="$1"
  shift
  local aggregate_args=("$@")
  local active_pattern="results/paper_tables/${group}"
  local run_dirs=()
  local parse_args=("$@")
  local idx=0
  while [[ $idx -lt ${#parse_args[@]} ]]; do
    case "${parse_args[$idx]}" in
      --run-dir)
        if [[ $((idx + 1)) -lt ${#parse_args[@]} ]]; then
          run_dirs+=("${parse_args[$((idx + 1))]}")
        fi
        idx=$((idx + 2))
        ;;
      *)
        idx=$((idx + 1))
        ;;
    esac
  done
  if group_ready "$group" "${run_dirs[@]}" >/dev/null 2>&1; then
    echo "[skip] $(timestamp) group_ready=${group}"
    return 0
  fi
  if process_active "$active_pattern"; then
    echo "[wait] $(timestamp) active_group_aggregate=${group}"
    wait_for_pattern_clear "$active_pattern"
    if group_ready "$group" "${run_dirs[@]}" >/dev/null 2>&1; then
      echo "[skip] $(timestamp) group_ready_after_wait=${group}"
      return 0
    fi
  fi
  echo "[run] $(timestamp) group_aggregate=${group}"
  ./.venv/bin/python -m src.experiments.paper_aggregate "${aggregate_args[@]}"
}

diagnostic_ready() {
  ./.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("results/full_reproduction/core13_v2_full_reset_20260522/diagnostic_paper_group_audit.json")
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
groups = payload.get("groups") or []
ok = bool(groups) and all(str(item.get("status") or "") == "diagnostic_complete" for item in groups)
raise SystemExit(0 if ok else 1)
PY
}

echo "[start] $(timestamp) p2_p8_diagnostics_waiter"

acquire_lock_or_exit

while true; do
  if diagnostic_ready >/dev/null 2>&1; then
    echo "[skip] $(timestamp) diagnostic_groups_already_ready"
    exit 0
  fi
  if main_hpo_ready >/dev/null 2>&1; then
    echo "[ready] $(timestamp) main_hpo_ready"
    break
  fi
  echo "[wait] $(timestamp) waiting_for_main_hpo_5seed_from_p1"
  sleep 300
done

echo "[run] $(timestamp) P2"
ensure_run_finished EXP10_P2_input_matrix_s42 --config configs/paper/input_matrix_ablation.yaml --seed 42 --run-name EXP10_P2_input_matrix_s42
ensure_run_finished EXP11_P2_pca_s42 --config configs/paper/pca_ablation.yaml --seed 42 --run-name EXP11_P2_pca_s42
ensure_group_aggregate p2_input_pca \
  --run-dir results/EXP10_P2_input_matrix_s42 \
  --run-dir results/EXP11_P2_pca_s42 \
  --output-dir results/paper_tables/p2_input_pca \
  --benchmark-model equal_weight \
  --benchmark-model cnn_ppo_native \
  --benchmark-model pgportfolio_eiie_native \
  --benchmark-model without_dqn_gate \
  --benchmark-model without_auxiliary \
  --benchmark-model no_pca \
  --paper-group-id p2_input_pca

echo "[run] $(timestamp) P3"
ensure_run_finished EXP12_P3_without_dqn_gate_s42 --config configs/paper/ablation_without_dqn_gate.yaml --seed 42 --run-name EXP12_P3_without_dqn_gate_s42
ensure_run_finished EXP13_P3_without_auxiliary_s42 --config configs/paper/ablation_without_auxiliary.yaml --seed 42 --run-name EXP13_P3_without_auxiliary_s42
ensure_run_finished EXP14_P3_mlp_encoder_s42 --config configs/paper/ablation_mlp_encoder.yaml --seed 42 --run-name EXP14_P3_mlp_encoder_s42
ensure_run_finished EXP15_P3_kernel_size_s42 --config configs/paper/kernel_size_ablation.yaml --seed 42 --run-name EXP15_P3_kernel_size_s42
ensure_group_aggregate p3_components \
  --run-dir results/EXP12_P3_without_dqn_gate_s42 \
  --run-dir results/EXP13_P3_without_auxiliary_s42 \
  --run-dir results/EXP14_P3_mlp_encoder_s42 \
  --run-dir results/EXP15_P3_kernel_size_s42 \
  --output-dir results/paper_tables/p3_components \
  --benchmark-model equal_weight \
  --benchmark-model cnn_ppo_native \
  --benchmark-model pgportfolio_eiie_native \
  --benchmark-model without_dqn_gate \
  --benchmark-model without_auxiliary \
  --benchmark-model no_pca \
  --paper-group-id p3_components

echo "[run] $(timestamp) P4"
ensure_run_finished EXP16_P4_reward_s42 --config configs/paper/reward_ablation.yaml --seed 42 --run-name EXP16_P4_reward_s42
ensure_group_aggregate p4_reward \
  --run-dir results/EXP16_P4_reward_s42 \
  --output-dir results/paper_tables/p4_reward \
  --benchmark-model equal_weight \
  --benchmark-model cnn_ppo_native \
  --benchmark-model pgportfolio_eiie_native \
  --benchmark-model without_dqn_gate \
  --benchmark-model without_auxiliary \
  --benchmark-model no_pca \
  --paper-group-id p4_reward

echo "[run] $(timestamp) P5"
ensure_run_finished EXP17_P5_cost_s42 --config configs/paper/transaction_cost_sensitivity.yaml --seed 42 --run-name EXP17_P5_cost_s42
ensure_run_finished EXP18_P5_rebalance_s42 --config configs/paper/rebalance_frequency_analysis.yaml --seed 42 --run-name EXP18_P5_rebalance_s42
ensure_group_aggregate p5_cost_rebalance \
  --run-dir results/EXP17_P5_cost_s42 \
  --run-dir results/EXP18_P5_rebalance_s42 \
  --output-dir results/paper_tables/p5_cost_rebalance \
  --benchmark-model equal_weight \
  --benchmark-model cnn_ppo_native \
  --benchmark-model pgportfolio_eiie_native \
  --benchmark-model without_dqn_gate \
  --benchmark-model without_auxiliary \
  --benchmark-model no_pca \
  --paper-group-id p5_cost_rebalance

echo "[run] $(timestamp) P6"
ensure_run_finished EXP19_P6_seed_stability --config configs/paper/seed_stability.yaml --run-name EXP19_P6_seed_stability
ensure_run_finished EXP20_P6_market_regime --config configs/paper/market_regime.yaml --run-name EXP20_P6_market_regime
ensure_run_finished EXP21_P6_asset_universe --config configs/paper/asset_universe_sensitivity.yaml --run-name EXP21_P6_asset_universe
ensure_run_finished EXP22_P6_walk_forward --config configs/paper/walk_forward.yaml --run-name EXP22_P6_walk_forward
ensure_group_aggregate p6_robustness \
  --run-dir results/EXP19_P6_seed_stability \
  --run-dir results/EXP20_P6_market_regime \
  --run-dir results/EXP21_P6_asset_universe \
  --run-dir results/EXP22_P6_walk_forward \
  --output-dir results/paper_tables/p6_robustness \
  --benchmark-model equal_weight \
  --benchmark-model cnn_ppo_native \
  --benchmark-model pgportfolio_eiie_native \
  --benchmark-model without_dqn_gate \
  --benchmark-model without_auxiliary \
  --benchmark-model no_pca \
  --paper-group-id p6_robustness

echo "[run] $(timestamp) P8"
ensure_run_finished EXP23_P8_preference_s42 --config configs/paper/preference_conditioned_analysis.yaml --seed 42 --run-name EXP23_P8_preference_s42
ensure_run_finished EXP24_P8_uncertainty_s42 --config configs/paper/uncertainty_analysis.yaml --seed 42 --run-name EXP24_P8_uncertainty_s42
ensure_run_finished EXP25_P8_distributional_cvar_s42 --config configs/paper/distributional_cvar_analysis.yaml --seed 42 --run-name EXP25_P8_distributional_cvar_s42
ensure_run_finished EXP26_P8_partial_rebalance_s42 --config configs/paper/partial_rebalance_analysis.yaml --seed 42 --run-name EXP26_P8_partial_rebalance_s42
ensure_group_aggregate p8_modules \
  --run-dir results/EXP23_P8_preference_s42 \
  --run-dir results/EXP24_P8_uncertainty_s42 \
  --run-dir results/EXP25_P8_distributional_cvar_s42 \
  --run-dir results/EXP26_P8_partial_rebalance_s42 \
  --output-dir results/paper_tables/p8_modules \
  --benchmark-model equal_weight \
  --benchmark-model cnn_ppo_native \
  --benchmark-model pgportfolio_eiie_native \
  --benchmark-model without_dqn_gate \
  --benchmark-model without_auxiliary \
  --benchmark-model no_pca \
  --paper-group-id p8_modules

echo "[run] $(timestamp) diagnostic_group_audit"
if diagnostic_ready >/dev/null 2>&1; then
  echo "[skip] $(timestamp) diagnostic_groups_already_ready"
else
  ./.venv/bin/python scripts/audit_diagnostic_paper_groups.py
fi
echo "[done] $(timestamp) p2_p8_diagnostics"
