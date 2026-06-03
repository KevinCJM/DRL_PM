#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOCK_ROOT="results/background_locks"
LOCK_DIR="${LOCK_ROOT}/exp40e_bundle_waiter.lock"

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
      echo "[skip] $(timestamp) exp40e_waiter_already_running pid=${owner_pid}"
      exit 0
    fi
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
  echo "$$" > "${LOCK_DIR}/pid"
  trap release_lock EXIT
}

aggregate_group_ready() {
  local group="$1"
  shift
  ./.venv/bin/python - <<'PY' "$group" "$@"
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
expected_run_dirs = [Path(item).resolve() for item in sys.argv[2:]]
base = Path("results/paper_tables") / group
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
actual_run_dirs = [
    (Path(line.strip()) if Path(line.strip()).is_absolute() else (Path.cwd() / line.strip())).resolve()
    for line in source_list.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if expected_run_dirs and set(actual_run_dirs) != set(expected_run_dirs):
    raise SystemExit(1)
main_frame = pd.read_csv(main)
if group == "main_hpo_plus_p9":
    source_runs = set(main_frame.get("source_run", pd.Series(dtype="object")).dropna().astype(str).unique())
    p1_runs = {f"EXP11_P1_hpo_final_main_native_from_hpo_s{seed}" for seed in (42, 123, 2024, 3407, 9999)}
    p9_runs = {f"EXP09_P9_formal_hpo_related_work_s{seed}" for seed in (42, 123, 2024, 3407, 9999)}
    source_files = set(main_frame.get("source_file", pd.Series(dtype="object")).dropna().astype(str).unique())
    if not (
        p1_runs.issubset(source_runs)
        and p9_runs.issubset(source_runs)
        and "baseline_comparison.csv" in source_files
        and "hpo_model_final_comparison.csv" in source_files
    ):
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
    if not source_manifest.exists():
        raise SystemExit(1)
    if source_manifest.stat().st_mtime > aggregate_mtime:
        raise SystemExit(1)
    if not p1_source_fresh(run_dir):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

p12_or_p13_promoted() {
  ./.venv/bin/python - <<'PY'
import csv
from pathlib import Path

path = Path("results/paper_tables/p12_p13_promotion_gate/promotion_gate_report.csv")
if not path.exists():
    print("0")
    raise SystemExit(0)
with path.open("r", encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))
flag = any(
    str(row.get("promotion_gate_passed") or "").strip().lower() == "true"
    for row in rows
)
print("1" if flag else "0")
PY
}

p13_promoted() {
  ./.venv/bin/python - <<'PY'
import csv
from pathlib import Path

path = Path("results/paper_tables/p12_p13_promotion_gate/promotion_gate_report.csv")
if not path.exists():
    print("0")
    raise SystemExit(0)
with path.open("r", encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))
flag = any(
    str(row.get("phase") or "").strip() == "P13"
    and str(row.get("promotion_gate_passed") or "").strip().lower() == "true"
    for row in rows
)
print("1" if flag else "0")
PY
}

p16_promoted() {
  ./.venv/bin/python - <<'PY'
import csv
from pathlib import Path

path = Path("results/paper_tables/p16_promotion_gate/promotion_gate_report.csv")
if not path.exists():
    print("0")
    raise SystemExit(0)
with path.open("r", encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))
flag = any(
    str(row.get("promotion_gate_passed") or "").strip().lower() == "true"
    for row in rows
)
print("1" if flag else "0")
PY
}

diagnostic_audit_ready() {
  ./.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
audit_path = root / "results/full_reproduction/core13_v2_full_reset_20260522/diagnostic_paper_group_audit.json"
csv_path = root / "results/full_reproduction/core13_v2_full_reset_20260522/diagnostic_paper_group_audit.csv"
expected_groups = (
    "p2_input_pca",
    "p3_components",
    "p4_reward",
    "p5_cost_rebalance",
    "p6_robustness",
    "p8_modules",
)
if not (audit_path.exists() and csv_path.exists()):
    raise SystemExit(1)
audit_mtime = min(audit_path.stat().st_mtime, csv_path.stat().st_mtime)
audit = json.loads(audit_path.read_text(encoding="utf-8"))
groups = audit.get("groups") or []
if len(groups) != len(expected_groups):
    raise SystemExit(2)
seen = {str(item.get("group_id") or "") for item in groups}
if seen != set(expected_groups):
    raise SystemExit(2)
if any(str(item.get("status") or "") != "diagnostic_complete" for item in groups):
    raise SystemExit(2)
for group in expected_groups:
    group_dir = root / "results/paper_tables" / group
    required = [
        group_dir / "paper_main_comparison.csv",
        group_dir / "paper_seed_summary.csv",
        group_dir / "paper_aggregate_manifest.json",
        group_dir / "source_run_dirs.txt",
        group_dir / "diagnostic_status.json",
    ]
    paired = group_dir / "paper_paired_statistics.csv"
    not_applicable = group_dir / "not_applicable_reason.txt"
    if paired.exists():
        required.append(paired)
    elif not not_applicable.exists():
        raise SystemExit(2)
    else:
        required.append(not_applicable)
    if not all(path.exists() for path in required):
        raise SystemExit(2)
    if max(path.stat().st_mtime for path in required) > audit_mtime:
        raise SystemExit(2)
PY
}

is_ready() {
  if ! aggregate_group_ready \
    "main_hpo_plus_p9" \
    results/EXP11_P1_hpo_final_main_native_from_hpo_s42 \
    results/EXP11_P1_hpo_final_main_native_from_hpo_s123 \
    results/EXP11_P1_hpo_final_main_native_from_hpo_s2024 \
    results/EXP11_P1_hpo_final_main_native_from_hpo_s3407 \
    results/EXP11_P1_hpo_final_main_native_from_hpo_s9999 \
    results/EXP09_P9_formal_hpo_related_work_s42 \
    results/EXP09_P9_formal_hpo_related_work_s123 \
    results/EXP09_P9_formal_hpo_related_work_s2024 \
    results/EXP09_P9_formal_hpo_related_work_s3407 \
    results/EXP09_P9_formal_hpo_related_work_s9999
  then
    return 1
  fi

  if ! diagnostic_audit_ready; then
    return 1
  fi

  if [[ "$(p12_or_p13_promoted)" == "1" ]]; then
    local p14_run_dirs=(
      results/EXP05_P7_formal_hpo_main_native_s42
      results/EXP05_P7_formal_hpo_main_native_s123
      results/EXP05_P7_formal_hpo_main_native_s2024
      results/EXP05_P7_formal_hpo_main_native_s3407
      results/EXP05_P7_formal_hpo_main_native_s9999
      results/EXP09_P9_formal_hpo_related_work_s42
      results/EXP09_P9_formal_hpo_related_work_s123
      results/EXP09_P9_formal_hpo_related_work_s2024
      results/EXP09_P9_formal_hpo_related_work_s3407
      results/EXP09_P9_formal_hpo_related_work_s9999
      results/EXP30_P12_formal_cage_eiie_s42
      results/EXP30_P12_formal_cage_eiie_s123
      results/EXP30_P12_formal_cage_eiie_s2024
      results/EXP30_P12_formal_cage_eiie_s3407
      results/EXP30_P12_formal_cage_eiie_s9999
    )
    if [[ "$(p13_promoted)" == "1" ]]; then
      p14_run_dirs+=(
        results/EXP33_P13_formal_gt_rcpo_lite_s42
        results/EXP33_P13_formal_gt_rcpo_lite_s123
        results/EXP33_P13_formal_gt_rcpo_lite_s2024
        results/EXP33_P13_formal_gt_rcpo_lite_s3407
        results/EXP33_P13_formal_gt_rcpo_lite_s9999
      )
    fi
    if ! aggregate_group_ready "p14_new_model_final" "${p14_run_dirs[@]}"; then
      return 1
    fi
  fi

  if [[ "$(p16_promoted)" == "1" ]]; then
    local p16_run_dirs=(
      results/EXP35_P16_formal_ra_gt_rcpo_s42
      results/EXP35_P16_formal_ra_gt_rcpo_s123
      results/EXP35_P16_formal_ra_gt_rcpo_s2024
      results/EXP35_P16_formal_ra_gt_rcpo_s3407
      results/EXP35_P16_formal_ra_gt_rcpo_s9999
      results/EXP05_P7_formal_hpo_main_native_s42
      results/EXP05_P7_formal_hpo_main_native_s123
      results/EXP05_P7_formal_hpo_main_native_s2024
      results/EXP05_P7_formal_hpo_main_native_s3407
      results/EXP05_P7_formal_hpo_main_native_s9999
      results/EXP09_P9_formal_hpo_related_work_s42
      results/EXP09_P9_formal_hpo_related_work_s123
      results/EXP09_P9_formal_hpo_related_work_s2024
      results/EXP09_P9_formal_hpo_related_work_s3407
      results/EXP09_P9_formal_hpo_related_work_s9999
      results/EXP36_P1_fixed_deterministic_formal_export
    )
    if [[ "$(p12_or_p13_promoted)" == "1" ]]; then
      p16_run_dirs+=(
        results/EXP30_P12_formal_cage_eiie_s42
        results/EXP30_P12_formal_cage_eiie_s123
        results/EXP30_P12_formal_cage_eiie_s2024
        results/EXP30_P12_formal_cage_eiie_s3407
        results/EXP30_P12_formal_cage_eiie_s9999
      )
    fi
    if [[ "$(p13_promoted)" == "1" ]]; then
      p16_run_dirs+=(
        results/EXP33_P13_formal_gt_rcpo_lite_s42
        results/EXP33_P13_formal_gt_rcpo_lite_s123
        results/EXP33_P13_formal_gt_rcpo_lite_s2024
        results/EXP33_P13_formal_gt_rcpo_lite_s3407
        results/EXP33_P13_formal_gt_rcpo_lite_s9999
      )
    fi
    if ! aggregate_group_ready "p16_ra_gt_rcpo_final" "${p16_run_dirs[@]}"; then
      return 1
    fi
  fi

  return 0
}

echo "[start] $(timestamp) bundle_readiness_waiter"

acquire_lock_or_exit

while true; do
  if is_ready; then
    break
  fi
  echo "[wait] $(timestamp) waiting_for_final_artifact_prereqs"
  sleep 300
done

echo "[run] $(timestamp) run_ledger"
./.venv/bin/python -m src.experiments.run_ledger \
  --results-root results \
  --output-dir results/full_reproduction/core13_v2_full_reset_20260522 \
  --protocol-id core13_v2_full_reset_20260522

echo "[run] $(timestamp) diagnostic_paper_group_audit_refresh"
./.venv/bin/python scripts/audit_diagnostic_paper_groups.py

echo "[run] $(timestamp) build_artifact_bundle"
./.venv/bin/python paper/scripts/build_artifact_bundle.py

echo "[run] $(timestamp) generate_paper_tables"
./.venv/bin/python paper/scripts/generate_paper_tables.py --bundle-only

echo "[run] $(timestamp) generate_paper_figures"
./.venv/bin/python paper/scripts/generate_paper_figures.py --bundle-only

echo "[run] $(timestamp) formal_readiness"
./.venv/bin/python scripts/audit_core13_formal_readiness.py --fail-on-no-go

echo "[done] $(timestamp) bundle_readiness"
