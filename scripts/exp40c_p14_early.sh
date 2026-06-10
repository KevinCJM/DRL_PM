#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOCK_ROOT="results/background_locks"
LOCK_DIR="${LOCK_ROOT}/exp40c_p14_waiter.lock"

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
      echo "[skip] $(timestamp) exp40c_waiter_already_running pid=${owner_pid}"
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
raise SystemExit(0)
PY
}

promoted_benchmark() {
  ./.venv/bin/python - <<'PY'
import csv
from pathlib import Path

path = Path("results/paper_tables/p12_p13_promotion_gate/promotion_gate_report.csv")
rows = []
if path.exists():
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
passed = [
    row for row in rows
    if str(row.get("promotion_gate_passed", "")).strip().lower() == "true"
]
if not passed:
    print("")
    raise SystemExit(0)

def score(row: dict[str, str]) -> float:
    try:
        return float(row.get("validation_return_cost_risk_utility") or "-inf")
    except ValueError:
        return float("-inf")

best = max(passed, key=score)
print(str(best.get("model_name") or "").strip())
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

p12_formal_ready() {
  aggregate_group_ready \
    "p12_cage_eiie_formal" \
    results/EXP30_P12_formal_cage_eiie_s42 \
    results/EXP30_P12_formal_cage_eiie_s123 \
    results/EXP30_P12_formal_cage_eiie_s2024 \
    results/EXP30_P12_formal_cage_eiie_s3407 \
    results/EXP30_P12_formal_cage_eiie_s9999
}

is_ready() {
  ./.venv/bin/python - <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path.cwd()
seeds = (42, 123, 2024, 3407, 9999)

gate_path = root / "results/paper_tables/p12_p13_promotion_gate/promotion_gate_report.csv"
if not gate_path.exists():
    raise SystemExit(1)
with gate_path.open("r", encoding="utf-8", newline="") as fh:
    gate_rows = list(csv.DictReader(fh))
promoted = [
    row for row in gate_rows
    if str(row.get("promotion_gate_passed") or "").strip().lower() == "true"
]
if not promoted:
    print("skip:no_p12_or_p13_promotion")
    raise SystemExit(0)

p12_manifest = root / "results/paper_tables/p12_cage_eiie_formal/paper_aggregate_manifest.json"
if not p12_manifest.exists():
    raise SystemExit(1)
payload = json.loads(p12_manifest.read_text(encoding="utf-8"))
rows = (((payload.get("row_counts") or {}).get("paper_main_comparison")) or 0)
if int(rows) <= 0:
    raise SystemExit(1)

prefixes = [
    "EXP05_P7_formal_hpo_main_native",
    "EXP09_P9_formal_hpo_related_work",
    "EXP30_P12_formal_cage_eiie",
]
if any(str(row.get("phase") or "").strip() == "P13" for row in promoted):
    prefixes.append("EXP33_P13_formal_gt_rcpo_lite")

for prefix in prefixes:
    for seed in seeds:
        manifest_path = root / "results" / f"{prefix}_s{seed}" / "logs" / "run_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(1)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "success":
            raise SystemExit(2)
        if manifest.get("diagnostic_status") != "formal":
            raise SystemExit(2)
        if manifest.get("rankable_in_unified_table") is not True:
            raise SystemExit(2)
print("ready")
PY
}

echo "[start] $(timestamp) p14_early_waiter"

acquire_lock_or_exit

promoted_model="$(promoted_benchmark)"
status_file="results/background_logs/EXP40C_p14_early.status"
mkdir -p "$(dirname "$status_file")"

while true; do
  if is_ready >"$status_file" 2>&1; then
    status="$(cat "$status_file" || true)"
    if [[ "$status" == "skip:no_p12_or_p13_promotion" ]]; then
      echo "[skip] $(timestamp) no_p12_or_p13_promotion"
      exit 0
    fi
    if p12_formal_ready >/dev/null 2>&1; then
      break
    fi
  fi
  echo "[wait] $(timestamp) waiting_for_p12_p7_p9_for_p14"
  sleep 300
done

benchmark="$(promoted_benchmark)"
if [[ -z "$benchmark" ]]; then
  echo "[skip] $(timestamp) no_promoted_benchmark"
  exit 0
fi

run_dirs=()
for seed in 42 123 2024 3407 9999; do
  run_dirs+=("results/EXP05_P7_formal_hpo_main_native_s${seed}")
done
for seed in 42 123 2024 3407 9999; do
  run_dirs+=("results/EXP09_P9_formal_hpo_related_work_s${seed}")
done
for seed in 42 123 2024 3407 9999; do
  run_dirs+=("results/EXP30_P12_formal_cage_eiie_s${seed}")
done
if [[ "$(p13_promoted)" == "1" ]]; then
  for seed in 42 123 2024 3407 9999; do
    run_dirs+=("results/EXP33_P13_formal_gt_rcpo_lite_s${seed}")
  done
fi

if aggregate_group_ready "p14_new_model_final" "${run_dirs[@]}" >/dev/null 2>&1; then
  echo "[skip] $(timestamp) p14_new_model_final_already_ready"
  exit 0
fi

cmd=(
  ./.venv/bin/python -m src.experiments.paper_aggregate
  --output-dir results/paper_tables/p14_new_model_final
  --benchmark-model "$benchmark"
  --paper-group-id p14_new_model_final
  --protocol-id core13_v2_full_reset_20260522
  --data-cutoff-date 2026-05-20
  --require-formal-manifest
  --require-availability-mask-contract
)
for run_dir in "${run_dirs[@]}"; do
  cmd+=(--run-dir "$run_dir")
done

echo "[run] $(timestamp) benchmark=${benchmark}"
"${cmd[@]}"
echo "[done] $(timestamp) p14_new_model_final"
