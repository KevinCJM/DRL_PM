#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/background_logs

LOCK_ROOT="results/background_locks"
LOCK_DIR="${LOCK_ROOT}/exp40d_p16_final_waiter.lock"

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
      echo "[skip] $(timestamp) exp40d_waiter_already_running pid=${owner_pid}"
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

p16_formal_ready() {
  ./.venv/bin/python - <<'PY'
import json
from pathlib import Path

def manifest_ok(path: Path) -> bool:
    if not path.exists():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return (
        manifest.get("status") == "success"
        and manifest.get("diagnostic_status") == "formal"
        and manifest.get("rankable_in_unified_table") is True
    )

root = Path.cwd() / "results"
for seed in (42, 123, 2024, 3407, 9999):
    if not manifest_ok(root / f"EXP35_P16_formal_ra_gt_rcpo_s{seed}" / "logs" / "run_manifest.json"):
        raise SystemExit(1)
if not manifest_ok(root / "EXP36_P1_fixed_deterministic_formal_export/logs/run_manifest.json"):
    raise SystemExit(1)
print("ready")
PY
}

p16_promotion_status() {
  ./.venv/bin/python - <<'PY'
import csv
from pathlib import Path

path = Path("results/paper_tables/p16_promotion_gate/promotion_gate_report.csv")
if not path.exists():
    print("pending")
    raise SystemExit(0)
with path.open("r", encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    print("pending")
    raise SystemExit(0)
passed = any(
    str(row.get("promotion_gate_passed") or "").strip().lower() == "true"
    for row in rows
)
print("passed" if passed else "failed")
PY
}

ensure_p16_wrapper() {
  if p16_formal_ready >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$(p16_promotion_status)" == "failed" ]]; then
    echo "[skip] $(timestamp) p16_promotion_gate_failed_no_formal"
    return 0
  fi
  if process_active 'scripts/run_exp35b_p16_gate_and_formal.sh'; then
    return 0
  fi
  if process_active 'scripts/recover_exp35_p16_formal_seed.py --config configs/paper/p16_ra_gt_rcpo_formal_seed_runner.yaml'; then
    return 0
  fi
  if process_active 'p16_ra_gt_rcpo_formal_seed_runner.yaml'; then
    return 0
  fi
  echo "[run] $(timestamp) restart_p16_formal_wrapper"
  nohup bash scripts/run_exp35b_p16_gate_and_formal.sh \
    >> results/background_logs/EXP35_P16_formal_wrapper.supervisor.log 2>&1 &
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

is_ready() {
  ./.venv/bin/python - <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path.cwd()
seeds = (42, 123, 2024, 3407, 9999)

def manifest_ok(path: Path) -> bool:
    if not path.exists():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return (
        manifest.get("status") == "success"
        and manifest.get("diagnostic_status") == "formal"
        and manifest.get("rankable_in_unified_table") is True
    )

prefixes = [
    "EXP05_P7_formal_hpo_main_native",
    "EXP09_P9_formal_hpo_related_work",
    "EXP35_P16_formal_ra_gt_rcpo",
]
gate_path = root / "results/paper_tables/p12_p13_promotion_gate/promotion_gate_report.csv"
p13_flag = False
promoted_flag = False
if gate_path.exists():
    with gate_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    promoted_flag = any(str(row.get("promotion_gate_passed") or "").strip().lower() == "true" for row in rows)
    p13_flag = any(
        str(row.get("phase") or "").strip() == "P13"
        and str(row.get("promotion_gate_passed") or "").strip().lower() == "true"
        for row in rows
    )
if p13_flag:
    prefixes.append("EXP33_P13_formal_gt_rcpo_lite")

for prefix in prefixes:
    for seed in seeds:
        path = root / "results" / f"{prefix}_s{seed}" / "logs" / "run_manifest.json"
        if not manifest_ok(path):
            raise SystemExit(1)

exp36 = root / "results/EXP36_P1_fixed_deterministic_formal_export/logs/run_manifest.json"
if not manifest_ok(exp36):
    raise SystemExit(1)

if promoted_flag:
    p14_base = root / "results/paper_tables/p14_new_model_final"
    p14_manifest = p14_base / "paper_aggregate_manifest.json"
    p14_main = p14_base / "paper_main_comparison.csv"
    p14_seed = p14_base / "paper_seed_summary.csv"
    p14_paired = p14_base / "paper_paired_statistics.csv"
    p14_source_list = p14_base / "source_run_dirs.txt"
    if not (p14_manifest.exists() and p14_main.exists() and p14_seed.exists() and p14_paired.exists() and p14_source_list.exists()):
        raise SystemExit(1)
    payload = json.loads(p14_manifest.read_text(encoding="utf-8"))
    formal = payload.get("formal_filter") or {}
    row_count = int((((payload.get("row_counts") or {}).get("paper_main_comparison")) or 0))
    if not (
        row_count > 0
        and formal.get("required_protocol_id") == "core13_v2_full_reset_20260522"
        and formal.get("required_data_cutoff_date") == "2026-05-20"
        and formal.get("require_formal_manifest") is True
        and formal.get("require_availability_mask_contract") is True
    ):
        raise SystemExit(2)
    expected = [
        (root / "results" / f"EXP05_P7_formal_hpo_main_native_s{seed}").resolve()
        for seed in seeds
    ] + [
        (root / "results" / f"EXP09_P9_formal_hpo_related_work_s{seed}").resolve()
        for seed in seeds
    ] + [
        (root / "results" / f"EXP30_P12_formal_cage_eiie_s{seed}").resolve()
        for seed in seeds
    ]
    if p13_flag:
        expected.extend(
            (root / "results" / f"EXP33_P13_formal_gt_rcpo_lite_s{seed}").resolve()
            for seed in seeds
        )
    actual = [
        (Path(line.strip()) if Path(line.strip()).is_absolute() else (root / line.strip())).resolve()
        for line in p14_source_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if set(actual) != set(expected):
        raise SystemExit(2)
    aggregate_mtime = max(
        p14_manifest.stat().st_mtime,
        p14_main.stat().st_mtime,
        p14_seed.stat().st_mtime,
        p14_paired.stat().st_mtime,
        p14_source_list.stat().st_mtime,
    )
    for run_dir in actual:
        source_manifest = run_dir / "logs" / "run_manifest.json"
        if not source_manifest.exists():
            raise SystemExit(2)
        if source_manifest.stat().st_mtime > aggregate_mtime:
            raise SystemExit(2)

print("ready")
PY
}

echo "[start] $(timestamp) p16_final_early_waiter"

acquire_lock_or_exit

while true; do
  p16_gate_status="$(p16_promotion_status)"
  if [[ "$p16_gate_status" == "failed" ]]; then
    echo "[skip] $(timestamp) p16_promotion_gate_failed_no_final"
    exit 0
  fi
  ensure_p16_wrapper
  if is_ready; then
    break
  fi
  echo "[wait] $(timestamp) waiting_for_p16_p7_p9_for_p16_final"
  sleep 300
done

run_dirs=()
for seed in 42 123 2024 3407 9999; do
  run_dirs+=("results/EXP35_P16_formal_ra_gt_rcpo_s${seed}")
done
for seed in 42 123 2024 3407 9999; do
  run_dirs+=("results/EXP05_P7_formal_hpo_main_native_s${seed}")
done
for seed in 42 123 2024 3407 9999; do
  run_dirs+=("results/EXP09_P9_formal_hpo_related_work_s${seed}")
done
run_dirs+=("results/EXP36_P1_fixed_deterministic_formal_export")
if [[ "$(p12_or_p13_promoted)" == "1" ]]; then
  for seed in 42 123 2024 3407 9999; do
    run_dirs+=("results/EXP30_P12_formal_cage_eiie_s${seed}")
  done
fi
if [[ "$(p13_promoted)" == "1" ]]; then
  for seed in 42 123 2024 3407 9999; do
    run_dirs+=("results/EXP33_P13_formal_gt_rcpo_lite_s${seed}")
  done
fi

if aggregate_group_ready "p16_ra_gt_rcpo_final" "${run_dirs[@]}" >/dev/null 2>&1; then
  echo "[skip] $(timestamp) p16_ra_gt_rcpo_final_already_ready"
  exit 0
fi

cmd=(
  ./.venv/bin/python -m src.experiments.paper_aggregate
  --output-dir results/paper_tables/p16_ra_gt_rcpo_final
  --benchmark-model risk_aware_graph_transformer_constrained_actor_critic
  --benchmark-model full_dqn_gated_multitask_cnn_ppo
  --benchmark-model ppo_dqn_hierarchical_reimplementation
  --benchmark-model risk_parity
  --paper-group-id p16_ra_gt_rcpo_final
  --protocol-id core13_v2_full_reset_20260522
  --data-cutoff-date 2026-05-20
  --require-formal-manifest
  --require-availability-mask-contract
)

if [[ "$(p12_or_p13_promoted)" == "1" ]]; then
  cmd+=(--benchmark-model cage_eiie_multilevel_gate)
fi
for run_dir in "${run_dirs[@]}"; do
  cmd+=(--run-dir "$run_dir")
done

echo "[run] $(timestamp) p16_ra_gt_rcpo_final"
"${cmd[@]}"
echo "[done] $(timestamp) p16_ra_gt_rcpo_final"
