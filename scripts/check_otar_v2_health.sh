#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${1:-$PROJECT_ROOT/outputs/otar_v2_full/OTAR_v2_full}"
MANIFEST="$OUTPUT_DIR/logs/otar_v2_full_manifest.json"
PID_FILE="$OUTPUT_DIR/logs/otar_v2_full.pid"
STALE_THRESHOLD_SEC="${2:-7200}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_manifest() {
    if [[ ! -f "$MANIFEST" ]]; then
        log "ERROR: Manifest not found: $MANIFEST"
        return 1
    fi

    local status
    status=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['status'])" 2>/dev/null || echo "unknown")
    log "Manifest status: $status"

    if [[ "$status" == "running" ]]; then
        if [[ -f "$PID_FILE" ]]; then
            local pid
            pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                log "Process PID $pid is alive."
            else
                log "WARNING: Process PID $pid is NOT alive. Manifest is stale."
                log "  Run with --resume-completed-children to auto-repair."
            fi
        else
            log "WARNING: No PID file found. Manifest may be stale."
        fi
    fi
}

check_child_runs() {
    local total_planned completed incomplete
    total_planned=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/logs/formal_matrix_plan.json'))['selected_run_count'])" 2>/dev/null || echo "?")

    completed=0
    incomplete=0
    for dir in "$OUTPUT_DIR"/[0-9]*; do
        [[ -d "$dir" ]] || continue
        local result_path="$dir/logs/experiment_result.json"
        local manifest_path="$dir/logs/run_manifest.json"
        if [[ -f "$result_path" && -f "$manifest_path" ]]; then
            local r_status m_status
            r_status=$(python3 -c "import json; print(json.load(open('$result_path'))['status'])" 2>/dev/null || echo "unknown")
            m_status=$(python3 -c "import json; print(json.load(open('$manifest_path'))['status'])" 2>/dev/null || echo "unknown")
            if [[ "$r_status" == "completed" && "$m_status" == "success" ]]; then
                ((completed++))
            else
                ((incomplete++))
                log "  Incomplete: $(basename "$dir") (result=$r_status, manifest=$m_status)"
            fi
        else
            ((incomplete++))
            log "  Incomplete: $(basename "$dir") (missing artifacts)"
        fi
    done

    log "Child runs: $completed completed, $incomplete incomplete, $total_planned planned"
}

check_latest_trial() {
    local latest_dir
    latest_dir=$(ls -dt "$OUTPUT_DIR"/[0-9]* 2>/dev/null | head -1)
    if [[ -z "$latest_dir" ]]; then
        log "No child run directories found."
        return
    fi

    local hpo_csv="$latest_dir/logs/hpo_trials.csv"
    if [[ -f "$hpo_csv" ]]; then
        local trial_info
        trial_info=$(python3 -c "
import csv, sys
with open('$hpo_csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)
    if rows:
        last = rows[-1]
        state_idx = header.index('state') if 'state' in header else 6
        end_idx = header.index('train_end') if 'train_end' in header else 18
        print(f'trials={len(rows)}')
        print(f'state={last[state_idx]}')
        print(f'end={last[end_idx]}')
    else:
        print('trials=0')
" 2>/dev/null || echo "parse_error")
        local trial_count latest_state latest_trial_time
        trial_count=$(echo "$trial_info" | grep '^trials=' | cut -d= -f2 || echo "?")
        latest_state=$(echo "$trial_info" | grep '^state=' | cut -d= -f2 || echo "?")
        latest_trial_time=$(echo "$trial_info" | grep '^end=' | cut -d= -f2 || echo "?")
        log "Latest child run: $(basename "$latest_dir")"
        log "  HPO trials completed: $trial_count"
        log "  Latest trial state: $latest_state"
        log "  Latest trial end: $latest_trial_time"
    fi
}

log "=== OTAR V2 Full Experiment Health Check ==="
check_manifest
check_child_runs
check_latest_trial
log "=== Done ==="
