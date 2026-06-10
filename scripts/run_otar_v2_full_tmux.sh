#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="/Users/chenjunming/Desktop/myenv_312/bin/python"
TMUX_SESSION="otar_v2_full"
WORKDIR="$PROJECT_ROOT"

CONFIG="${1:-configs/paper/otar_small8_pilot.yaml}"
MATRIX="${2:-configs/paper/otar_formal_matrix.yaml}"
OUTPUT="${3:-outputs/otar_v2_full}"
RUN_NAME="${4:-OTAR_v2_full}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    log "WARN: tmux session '$TMUX_SESSION' already exists."
    log "  Attach: tmux attach -t $TMUX_SESSION"
    log "  Kill:   tmux kill-session -t $TMUX_SESSION"
    exit 1
fi

log "Creating tmux session: $TMUX_SESSION"
log "  Config: $CONFIG"
log "  Matrix: $MATRIX"
log "  Output: $OUTPUT"
log "  Run Name: $RUN_NAME"

tmux new-session -d -s "$TMUX_SESSION" -c "$WORKDIR"

tmux send-keys -t "$TMUX_SESSION" \
    "caffeinate -dims $PYTHON -m src.experiments.otar_v2_full \
  --config '$CONFIG' \
  --formal-matrix '$MATRIX' \
  --output '$OUTPUT' \
  --run-name '$RUN_NAME' \
  --resume-completed-children \
  2>&1 | tee 'logs/otar_v2_full_$(date +%Y%m%d_%H%M%S).log'" C-m

log "tmux session '$TMUX_SESSION' started."
log "  Attach: tmux attach -t $TMUX_SESSION"
log "  Detach: Ctrl-b then d"
log "  Status: tmux capture-pane -t $TMUX_SESSION -p | tail -20"
