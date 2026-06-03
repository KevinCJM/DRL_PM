#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/background_logs

process_active() {
  local pattern="$1"
  pgrep -f "$pattern" >/dev/null 2>&1
}

if process_active 'scripts/exp37b_resume_after_mainplusp9.sh'; then
  echo "[skip] EXP37B_resume_after_mainplusp9 already_running"
else
  nohup bash -lc 'bash scripts/exp37b_resume_after_mainplusp9.sh' > results/background_logs/EXP37B_resume_after_mainplusp9.log 2>&1 &
  echo "[start] EXP37B_resume_after_mainplusp9"
fi

echo "started"
