#!/usr/bin/env bash
# Run after RULER 2M eval finishes. Stops vLLM, then kicks off Stage A 128K CPT.
# Stage A uses FSDP2 (proven) at 128K — slower path to 2M improvement but safe.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="${CFG:-configs/training/stage_a_v2_128k_fsdp2.yaml}"
LOG="/tmp/stage_a_v2_128k.log"

# 1. Stop vllm — kill any procs holding GPU memory on cards 0-3.
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1 | grep -v '^$' | xargs)
if [[ -n "$PIDS" ]]; then
    echo "[stage_a_kickoff] killing GPU pids: $PIDS"
    kill -9 $PIDS 2>/dev/null || true
fi
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 0,1,2,3

# 2. Launch Stage A under FSDP2 + Liger + bf16.
echo "[stage_a_kickoff] launching $CFG at $(date)"
bash scripts/launch_lf_smoke.sh "$CFG" > "$LOG" 2>&1 &
TRAIN_PID=$!
echo "[stage_a_kickoff] training pid=$TRAIN_PID, log=$LOG"
sleep 5
echo "[stage_a_kickoff] confirming launcher running..."
pgrep -af "launcher.py.*stage_a_v2_128k" | head -3
echo "[stage_a_kickoff] kickoff complete @ $(date)"
