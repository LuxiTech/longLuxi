#!/usr/bin/env bash
# Poll nvidia-smi every 2s for `${DURATION:-120}` seconds, log per-card used MB.
# Usage: DURATION=120 bash scripts/probe_gpu_mem.sh > /tmp/gpu_probe.csv
set -euo pipefail
DURATION="${DURATION:-120}"
INTERVAL="${INTERVAL:-2}"
END=$(( $(date +%s) + DURATION ))
echo "timestamp,gpu0_mb,gpu1_mb,gpu2_mb,gpu3_mb"
while [[ $(date +%s) -lt $END ]]; do
  ts=$(date +%s)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 | paste -sd,)
  echo "$ts,$mem"
  sleep "$INTERVAL"
done
