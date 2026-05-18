#!/usr/bin/env bash
# Stage B: 2M dense CPT
set -euo pipefail

CONFIG="configs/training/stage_b_2m_cpt.toml"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

[ -d "external/flame" ] || { echo "run 'make setup-flame' first"; exit 1; }
[ -d "checkpoints/stage_a_1m/latest" ] || { echo "Stage A checkpoint missing"; exit 1; }

echo "[stage-b] config=${CONFIG} 2M CPT, resume from stage_a_1m/latest"
echo "[stage-b] (placeholder) wired up after Stage A success"
