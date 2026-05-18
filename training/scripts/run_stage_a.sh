#!/usr/bin/env bash
# Stage A: 1M dense CPT
# 用法:
#   bash training/scripts/run_stage_a.sh --smoke      # smoke 50M tokens
#   bash training/scripts/run_stage_a.sh              # main 250M tokens
#
# 多节点示例（torchrun，假设 head=node0，rank0 / rank1 在两台机器上分别跑）:
#   MASTER_ADDR=node0 NODE_RANK=0 bash training/scripts/run_stage_a.sh
#   MASTER_ADDR=node0 NODE_RANK=1 bash training/scripts/run_stage_a.sh

set -euo pipefail

PHASE="main"
if [[ "${1:-}" == "--smoke" ]]; then PHASE="smoke"; fi

CONFIG="configs/training/stage_a_1m_cpt.toml"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

FLAME_DIR="external/flame"
if [ ! -d "${FLAME_DIR}" ]; then
    echo "[stage-a] flame not found in ${FLAME_DIR}; run 'make setup-flame' first."
    exit 1
fi

echo "[stage-a] phase=${PHASE} config=${CONFIG} nproc=${NPROC_PER_NODE} nnodes=${NNODES} rank=${NODE_RANK}"

# TODO[W2]: 把 stage_a_1m_cpt.toml 通过 flame_wrapper/config_to_flame.py 转成 flame CLI 参数。
# 暂以占位 echo 形式，等 W2 flame smoke 通了再实装。
#
# torchrun \
#   --nproc-per-node "${NPROC_PER_NODE}" \
#   --nnodes "${NNODES}" \
#   --node-rank "${NODE_RANK}" \
#   --master-addr "${MASTER_ADDR}" \
#   --master-port "${MASTER_PORT}" \
#   external/flame/train.py \
#   --config "${CONFIG}" \
#   --phase "${PHASE}"

echo "[stage-a] (placeholder) actual launch wired up in W2"
