#!/usr/bin/env bash
# Launch vLLM serving Qwen3.5-4B with YaRN factor=4 (1M context) on cards 0-3.
# Defaults: TP=4, max-model-len=1010000 (matches Qwen3.5-4B official 1M extrapolation cap).
#
# Usage:
#   bash eval/runners/vllm_server.sh                  # foreground
#   bash eval/runners/vllm_server.sh > vllm.log 2>&1 & # background
#
# Then in another shell, hit http://localhost:8001/v1/completions

set -euo pipefail

MODEL="${MODEL:-/home/user01/Minko/models/Qwen3.5-4B}"
PORT="${PORT:-8001}"
TP="${TP:-4}"
MAX_LEN="${MAX_LEN:-1010000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
DTYPE="${DTYPE:-bfloat16}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# YaRN config matches configs/yarn/yarn_1m.json: factor=4.0, original=262144
ROPE_SCALING='{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}'

echo "[vllm-serve] model=$MODEL port=$PORT tp=$TP max_len=$MAX_LEN gpus=$CUDA_VISIBLE_DEVICES"

exec uv run vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --rope-scaling "$ROPE_SCALING" \
    --enable-chunked-prefill \
    --disable-log-requests
