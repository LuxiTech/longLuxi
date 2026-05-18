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

# Force FlashInfer attention backend: our installed flash-attn wheel is built for
# torch 2.8+cu12, but current venv torch is 2.11+cu13 (transitive dep of vllm 0.21).
# FlashInfer is installed and matches torch 2.11+cu13. flash-attn stays in the env
# for flame training (which is launched separately).
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export VLLM_USE_FLASH_ATTN="${VLLM_USE_FLASH_ATTN:-0}"

# YaRN factor=4 (1M ctx) is baked into the local model's config.json under
# text_config.rope_parameters. Original config preserved at config.json.bak.
# vLLM 0.21+ no longer accepts --rope-scaling on the CLI; it reads HF config.

echo "[vllm-serve] model=$MODEL port=$PORT tp=$TP max_len=$MAX_LEN gpus=$CUDA_VISIBLE_DEVICES"

exec uv run vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --enable-chunked-prefill \
    --no-enable-log-requests
