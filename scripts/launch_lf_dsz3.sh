#!/usr/bin/env bash
# Launch LF pt training under DeepSpeed ZeRO-3 on cards 0-3.
# Reads deepspeed: <path> from the LF YAML; we just torchrun the launcher.
# Usage: bash scripts/launch_lf_dsz3.sh configs/training/lf_qwen35_4b_smoke_256k_dsz3.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="${1:?usage: $0 <training.yaml>}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VIRTUAL_ENV="$(pwd)/.venv-lf"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

# DS gets confused by torch's distributed launcher env vars if mixed with
# accelerate. We use the standalone torchrun path; LF detects `deepspeed:` in
# the YAML and initializes DS engine itself.
echo "[launch_lf_dsz3] cfg=$CFG gpus=$CUDA_VISIBLE_DEVICES"

exec torchrun --standalone --nproc_per_node=4 \
    external/LLaMA-Factory/src/llamafactory/launcher.py "$CFG"
