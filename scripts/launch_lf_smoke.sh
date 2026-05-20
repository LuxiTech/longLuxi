#!/usr/bin/env bash
# Launch a LLaMA-Factory pt-stage training under FSDP2 on cards 0-3.
# Calls launcher.py directly to avoid llamafactory-cli's double-torchrun fork bomb.
#
# Usage:
#   bash scripts/launch_lf_smoke.sh configs/training/stage_b_v3_128k_fsdp2.yaml
#
# Optional env:
#   CUDA_VISIBLE_DEVICES   default "0,1,2,3" (hard rule: never use cards 4-7)
#   ACC_CFG                accelerate config (default configs/accelerate/fsdp2_cards0_3.yaml)

set -euo pipefail
cd "$(dirname "$0")/.."

CFG="${1:?usage: $0 <training.yaml>}"
ACC_CFG="${ACC_CFG:-configs/accelerate/fsdp2_cards0_3.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VIRTUAL_ENV="$(pwd)/.venv-lf"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

echo "[launch_lf] cfg=$CFG acc=$ACC_CFG gpus=$CUDA_VISIBLE_DEVICES"

# Call launcher.py directly. `llamafactory.cli train` triggers an internal
# torchrun --nproc-per-node=4 under each of accelerate's 4 worker procs,
# producing a 16-worker fork bomb.
exec accelerate launch \
    --config_file "$ACC_CFG" \
    --num_processes 4 \
    external/LLaMA-Factory/src/llamafactory/launcher.py "$CFG"
