#!/usr/bin/env bash
# Launch LF pt training under FSDP2 on cards 0-3.
# Usage: bash scripts/launch_lf_smoke.sh configs/training/lf_qwen35_4b_smoke_32k.yaml
#
# NOTE: we invoke `launcher.py` as a script (NOT `-m llamafactory.cli`).
# LF's cli.py routes through `launcher.launch()`, which auto-spawns its own
# `torchrun --nproc-per-node $(get_device_count)` whenever it sees >1 visible
# GPU and FORCE_TORCHRUN is unset. Under `accelerate launch --num_processes 4`
# that double-launches a 4x4=16 worker fork bomb (observed once, killed). Running
# `launcher.py` as __main__ skips the command-parser branch and calls run_exp()
# directly, so accelerate's own process group is the only one.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="${1:?usage: $0 <training.yaml>}"
ACC_CFG="${ACC_CFG:-configs/accelerate/fsdp2_cards0_3.yaml}"
LF_LAUNCHER="external/LLaMA-Factory/src/llamafactory/launcher.py"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VIRTUAL_ENV="$(pwd)/.venv-lf"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

echo "[launch_lf] cfg=$CFG acc=$ACC_CFG gpus=$CUDA_VISIBLE_DEVICES"

exec accelerate launch \
    --config_file "$ACC_CFG" \
    --num_processes 4 \
    "$LF_LAUNCHER" "$CFG"
