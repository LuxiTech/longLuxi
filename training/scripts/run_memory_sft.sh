#!/usr/bin/env bash
# Stage E: Memory-format SFT
# 框架: LLaMA-Factory + DeepSpeed Ulysses
set -euo pipefail

TOML="configs/training/stage_e_memory_sft.toml"
LF_YAML="configs/training/_generated/stage_e_memory_sft.lf.yaml"

[ -d "external/LLaMA-Factory" ] || { echo "run 'make setup-llamafactory' first"; exit 1; }
[ -f "data/processed/memory_sft_mix.jsonl" ] || {
    echo "[stage-e] memory SFT data not built. Run:"
    echo "  uv run python data/scripts/build_memory_sft.py --queries ... --retrieval-corpus ... --out data/processed/memory_sft_mix.jsonl"
    exit 1
}

echo "[stage-e] regenerating LF yaml from $TOML"
uv run python training/llamafactory_wrapper/toml_to_lf_yaml.py --toml "$TOML" --out "$LF_YAML"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "[stage-e] launching llamafactory-cli with Ulysses SP=16 on ${NNODES}x${NPROC_PER_NODE}"
echo "[stage-e] (placeholder) full launch wired up in W9"
echo "[stage-e] generated yaml: $LF_YAML"
