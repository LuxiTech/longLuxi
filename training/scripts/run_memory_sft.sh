#!/usr/bin/env bash
# Stage E: Memory-format SFT (512K context, ~20K teacher-distilled examples)
set -euo pipefail

CONFIG="configs/training/stage_e_memory_sft.toml"
[ -f "data/processed/memory_sft_mix.jsonl" ] || {
    echo "[stage-e] memory SFT data not built. Run:"
    echo "  uv run python data/scripts/build_memory_sft.py --queries ... --retrieval-corpus ... --out data/processed/memory_sft_mix.jsonl"
    exit 1
}

echo "[stage-e] memory-format SFT"
echo "[stage-e] (placeholder) wired up in W9"
