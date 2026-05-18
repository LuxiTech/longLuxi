#!/usr/bin/env bash
# Stage D: Short instruction-recovery SFT
# 框架: LLaMA-Factory
set -euo pipefail

TOML="configs/training/stage_d_short_sft.toml"
LF_YAML="configs/training/_generated/stage_d_short_sft.lf.yaml"

[ -d "external/LLaMA-Factory" ] || { echo "run 'make setup-llamafactory' first"; exit 1; }
[ -d "checkpoints/stage_b_2m/latest" ] || { echo "Stage B ckpt missing"; exit 1; }
[ -f "data/processed/short_sft_mix.jsonl" ] || { echo "short_sft_mix.jsonl missing; run data/scripts/build_short_sft.py"; exit 1; }

echo "[stage-d] regenerating LF yaml from $TOML"
uv run python training/llamafactory_wrapper/toml_to_lf_yaml.py --toml "$TOML" --out "$LF_YAML"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "[stage-d] launching llamafactory-cli on ${NNODES}x${NPROC_PER_NODE}"

# TODO[W8]: 真启动 — 现在先 dry print，验证 yaml 生成 OK
# torchrun \
#   --nproc-per-node "${NPROC_PER_NODE}" --nnodes "${NNODES}" --node-rank "${NODE_RANK}" \
#   --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT}" \
#   external/LLaMA-Factory/src/llamafactory/launcher.py \
#   "$LF_YAML"

echo "[stage-d] (placeholder) full launch wired up in W8"
echo "[stage-d] generated yaml: $LF_YAML"
