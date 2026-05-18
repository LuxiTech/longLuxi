#!/usr/bin/env bash
# Stage C: 4M boundary CPT (OPTIONAL / STRETCH)
set -euo pipefail

CONFIG="configs/training/stage_c_4m_cpt.toml"
[ -d "external/flame" ] || { echo "run 'make setup-flame' first"; exit 1; }
[ -d "checkpoints/stage_b_2m/latest" ] || { echo "Stage B checkpoint missing"; exit 1; }

echo "[stage-c] OPTIONAL 4M CPT — only run if Stage B success criteria met."
echo "[stage-c] go/no-go gate must be approved before this."
echo "[stage-c] (placeholder) wired up after Stage B success"
