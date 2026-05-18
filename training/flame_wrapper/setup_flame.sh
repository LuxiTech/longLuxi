#!/usr/bin/env bash
# Clone flame + flash-linear-attention into external/ and install in editable mode.
# Run from repo root:
#   make setup-flame
# 或：
#   bash training/flame_wrapper/setup_flame.sh

set -euo pipefail

EXTERNAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../external" && pwd)"
cd "${EXTERNAL_DIR}"

# flame
if [ ! -d "flame" ]; then
    echo "[setup-flame] cloning fla-org/flame ..."
    git clone --depth 1 https://github.com/fla-org/flame.git
else
    echo "[setup-flame] flame already exists, pulling latest"
    (cd flame && git pull --ff-only)
fi

# flash-linear-attention (depended by flame for FLA kernels)
if [ ! -d "flash-linear-attention" ]; then
    echo "[setup-flame] cloning fla-org/flash-linear-attention ..."
    git clone --depth 1 https://github.com/fla-org/flash-linear-attention.git
else
    echo "[setup-flame] flash-linear-attention already exists, pulling latest"
    (cd flash-linear-attention && git pull --ff-only)
fi

echo "[setup-flame] done. To install in current venv:"
echo "  uv pip install -e external/flame"
echo "  uv pip install -e external/flash-linear-attention"
echo "(暂不自动 install，避免污染默认 lockfile；CPT 启动前手动跑)"
