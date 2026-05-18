#!/usr/bin/env bash
# Clone LLaMA-Factory into external/ and document the install step.
# Run from repo root:
#   make setup-llamafactory

set -euo pipefail

EXTERNAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../external" && pwd)"
cd "${EXTERNAL_DIR}"

if [ ! -d "LLaMA-Factory" ]; then
    echo "[setup-lf] cloning hiyouga/LLaMA-Factory ..."
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
else
    echo "[setup-lf] LLaMA-Factory already exists, pulling latest"
    (cd LLaMA-Factory && git pull --ff-only)
fi

echo "[setup-lf] done. To install in current venv:"
echo "  uv pip install -e 'external/LLaMA-Factory[torch,metrics,deepspeed,liger-kernel]'"
echo "(暂不自动 install；SFT 启动前手动跑，避免和 flame 的 torch 版本打架)"
