#!/usr/bin/env bash
# Install flash-attn from the locally-staged prebuilt wheel.
#
# Why local wheel and not PyPI?
# - PyPI flash-attn builds against the installed torch (~3hr compile on H100 box for cu13).
# - The prebuilt wheel matches torch==2.8 + cu12 — paired with the torch pin in pyproject.toml
#   (`torch>=2.8,<2.9`), this is the working combo as of 2026-05.
#
# Run from repo root after any `uv sync`:
#   bash scripts/install_flash_attn_local.sh

set -euo pipefail

WHEEL="/home/user01/Minko/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

if [ ! -f "$WHEEL" ]; then
    echo "[install-fa] wheel not found at $WHEEL"
    echo "             Download from https://github.com/Dao-AILab/flash-attention/releases"
    echo "             (need torch2.8 + cu12 + cp311 cxx11abiFALSE)"
    exit 1
fi

echo "[install-fa] installing $(basename "$WHEEL") ..."
uv pip install --no-deps "$WHEEL"

echo "[install-fa] smoke-importing flash_attn ..."
uv run python -c "
import flash_attn
from flash_attn import flash_attn_func
print(f'flash_attn {flash_attn.__version__} imported OK')
import torch
print(f'torch {torch.__version__}')
"

echo "[install-fa] done."
