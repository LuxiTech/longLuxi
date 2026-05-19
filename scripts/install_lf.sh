#!/usr/bin/env bash
# Build isolated LF training venv. Idempotent: re-run after edits.
# Outputs: .venv-lf/ with torch 2.8 + flash-attn 2.8.3 + LF + accelerate 1.11 + transformers 5.6.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d .venv-lf ]]; then
  uv venv .venv-lf --python 3.11
fi

# Activate by setting VIRTUAL_ENV so `uv pip` targets the right venv.
export VIRTUAL_ENV="$(pwd)/.venv-lf"

# 1) Pin torch 2.8.0+cu128 first (flash-attn local wheel was built against this).
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# 2) LF editable + its training extras (datasets, peft, trl, deepspeed kept off-train path).
# --index-strategy unsafe-best-match lets uv pull `requests` and other generic deps from
# pypi even though the pytorch index ships older copies (pytorch index has requests==2.28.1
# which conflicts with datasets>=3.0.0 needing requests>=2.32.2).
uv pip install -e ./external/LLaMA-Factory \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match

# 3) Force LF-compatible pins (LF's pyproject allows ranges; lock at upper bound that we know works).
# LF 0.9.5.dev0 actually requires trl>=0.18,<=0.24 and datasets>=2.16,<=4.0 — we hold the line
# at the upper edge of LF's allowed ranges that we have empirically verified compiles cleanly.
uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    "transformers>=4.55,<=5.6.0" \
    "accelerate>=1.10,<=1.11.0" \
    "datasets>=3.0,<4.0" \
    "trl>=0.18,<=0.24"

# 4) Flash-attn local wheel (must come AFTER torch is pinned).
uv pip install /home/user01/Minko/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-deps

# 5) Smoke imports. Use the venv's python explicitly — uv sets VIRTUAL_ENV but doesn't
# automatically put .venv-lf/bin on PATH, so a bare `python` would resolve to whatever
# is on PATH (typically the main .venv/ python, which lacks llamafactory).
"$VIRTUAL_ENV/bin/python" -c "
import torch, transformers, accelerate, llamafactory, flash_attn
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} ngpu={torch.cuda.device_count()}')
print(f'transformers={transformers.__version__}')
print(f'accelerate={accelerate.__version__}')
print(f'llamafactory={llamafactory.__version__}')
print(f'flash_attn={flash_attn.__version__}')
"
echo '[install_lf] OK'
