"""Verify Qwen3.5-4B loads in the LF venv with YaRN config intact.

This catches: (a) transformers version too old to know Qwen3_5ForCausalLM,
(b) trust_remote_code path not wired, (c) YaRN config dropped on load.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LF_VENV_PY = REPO_ROOT / ".venv-lf" / "bin" / "python"
MODEL_PATH = "/home/user01/Minko/models/Qwen3.5-4B"


def test_qwen35_loads_in_lf_venv():
    assert LF_VENV_PY.exists(), f"LF venv missing — run scripts/install_lf.sh first ({LF_VENV_PY})"
    script = f"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM

cfg = AutoConfig.from_pretrained({MODEL_PATH!r}, trust_remote_code=True)
assert cfg.__class__.__name__ in ('Qwen3_5Config', 'Qwen3Config'), f'unexpected config class: {{cfg.__class__.__name__}}'

# YaRN: factor=4 should be present in text_config.rope_parameters or rope_scaling.
text_cfg = getattr(cfg, 'text_config', cfg)
rope = getattr(text_cfg, 'rope_parameters', None) or getattr(text_cfg, 'rope_scaling', None)
assert rope is not None, 'YaRN config missing — model config.json not patched'
factor = rope.get('factor') if isinstance(rope, dict) else None
assert factor == 4, f'YaRN factor != 4 ({{factor}})'

# Load weights to CPU to keep test fast; just need the class to resolve.
model = AutoModelForCausalLM.from_pretrained(
    {MODEL_PATH!r}, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='meta'
)
assert model.__class__.__name__ in ('Qwen3_5ForCausalLM', 'Qwen3ForCausalLM'), \\
    f'unexpected model class: {{model.__class__.__name__}}'
print('OK', model.__class__.__name__, 'YaRN factor=', factor)
"""
    proc = subprocess.run(
        [str(LF_VENV_PY), "-c", script],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert "OK" in proc.stdout, proc.stdout
