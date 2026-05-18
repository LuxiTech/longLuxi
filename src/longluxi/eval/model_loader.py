"""Load HF causal LM + tokenizer with optional YaRN config override.

Heavy imports (torch, transformers) are gated to keep --dry-run lightweight.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yarn_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    return data["rope_parameters"]


def apply_yarn_to_config(model_config, rope_params: dict[str, Any]) -> None:
    """Patch a HF AutoConfig in place.

    Qwen3.5 uses `rope_parameters` (multi-RoPE);
    older HF models use `rope_scaling`.
    """
    if hasattr(model_config, "rope_parameters"):
        model_config.rope_parameters = rope_params
    else:
        model_config.rope_scaling = {
            "rope_type": rope_params["rope_type"],
            "factor": rope_params["factor"],
            "original_max_position_embeddings": rope_params["original_max_position_embeddings"],
        }


def _resolve_attn_impl(requested: str) -> str:
    """Return the best available attn implementation.

    If flash_attention_2 is requested but flash_attn is not installed,
    fall back to sdpa (PyTorch scaled-dot-product-attention, available
    in torch >= 2.0 and H100-compatible).
    """
    if requested != "flash_attention_2":
        return requested
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        import warnings
        warnings.warn(
            "flash-attn not installed; falling back to attn_implementation='sdpa'. "
            "Install flash-attn for optimal memory use at long contexts.",
            stacklevel=3,
        )
        return "sdpa"


def load_model_and_tokenizer(model_id: str, yarn_path: str | Path | None = None,
                             dtype: str = "bfloat16", attn_impl: str = "flash_attention_2"):
    """Heavy path: actually load model. Requires torch+transformers.

    flash-attn is used when available; sdpa is the fallback.
    Returns (model, tokenizer, config).
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    resolved_impl = _resolve_attn_impl(attn_impl)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if yarn_path:
        apply_yarn_to_config(config, load_yarn_json(yarn_path))

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=resolved_impl,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, config
