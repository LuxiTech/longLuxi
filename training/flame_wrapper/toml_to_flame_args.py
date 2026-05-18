"""Translate configs/training/stage_*_cpt.toml into flame/torchtitan CLI args.

Source of truth: TOML (our schema).
Downstream: flame/torchtitan CLI (downstream-defined).

Usage:
    uv run python training/flame_wrapper/toml_to_flame_args.py \
        --toml configs/training/stage_a_1m_cpt.toml --phase smoke
    # prints args on stdout, one whitespace-separated string.
"""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def to_flame_args(toml_path: Path, phase: str) -> list[str]:
    data = tomllib.loads(toml_path.read_text())
    run = data["run"]
    model = data["model"]
    training = data["training"].copy()
    parallel = data["parallelism"]
    ckpt = data.get("checkpoint", {})

    # Apply phase overrides if present
    overrides = data.get(phase, {})
    if not overrides and phase == "smoke":
        # If TOML doesn't have a [smoke] block, fall back to main but smaller
        overrides = {"context_length": 262144, "max_steps": 40}
    training.update(overrides)

    seq_len = training["context_length"]
    args: list[str] = [
        f"--job.config_file={toml_path}",  # for downstream record-keeping
        f"--training.seq_len={seq_len}",
        f"--training.dtype={model['dtype']}",
        f"--training.steps={training.get('max_steps', training.get('max_steps_main', 0))}",
        f"--training.learning_rate={training['learning_rate']}",
        f"--training.warmup_steps={training.get('warmup_steps', 0)}",
        f"--training.lr_scheduler={training.get('lr_scheduler', 'cosine_with_warmup')}",
        f"--training.weight_decay={training.get('weight_decay', 0.01)}",
        f"--training.gradient_clip={training.get('max_grad_norm', 1.0)}",
        f"--training.micro_batch_size={training.get('micro_batch_size', 1)}",
        f"--training.gradient_accumulation_steps={training.get('gradient_accumulation', 1)}",
        f"--experimental.context_parallel_degree={parallel['context_parallel_degree']}",
        f"--experimental.context_parallel_rotate_method={parallel['context_parallel_rotate_method']}",
        f"--training.tensor_parallel_degree={parallel['tensor_parallel_degree']}",
        f"--training.data_parallel_degree={parallel['data_parallel_degree']}",
        f"--model.name_or_path={model.get('local_path', '/home/user01/Minko/models/Qwen3.5-4B')}",
        f"--model.attn_impl={model.get('attn_impl', 'flash_attention_2')}",
        f"--checkpoint.folder={run['output_dir']}",
        f"--checkpoint.interval={ckpt.get('save_every_steps', overrides.get('save_every_steps', 25))}",
        f"--checkpoint.keep_last={ckpt.get('keep_last', 3)}",
    ]
    if model.get("gradient_checkpointing"):
        args.append("--training.activation_checkpoint_mode=full")
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--toml", required=True, type=Path)
    p.add_argument("--phase", default="main", choices=["main", "smoke"])
    args_ns = p.parse_args()
    args = to_flame_args(args_ns.toml, args_ns.phase)
    print(" ".join(args))


if __name__ == "__main__":
    main()
