"""Phase 2 smoke: minimal HF transformers CPT on Qwen3.5-4B at 256K context.

WHY this script exists (not flame):
- flame is designed for from-scratch FLA pretraining with custom JSON arch configs.
- CPT from HF safetensors via flame requires DCP conversion + torchtitan install +
  matching flame's TOML schema. Too many moving pieces for a 10M-token smoke run.
- This script does the minimum to prove the CPT loop works end-to-end on our 4-card
  budget. Phase 3 will adopt a proper framework (flame + DCP OR LLaMA-Factory pt).

Constraints baked in:
- 4× H100 80GB (cards 0-3 via CUDA_VISIBLE_DEVICES).
- bf16 + SDPA attention (no flash-attn — torch 2.11+cu13 vs flash-attn 2.8 wheel ABI clash).
- FSDP-2 full shard for weights + Adam states; grad checkpointing required at 256K seq.
- Reads pre-packed sequences from data/processed/packed_256k.jsonl.

Usage (4-GPU, foreground):
    cd /home/user01/Minko/longluxi
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run accelerate launch \\
            --num_processes 4 --mixed_precision bf16 \\
            --use_fsdp --fsdp_sharding_strategy FULL_SHARD \\
            --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \\
            --fsdp_state_dict_type SHARDED_STATE_DICT \\
            training/scripts/cpt_smoke_hf.py \\
            --model /home/user01/Minko/models/Qwen3.5-4B \\
            --data data/processed/packed_256k.jsonl \\
            --output-dir checkpoints/stage_a_smoke \\
            --max-steps 10 --seq-len 262144 --lr 1e-5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Avoid HF tokenizers warning under multiprocessing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to HF model dir (Qwen3.5-4B)")
    p.add_argument("--data", required=True, help="packed jsonl from data/scripts/pack_docs.py")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=262144)
    p.add_argument("--lr", type=float, default=1.0e-5)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from accelerate import Accelerator
    from accelerate.utils import set_seed

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)

    accel = Accelerator(mixed_precision="bf16")
    if accel.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[smoke] world_size={accel.num_processes} mixed_precision=bf16 model={args.model}", flush=True)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Model (FSDP via accelerate-launch flags) ----
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    # Increase max_position_embeddings to accommodate seq_len. The local config.json
    # already has 1010000 (YaRN-baked); but we override here defensively in case it isn't.
    if hasattr(config, "text_config"):
        config.text_config.max_position_embeddings = max(
            args.seq_len + 16, getattr(config.text_config, "max_position_embeddings", args.seq_len + 16)
        )

    if accel.is_main_process:
        print(f"[smoke] loading model {args.model} bf16 sdpa ...", flush=True)
    # Try flash_attention_2 first (the wheel-installed flash-attn matches our torch 2.8+cu12);
    # fall back to sdpa if import fails (CI / fresh env).
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "sdpa"
        if accel.is_main_process:
            print("[smoke] flash_attn missing -> falling back to sdpa", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl, trust_remote_code=True,
    )
    # Activation checkpointing — essential at 256K
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False  # incompatible with grad ckpt

    # Qwen3.5 has vocab=250K. Materializing logits [B, T, 250K] in fp32 burns
    # ~8GB at seq=8K and is the dominant OOM cause. Use FLA's fused linear-CE
    # which never materializes the full logits tensor.
    from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
    fused_loss = FusedLinearCrossEntropyLoss(reduction="mean", ignore_index=-100)

    # ---- Data: streaming sequences from packed jsonl ----
    class PackedJsonl(IterableDataset):
        def __init__(self, path: str, tok, seq_len: int, world_size: int, rank: int):
            self.path = path
            self.tok = tok
            self.seq_len = seq_len
            self.world_size = world_size
            self.rank = rank

        def __iter__(self):
            with open(self.path) as f:
                for i, line in enumerate(f):
                    if i % self.world_size != self.rank:
                        continue
                    rec = json.loads(line)
                    text = rec["text"]
                    ids = self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                    # pad / truncate to exactly seq_len
                    if ids.numel() >= self.seq_len:
                        ids = ids[: self.seq_len]
                    else:
                        pad = torch.full((self.seq_len - ids.numel(),), self.tok.pad_token_id, dtype=ids.dtype)
                        ids = torch.cat([ids, pad])
                    # Loop the dataset if exhausted (so we can do max_steps regardless of data size)
                    yield {"input_ids": ids, "labels": ids.clone()}

    ds = PackedJsonl(args.data, tokenizer, args.seq_len, accel.num_processes, accel.process_index)

    def collate(batch):
        # micro_batch=1; just stack
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        # Mask padding from loss
        labels = labels.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}

    loader = DataLoader(ds, batch_size=1, collate_fn=collate, num_workers=0)

    # ---- Optimizer + schedule ----
    # 8-bit Adam to fit 4B model + states on 4xH100 (full Adam fp32 = ~60GB, 8-bit = ~15GB)
    try:
        import bitsandbytes as bnb
        optim = bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr,
                                          betas=(0.9, 0.95), eps=1e-8,
                                          weight_decay=args.weight_decay)
        if accel.is_main_process:
            print("[smoke] using PagedAdamW8bit (bitsandbytes)", flush=True)
    except ImportError:
        if accel.is_main_process:
            print("[smoke] bitsandbytes missing -> falling back to torch AdamW fp32 (will OOM at 4B)", flush=True)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                   eps=1e-8, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        # cosine to 10% of peak after max_steps
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    model, optim, loader, sched = accel.prepare(model, optim, loader, sched)

    # ---- Train loop ----
    step = 0
    data_iter = iter(loader)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        t0 = time.time()
        with accel.accumulate(model):
            # Forward to hidden states only — avoid the huge fp32 logits + CE inside the model.
            # Under DDP/FSDP the inner CausalLM is wrapped; unwrap once and reuse its backbone.
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            inner = accel.unwrap_model(model)
            backbone = inner.model  # Qwen3_5Model
            outputs = backbone(input_ids=input_ids, use_cache=False, output_hidden_states=False)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            # Shift for causal LM: predict token t+1 from hidden t
            shift_hidden = hidden[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # FLA fused: forward signature is (x, target, weight, bias).
            # Internally flattens batch/seq, computes loss without materializing full logits.
            loss = fused_loss(shift_hidden, shift_labels, inner.lm_head.weight)
            accel.backward(loss)
            if accel.sync_gradients:
                accel.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()
            optim.zero_grad()
        dt = time.time() - t0
        if accel.is_main_process:
            cur_lr = sched.get_last_lr()[0]
            print(f"[smoke] step {step+1}/{args.max_steps} loss={loss.item():.4f} lr={cur_lr:.2e} dt={dt:.1f}s", flush=True)
        step += 1
        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_dir = args.output_dir / f"step_{step}"
            if accel.is_main_process:
                print(f"[smoke] saving ckpt -> {ckpt_dir}", flush=True)
            accel.wait_for_everyone()
            unwrapped = accel.unwrap_model(model)
            unwrapped.save_pretrained(ckpt_dir, is_main_process=accel.is_main_process,
                                       save_function=accel.save, state_dict=accel.get_state_dict(model))
            if accel.is_main_process:
                tokenizer.save_pretrained(ckpt_dir)
                # symlink latest
                latest = args.output_dir / "latest"
                if latest.exists() or latest.is_symlink():
                    latest.unlink()
                latest.symlink_to(ckpt_dir.name)
                print(f"[smoke] saved {ckpt_dir}", flush=True)

    if accel.is_main_process:
        print("[smoke] done.", flush=True)


if __name__ == "__main__":
    main()
