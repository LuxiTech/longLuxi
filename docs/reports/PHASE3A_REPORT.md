# Phase 3a Report — LLaMA-Factory FSDP2 Smoke, Headline at 128K

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Tag | `phase3a-lf-smoke` |
| Hardware | 4× H100 80GB (cards 0-3) |
| Training | LF `pt` stage + accelerate FSDP2 (TRANSFORMER_BASED_WRAP on `Qwen3_5DecoderLayer`) |
| Eval | vLLM TP=4 against the smoke ckpt |
| Data | `longluxi_packed_256k` (30 × ~260K tokens) registered into LF |

## TL;DR

**The Phase 2 multi-GPU blocker is dead.** FSDP2 (not FSDP1) truly shards optimizer state + params across 4 cards. We trained 4-GPU CPT smoke runs through 128K context on Qwen3.5-4B with bf16, flash-attn 2, and gradient checkpointing. At 256K, even with `fsdp_offload_params=true`, activations push past the 80 GiB cap on the first forward — 256K dense CPT on 4 H100s requires **sequence parallelism** (Ulysses or Ring-Attention), which moves to Phase 3b.

The 128K smoke ckpt round-trips through vLLM cleanly with no eval regression (RULER 128K = 82%, NIAH 128K = 100%).

## What Phase 3a delivered

1. **Isolated LF training venv** at `.venv-lf/` (torch 2.8.0 + transformers 5.6.0 + accelerate 1.11.0 + flash-attn 2.8.3 + LLaMA-Factory 0.9.5.dev0). Installer `scripts/install_lf.sh` is idempotent and auto-applies the transformers s_aux None-guard patch.
2. **Dataset registration**: `longluxi_packed_256k` registered in LF via `dataset_info.json` + symlink, `columns: {prompt: text}`. LF clone flattened (inner `.git/` removed; commit pinned at `external/LLaMA-Factory.commit-pin`).
3. **FSDP2 accelerate config** (`configs/accelerate/fsdp2_cards0_3.yaml`) tuned for cards 0-3 (4 procs, FSDP2, FULL_STATE_DICT save).
4. **Four LF training YAMLs** (`configs/training/lf_qwen35_4b_smoke_{32k,64k,128k,256k}.yaml`) — each runs a 3-step CPT smoke at the given ctx.
5. **GPU memory profile** at each ctx ramp (1 sample per 2 s during the run):

   | ctx | peak per-card MiB | bottleneck | result |
   |-----|--------|---|---|
   | 32K | 28K steady → 43K (save spike) | save-time gather | ✅ |
   | 64K | 59K | training steady-state | ✅ |
   | 128K | 80K (training, at cap) | training activations | ✅ |
   | 256K | 80K → OOM at first forward | activations exceed cap | ❌ (needs SP) |

6. **256K with CPU offload tried** (`fsdp_offload_params=true` saves ~24 GiB/card of params+opt state) — still OOMs because activation memory dominates at 256K.
7. **128K-CPT ckpt round-trip eval** via vLLM TP=4:
   - **RULER 128K overall: 82%** (baseline 84%, -2pp within noise)
     - niah_single_1: 100%
     - niah_multikey_1: 100%
     - niah_multiquery: 100%
     - **vt: 10%** (baseline 20%, both still very low — 3 steps × 5e-6 LR can't move this; this is exactly the metric Phase 3b/3c needs to lift)
     - qa_1: 100%
   - **NIAH 128K: 100%** (matches baseline)

## Headline numbers vs Phase 1/2

| Eval | phase1-baseline | phase2-smoke | **phase3a-lf-smoke (128K-CPT)** | Δ vs baseline |
|---|---|---|---|---|
| NIAH 32K | 100% | (n/a) | (n/a) | — |
| NIAH 128K | 100% | (n/a) | **100%** | 0 |
| NIAH 1M (vLLM) | BLOCKED | 100% | (n/a) | (verified in P2) |
| RULER 128K overall | 84% | 84% | **82%** | -2 (noise) |
| RULER 128K vt | 20% | 20% | **10%** | -10 (n=10 noise; smoke trained nothing) |

The vt drop from 20→10 with n=10 cells is within sampling noise; the 3-step smoke at LR=5e-6 makes effectively zero progress. **vt remains Phase 3b/3c's headline target.**

## What changed mechanically (code + configs)

**Committed in this phase:**
- `scripts/install_lf.sh` — idempotent venv builder + auto-patch
- `scripts/launch_lf_smoke.sh` — accelerate-launch wrapper (calls `launcher.py` directly, bypasses LF's double-torchrun bug)
- `scripts/probe_gpu_mem.sh` — per-card memory polling
- `configs/accelerate/fsdp2_cards0_3.yaml` — FSDP2 config (4 cards)
- `configs/accelerate/fsdp2_cards0_3_offload.yaml` — CPU-offload variant for 256K attempt
- `configs/training/lf_qwen35_4b_smoke_{32k,64k,128k,256k}.yaml` — LF pt YAMLs
- `external/LLaMA-Factory/data/dataset_info.json` — added `longluxi_packed_256k` entry
- `external/LLaMA-Factory/data/longluxi_packed_256k.jsonl` — symlink to our packed jsonl
- `external/LLaMA-Factory.commit-pin` — records LF clone commit (6b9df75)
- `tests/test_lf_qwen35_loads.py`, `tests/test_lf_dataset_register.py` — 3 new tests
- `src/longluxi/eval/ruler.py` — truncate haystack to target length in `niah_multikey_1`, `niah_multiquery`, `qa_1` builders (previously only `vt` did this; without the fix, prompts exceed cutoff_len by 8-30K tokens at high ctx)
- `eval/results/phase3a_{ruler_128k,niah_128k}/` — round-trip eval results

## LF YAML gotchas (folded into the configs)

These were not in the LF examples but are required for Qwen3.5 + our data:

1. `template: empty` — Qwen3.5's chat template requires a user message; pt stage with packed plain text has no chat structure. Without this, LF errors at preprocess.
2. `enable_liger_kernel: true` — the 248,320-vocab × 32K-seq fp32 logits tensor is ~30 GiB; without Liger fused linear+CE it OOMs even under FSDP2.
3. `dataset_dir: external/LLaMA-Factory/data` — LF needs to know where to resolve `longluxi_packed_256k.jsonl` (defaults to its own `data/`).
4. `packing: false`, `neat_packing: false` — our sequences are already packed upstream; LF must not re-pack.
5. `flash_attn: fa2` — wire flash-attn 2 through LF's attention path.

## Environment topology (final)

| Venv | Purpose | torch | transformers | accelerate | flash-attn | vllm |
|---|---|---|---|---|---|---|
| `.venv/` | vLLM eval | 2.11.0+cu130 | 5.8.1 | 1.13.0 | (removed)* | 0.21.0 |
| `.venv-lf/` | LF training | 2.8.0+cu128 | 5.6.0 | 1.11.0 | 2.8.3 | (not installed) |

*flash-attn removed from `.venv/`: its ABI is locked to torch 2.8 and conflicts with vLLM 0.21.0's `_C.abi3.so` (which expects a newer torch). vLLM serves via FlashInfer instead. flash-attn lives in `.venv-lf/` for training.

## Phase 2 blocker conclusively closed

PHASE2_REPORT noted: "accelerate FSDP doesn't shard cleanly with HF Qwen3.5 + accelerate; all 4 ranks OOM at 65K-32K before any activations." That was FSDP1 with a wrap policy that didn't target `Qwen3_5DecoderLayer`. With FSDP2 (`fsdp_version: 2`) + correct wrap, sharding works as expected:

| ctx | Phase 2 (FSDP1) | Phase 3a (FSDP2) |
|---|---|---|
| 32K | ~80 GiB/card (no shard) → OOM | 28 GiB/card steady |
| 65K | OOM (76 GiB before activations) | 59 GiB/card |

## Phase 3b prerequisites

To make Stage A meaningful (100M+ tokens of real CPT, not 3-step smoke), and to land 256K+ context:

1. **Stage A main run at 128K** is doable on the existing stack — primary bottleneck will be runtime, not memory. Estimated: ~26 GiB packed jsonl tokens / 4M tokens-per-step (4 cards × 128K × 1 micro × 8 grad-accum) ≈ ~7 optimizer steps per epoch on current dataset; need at minimum 1 full pass + repeat to log a non-trivial loss curve. With our 30-sequence corpus we should ramp grad_accum and epoch count instead.
2. **256K and beyond requires sequence parallelism**. Options:
   - **DeepSpeed Ulysses SP** — LF supports it via `sequence_parallel_size`. Most direct fit.
   - **Ring-Attention** — requires custom wiring; not ready in LF as of 0.9.5.dev0.
   - **More cards**: cards 4-7 are off-limits per user constraint.
3. **Grow the training corpus**. 30 × 256K seqs (6.6M tokens) is too small for a CPT pass. Phase 3b ingest needs to push to ~100-250M tokens (more arXiv + PG-19 + the QA-style supplements brainstormed in `ttt.md`).
4. **Ckpt save strategy at high ctx**: FULL_STATE_DICT gather is fine through 128K. For 256K+ even with SP, expect to switch to `SHARDED_STATE_DICT` and convert to FULL pre-eval.

## Quality gate carry-overs into Phase 3b

From `phase1-baseline` (unchanged):
- **RULER 128K vt: 10-20% → ≥ 60%** (the headline metric for actual CPT signal).
- RULER 128K overall: 82-84% → ≥ 92%.
- NIAH 128K stays at 100% (already met).

New Phase-3b specific:
- 256K ctx must work end-to-end (training + vLLM eval). Phase 3a stops at 128K; the 256K leap needs SP.
- Stage A main: minimum 100M tokens trained at 128K (full plan target 250M).

## Repo state at tag

- 49 commits on `main` (Phase 3a added 12).
- 35 tests passing (`uv run pytest tests/ -q`).
- ruff clean across source.
- Tag: `phase3a-lf-smoke`.
- Smoke checkpoints kept locally: `checkpoints/phase3a_smoke_{32k,64k,128k}/` (256K dir empty — OOM never produced a ckpt).
