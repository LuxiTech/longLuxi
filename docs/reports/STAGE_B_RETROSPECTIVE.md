# Stage B Retrospective — 128K CPT v3 data, lower LR, partial eval

| Field | Value |
|---|---|
| Date | 2026-05-20 |
| Training | LF + accelerate FSDP2, bf16, Liger fused CE, grad ckpt |
| Ctx | 128K dense (4× H100, cards 0-3) |
| Data | `packed_128k_v3.jsonl` — 1690 seqs × 128K = 192M tokens; v3 synthetic with COT traces + multi-target chains |
| LR / sched | 1e-5 cosine, warmup 0.05 (vs Stage A 3e-5) |
| Steps | 250 (vs Stage A 500) |
| Outcome | SIGKILL at step 249/250; checkpoint-200 is the latest intact artifact |
| Eval | Partial — NIAH 1M/1.5M/2M + RULER 1M/1.5M completed; RULER 2M + LongBench-v2 long aborted by user |

## What Stage B was designed to fix

| Stage A problem | Stage B change |
|---|---|
| YaRN extrapolation broken at 1.5M-2M (NIAH 2M 100%→80%) | LR 3e-5 → 1e-5 (less weight drift) |
| niah_multiquery -50pp (single-answer bias) | v3 data multi-target chains (alpha/beta/gamma/delta per doc) |
| vt no improvement (model copied "Answer: X" without traversing chain) | v3 data with explicit `<trace>Step 1: v0→v1. ...</trace>` blocks |
| Stage A 500 steps too many | 250 steps |

## Headline result: extrapolation preserved, instruction-following degraded

**Win (vs Stage A):**
- NIAH 1M / 1.5M / **2M = 100% / 100% / 100%** (vs Stage A 100% / 90% / 80%). **Lower LR fully preserved YaRN extrapolation.**
- RULER niah_multiquery @ 1M improved 0.50 → 0.60 (v3 multi-target helped)

**Loss:**
- RULER overall WORSE than Stage A: 1M 0.70 (vs 0.72), 1.5M 0.46 (vs 0.56)
- RULER niah_single @ 1M: 1.00 → 0.80 (Stage A held this at 1.00) — Stage B's weight drift was less severe on far-extrapolation but slightly damaged some retrieval at 1M
- RULER qa_1 @ 1.5M: 0.60 (base) → 0.20 (Stage A) → 0.30 (Stage B) — improvement over A, still far below base
- **RULER vt: still 0.00-0.10** — COT trace + multi-target design **did NOT lift vt**

## Detailed metrics (Stage B vs Stage A vs base @ factor=8)

### NIAH

| ctx | base | Stage A | Stage B |
|---|---|---|---|
| 1M | 1.00 | 1.00 | **1.00** ✅ |
| 1.5M | 1.00 | 0.90 | **1.00** ✅ (fully recovered) |
| 2M | 1.00 | 0.80 | **1.00** ✅ (fully recovered, +0.20 vs Stage A) |

### RULER

| ctx | task | base | Stage A | Stage B |
|---|---|---|---|---|
| 1M | overall | 0.82 | 0.72 | **0.70** |
| 1M | niah_single | 1.00 | 1.00 | **0.80** ↓ |
| 1M | niah_multikey | 1.00 | 1.00 | 1.00 |
| 1M | niah_multiquery | 1.00 | 0.50 | **0.60** ↑ |
| 1M | vt | 0.10 | 0.10 | **0.10** (no change) |
| 1M | qa_1 | 1.00 | 1.00 | 1.00 |
| 1.5M | overall | 0.72 | 0.56 | **0.46** |
| 1.5M | niah_single | 1.00 | 1.00 | 1.00 |
| 1.5M | niah_multikey | 0.90 | 1.00 | 0.90 |
| 1.5M | niah_multiquery | 1.00 | 0.40 | **0.10** ↓ |
| 1.5M | vt | 0.10 | 0.20 | **0.00** ↓ |
| 1.5M | qa_1 | 0.60 | 0.20 | **0.30** ↑ |
| 2M | (not run) | — | 0.48 | (aborted) |

## Root cause analysis: why Stage B still failed on RULER

Both Stage A and B confirm the same pattern: **PT loss on synthetic Q+A data does NOT teach the model to execute the algorithm.** The model learns next-token statistics of the doc; when the test prompt ends at "Question: ... Answer:", the model is supposed to GENERATE the trace + answer, but at inference time it's only conditioned on the question, not on the position-in-document patterns it learned during training.

Specifically for v3 vt_chain data:
- Training: model sees `Question: ... <trace>Step 1: v0→v1. Step 2: ...</trace> Answer: 42` and learns to continue trace + emit answer
- Inference (RULER vt): model sees `Question: ... Answer with just the number.` — different prompt suffix
- The chain-following BEHAVIOR didn't transfer because pretrain-loss doesn't reward "match this specific question format"

In other words: **PT on Q+A data teaches the model the JOINT distribution, not the conditional `p(answer | question)`**. SFT is the only way to learn that conditional.

## Conclusion: PT alone cannot improve reasoning at long ctx

Both Stages confirm the **reasoning bottleneck is not solvable by more long-ctx training**. Qwen3.5-4B's reasoning capability ceiling (RULER vt ~10-30%) is set by its base pretraining. The next stage to actually move vt is **SFT with reasoning data** (loss-masked to answer tokens, format-matched to evaluation prompts).

## Stage B as a deliverable

Despite failing to improve RULER, **Stage B's ckpt is the cleanest "2M-capable" training artifact** because:
- NIAH 1M/1.5M/2M = 100% (no regression from base)
- Less weight drift than Stage A (lower LR)
- LongBench-v2 long eval was not run (interrupted) — would expect similar +7pp gain to Stage A given identical real-doc training mix

## Repo state

- Ckpt: `checkpoints/stage_b_v3_128k/checkpoint-200/` (20 GB, complete safetensors)
- Eval: `eval/results/stage_b_v3_128k_{niah_1m,niah_1500k,niah_2m,ruler_1m,ruler_1500k}/`
- Training config: `configs/training/stage_b_v3_128k_fsdp2.yaml`
- Data: `data/processed/v3/packed_128k_v3.jsonl` (1690 seqs, 192M tokens)
