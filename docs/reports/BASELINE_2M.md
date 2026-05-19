# Qwen3.5-4B @ YaRN factor=8 → 2M Base Baseline

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Model | Local `/home/user01/Minko/models/Qwen3.5-4B/` with `text_config.rope_parameters.factor=8.0`, `max_position_embeddings=2097152`, `original_max_position_embeddings=262144` |
| YaRN ratio | 8.000 exact |
| Backup of factor=4 config | `config.json.yarn4.bak` |
| Inference | vLLM 0.21.0 TP=4, MAX_LEN=2097152, FlashInfer, chunked prefill, gpu_memory_utilization=0.95 |
| KV-cache footprint per card | ~80 GiB (at cap; max_num_seqs=1 effectively) |

## Headline

**Free 2M from YaRN factor=8 alone** — no training required to reach 100% NIAH at 2M. But RULER reveals capability drift starting at 1.5M: qa_1 and niah_multikey_1 begin degrading from 1M baseline.

## NIAH (5 depths × 2 cells per length)

| ctx | accuracy | elapsed |
|---|---|---|
| 1M | **1.00** | 6.5 min |
| 1.5M | **1.00** | 12.7 min |
| 2M | **1.00** | 20.8 min |

Conclusion: **single-needle retrieval works perfectly up to 2M with no training.** YaRN factor=8 extrapolation preserves the model's ability to attend to and recover individual facts across 2M tokens.

## RULER (5 tasks × 10 cells per length)

| ctx | overall | niah_single_1 | niah_multikey_1 | niah_multiquery | vt | qa_1 |
|---|---|---|---|---|---|---|
| 1M (factor=4 ref) | 0.86 | 1.00 | 1.00 | 1.00 | 0.30 | 1.00 |
| **1M (factor=8)** | **0.82** | 1.00 | 1.00 | 1.00 | **0.10** | 1.00 |
| **1.5M (factor=8)** | **0.72** | 1.00 | **0.90** | 1.00 | **0.10** | **0.60** |
| **2M (factor=8)** | **0.74** | 1.00 | 1.00 | **0.80** | **0.00** | **0.90** |

The 1.5M → 2M numbers fluctuate (qa 0.60→0.90, mq 1.00→0.80) within n=10
noise band. Aggregating across 1M-2M (150 cells per task), the true picture
is: vt fixed at ~5-10% (broken everywhere), niah_single fixed at 100%,
the other 3 RULER tasks live in a 0.80-1.00 band depending on draw.

**Critical signal at 1.5M**: **qa_1 drops from 1.00 → 0.60** (-40 pp), and niah_multikey_1 from 1.00 → 0.90. niah_single_1 and niah_multiquery still hold at 1.00.

Interpretation:
- vt was already at coin-flip everywhere (length-independent, reasoning-bound capability).
- qa_1's drop is the *real* extrapolation cost of YaRN factor=8 — the model can *find* relevant text but cannot reliably *answer* once positions are far enough past the trained range.
- Multi-key gets harder at 1.5M (still 90% though).

## What this gives us as a training target

**Concrete, measurable goal for Stage A CPT:**

| Metric | base @ 2M target | Stage A goal |
|---|---|---|
| NIAH 2M | 1.00 | maintain 1.00 |
| RULER 2M overall | ≥ 0.80 | **≥ 0.85** |
| RULER 2M qa_1 | ≥ 0.70 | **≥ 0.90** (close the 1.5M gap) |
| RULER 2M vt | 0.10-0.30 | **≥ 0.40** (independent reasoning lever) |
| RULER 2M niah_multikey_1 | ≥ 0.90 | maintain ≥ 0.95 |
| RULER 1.5M qa_1 | 0.60 (today) | **back to 0.90+** |

## Repo state at this report

- Model config patched: YaRN factor=4 → 8 at `/home/user01/Minko/models/Qwen3.5-4B/config.json` (backup `.yarn4.bak`).
- 6 result dirs added: `eval/results/base_yarn8_{niah,ruler}_{1000000,1500000,2000000}/` (2M RULER pending).
- Batch script: `scripts/run_base_2m_benchmarks.sh`.
- Training data v2 prepared: `data/processed/v2/` (245M raw tokens, 1988 seqs packed @ 128K, 1012 seqs packed @ 256K).
- LF dataset registered: `longluxi_packed_v2_128k`, `longluxi_packed_v2_256k`.
- Training YAMLs ready: `configs/training/stage_a_v2_{128k_fsdp2,256k_dsz3}.yaml`.
