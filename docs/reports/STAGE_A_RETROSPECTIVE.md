# Stage A Retrospective — 128K CPT, Mixed Results

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Training | LF v0 + accelerate FSDP2, bf16, Liger fused CE, gradient checkpointing |
| Ctx | 128K dense (4× H100, cards 0-3) |
| Data | `packed_128k_v2.jsonl` — 1988 seqs × 128K = 228M tokens; mix: arxiv 24% + pg19 33% + synthetic vt_chain 23% + entity_kv 11% + code_trace 11% |
| LR / sched | 3e-5 cosine, warmup 0.05 |
| Steps | 500, ~1 epoch |
| Wall time | 4h42m |
| train_loss | 1.77 → 1.43 (-19%) |
| Output ckpt | `checkpoints/stage_a_v2_128k/checkpoint-500/` |

## Headline result

**Net loss for the explicit 2M-capability target.** The hypothesis that "training reasoning at 128K extrapolates to 2M via YaRN factor=8" failed: training disrupts YaRN extrapolation at 1.5M+. NIAH 2M dropped from 100% to 80%, RULER 2M overall from 0.74 to 0.48.

**One bright spot**: LongBench-v2 long (real-doc QA) improved from random (0.254) to above-random (0.325, +7pp). The long-doc CPT signal helped real comprehension, even though it didn't transfer to synthetic RULER.

## Detailed metrics — Stage A vs base @ YaRN factor=8

### NIAH

| ctx | base | Stage A | Δ |
|---|---|---|---|
| 1M | 1.00 | **1.00** | 0 |
| 1.5M | 1.00 | **0.90** | -0.10 |
| 2M | 1.00 | **0.80** | -0.20 |

The further from the training context (128K), the more degradation. Classic YaRN-extrapolation disruption.

### RULER

| ctx | overall | niah_single | niah_multikey | niah_multiquery | vt | qa_1 |
|---|---|---|---|---|---|---|
| 1M base | 0.82 | 1.00 | 1.00 | 1.00 | 0.10 | 1.00 |
| 1M Stage A | **0.72** (-10) | 1.00 | 1.00 | **0.50** (-50) | 0.10 (0) | 1.00 |
| 1.5M base | 0.72 | 1.00 | 0.90 | 1.00 | 0.10 | 0.60 |
| 1.5M Stage A | **0.56** (-16) | 1.00 | 1.00 (+10) | **0.40** (-60) | 0.20 (+10) | **0.20** (-40) |
| 2M base | 0.74 | 1.00 | 1.00 | 0.80 | 0.00 | 0.90 |
| 2M Stage A | **0.48** (-26) | **0.70** (-30) | **0.70** (-30) | **0.30** (-50) | 0.00 (0) | 0.70 (-20) |

### LongBench-v2 long (real adversarial multi-doc QA, n=80 after filter)

| domain | base | Stage A | Δ |
|---|---|---|---|
| Long In-context Learning | 0.28 | 0.36 | **+0.08** |
| Multi-Document QA | 0.24 | 0.24 | 0 |
| Code Repository Understanding | 0.57 | 0.50 | -0.07 |
| **Single-Document QA** | **0.087** | **0.26** | **+0.17** |
| Long Structured Data (n=3, noise) | 1.00 | 0.67 | -0.33 |
| **overall** | **0.254** | **0.325** | **+0.07** |

## Root cause analysis

### 1. Synthetic vt_chain didn't lift RULER vt

Hypothesis: Training data answer was IN-DOCUMENT (`"Answer: After following the chain, the final numeric value is 98696."`) — the model learned to predict the answer token following "is" without learning the chain-resolution algorithm. The chain-following capability never developed because there was no incentive to develop it.

**Fix for v2 synthetic data**:
- Move answer OUT of document — use SFT-style separate target with loss only on answer span.
- OR: use multi-question doc (same chain, multiple questions about different links), so the model can't memorize one answer position.
- OR: emit the chain in REVERSE order (`v8 = 98696. v7 maps to v8. v6 maps to v7. ...`) and ask for v0 — forces the model to actually follow.

### 2. niah_multiquery collapsed across all lengths

RULER niah_multiquery has 4 needles (alpha/beta/gamma/delta) and asks for ONE specific needle. Our synthetic data trained the model heavily on "find the ONE numerical answer". Distribution mismatch → model picks wrong needle 50-70% of the time.

**Fix**: include multi-target synthetic data (`"What is the alpha value?" → distinct from beta/gamma/delta`).

### 3. 128K training disrupts YaRN extrapolation at 1.5M-2M

The model's attention pattern at positions 128K-2M was "frozen" from Qwen3.5-4B's original training. After CPT at 128K, those parameters shifted slightly, changing the attention behavior at far-extrapolated positions in unpredictable ways. NIAH 1M held (1M is YaRN-trained); 1.5M-2M degraded.

**Fix**: Train at or near target ctx (UltraLong's recipe). With 4 H100s, this requires DS ZeRO-3 + Ulysses CP=4 (LF V1 mode supports this).

### 4. The one win was real-doc QA

arxiv + pg19 long-doc training improved LongBench-v2 long by 7pp overall, with Single-Doc QA jumping from 8.7% → 26%. **The real long-doc content is genuinely useful** — just not at 128K alone for 2M extrapolation.

## What to do for Stage B

### Must-fix
1. **Direct training at higher ctx** (≥ 256K, ideally 1M-2M) using DS Z3 + Ulysses CP=4 via LF V1 mode. This is the only way to avoid YaRN extrapolation disruption.
2. **Redesign synthetic data** so model can't memorize next-token-after-question. Three concrete patterns:
   - **chain-reversed**: `v8=42. v7→v8. v6→v7. … v0→v1.` then ask `what is v0?` — forces actual traversal because the literal "42 is the value of v0" never appears.
   - **multi-target**: insert 4 chains in one doc, ask about a random one. Forces selective traversal.
   - **mid-doc QA**: Q+A pair embedded INSIDE the haystack at random position; the natural-language continuation explains. Tests model's "reading comprehension" rather than "copy after question".
3. **Distribute multi-needle properly**: half the synthetic data should have multiple distinct targets (like niah_multiquery's alpha/beta/gamma/delta).

### Could-do
4. Lower LR (1e-5 instead of 3e-5) to reduce drift from base weights.
5. Mix in some short-context data (~10% < 32K) to preserve short-ctx capabilities — UltraLong did this for SFT but we could include in CPT.

### Stage A → Stage B decision

**Discard Stage A ckpt for the 2M goal**: it's a net loss on the headline metrics (NIAH 2M, RULER 2M). 

**Keep Stage A as a data point**: it proves the "128K training extrapolates to 2M" hypothesis is FALSE for Qwen3.5-4B. Saves us from re-running this experiment.

**Save the real-doc QA improvement separately**: LongBench-v2 +7pp is genuine. If we end up needing a "best LongBench-v2" model rather than "best 2M synthetic", this ckpt is still useful.

## Stage B sketch

1. **Patch LF to V1 mode** with `dist_config.cp_mode: ulysses, cp_size: 4` (see `external/LLaMA-Factory/examples/v1/train_full/train_full_ulysses_cp.yaml`)
2. **Pack data at 2M** (`pack_v2.py --target-ctx 2097152` — currently we have packed_256k_v2 but not 2M)
3. **Redesign synthetic data** per fixes above
4. **Smoke test** DS Z3 + Ulysses CP=4 at 1M (cheaper than 2M) to validate stack
5. **Stage B run** at 2M dense with corrected data, ~50M tokens budget initially

## Repo state at this report

- Stage A ckpt: `checkpoints/stage_a_v2_128k/checkpoint-500/` (kept; not yet deleted)
- Eval results: `eval/results/stage_a_v2_128k_*/` (9 dirs: 3 NIAH + 3 RULER + 1 LongBench-v2)
- Training script: `configs/training/stage_a_v2_128k_fsdp2.yaml`
- Auto-eval pipeline: `scripts/post_stage_a_eval.sh`
- Data: 246M raw tokens, packed at 128K and 256K (256K not used in Stage A)
