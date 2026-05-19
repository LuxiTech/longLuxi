# UltraLong-8B — methodology distilled for longluxi

> Paper: "From 128K to 4M: Efficient Training of Ultra-Long Context Large Language Models" (UIUC + NVIDIA, arXiv:2504.06214, 2025-04). Released models: `nvidia/Llama-3.1-Nemotron-8B-UltraLong-{1M,2M,4M}-Instruct` on HuggingFace.

## Their recipe in one page

**Base**: Llama-3.1-8B-Instruct (128K via prior YaRN; RoPE base 5×10⁵, initial s=8).

**Two stages**:

1. **Continued pretraining (CPT)** — direct at target ctx, no curriculum.
2. **SFT** — short-ctx only (<8K), 100K instances, general/math/code blend.

### RoPE scaling factors

| Variant | YaRN `factor` (s) | RoPE base | Notes |
|---|---|---|---|
| 1M | 128 | unchanged from Llama base | 8K → 1M extension factor 128 |
| 2M | 256 | unchanged | 8K → 2M extension factor 256 |
| 4M | 512 | unchanged | 8K → 4M extension factor 512 |
| YaRN α / β | 1 / 4 | fixed | "alpha=1, beta=4" — these are YaRN's interpolation/extrapolation knobs |

Map to Qwen3.5-4B (256K native, YaRN factor=4 → 1M already shipped):
- **2M target** = factor=8 (extension from 256K base, ratio 8)
- **4M target** = factor=16

### CPT data

- **1B tokens, 1 epoch, no curriculum** (direct training at full target ctx).
- Source: "per-source upsampled pretraining corpus following Fu et al. 2024" (basically the LongRoPE / Together-CodeLLama style pretrain mix, biased toward long docs).
- Length-aware upsample: docs <4K downsampled, docs >8K upsampled (so the 1B tokens carries more long-doc representation than naive sampling).
- **Document separator: special characters** (NOT BOS/EOS tokens like `<|begin_of_text|>`). Their ablation: using BOS hurts by ~1 pt on RULER.
- **Cross-document attention IS NOT masked** (i.e., docs in one packed sequence attend to each other across the separator). They show this is BETTER than masking.

### CPT training

- Optimizer: AdamW (β1=0.9, β2=0.95)
- LR: **3e-5** (single LR, no warmup explicitly mentioned in the paper)
- Batch: not explicitly specified — likely 1-4 sequences per GPU at 1M, 1-2 at 2M/4M
- Hardware: 256 × H100, TP=8 + context parallelism (CP=4 for 1M, **CP=16 for 2M/4M**) under Megatron-LM
- Wall time per variant: 5h (1M), 6h (2M), 13h (4M)

### SFT data

100K instances, all <8K tokens, three domains:
- General: ShareGPT, SlimOrca, EvolInstruct, GPTeacher, AlpacaGPT4, UltraInteract
- Math: OrcaMathWordProblems, MathInstruct, MetaMath
- Code: Magicoder, WizardCoder, GlaiveCodeAssistant

Filtered + decontaminated using GPT-4o/4o-mini.

### SFT training

- LR 5e-6, batch 128, AdamW (β1=0.9, β2=0.95), ~30 min on 256 H100s.

## Their numbers

| benchmark | 1M variant | 2M | 4M | Llama-3.1-8B baseline |
|---|---|---|---|---|
| NIAH (1M/2M/4M depending on variant) | 100% | 100% | 100% | degrades past 128K |
| RULER (<1M) | 79.1 | 78.2 | 78.0 | ~80 |
| LV-Eval (<256K) | 27.0 | 28.9 | 28.1 | — |
| InfiniteBench | 32.1 | 32.5 | 30.4 | — |
| MMLU/Math/Code avg | 62.5 | 61.1 | 61.0 | 61.5 |

**Read**: short benchmarks don't regress (61-62 vs 61.5 baseline). Long benchmarks: NIAH 100% achieved; RULER/LV-Eval stay within 1 pt of base (Llama-3.1-8B at long ctx is itself ~80 on RULER, so they didn't IMPROVE — they EXTENDED).

## Lessons applicable to longluxi (Qwen3.5-4B → 2M target)

1. **Direct training at target ctx works** — they don't ramp from 128K to 1M to 4M. We can train directly at 2M if we have the infra.
2. **1B tokens is enough for ctx extension** if you direct-train. Our current 6.6M packed corpus is ~150× too small. **Need to scale data pipeline first.**
3. **YaRN factor change alone (no CPT) probably won't work at 2M**. Their paper explicitly motivates CPT as fixing the YaRN extrapolation artifacts. Test base @ 2M to confirm this matches our setup.
4. **Cross-doc attention should NOT be masked** for our CPT — our packer's `mask_doc_boundaries` field should be left unused or used only for analytics.
5. **Document separator: keep `<doc id="X" source="Y">` tag** (our current format). NOT BOS/EOS.
6. **Training infra**: their CP=16 implies 16-way sequence parallelism. On 4 H100s, CP=4 is the most we can do (one card per CP rank, no TP). LF supports DS Ulysses; needs to be wired and validated.
7. **NIAH at 2M is achievable** with 1B tokens CPT — but RULER stays at ~80. **They didn't improve reasoning at long ctx either**. Matches our finding that Qwen3.5-4B's bottleneck is reasoning, not ctx.

## Open questions for our setup

- **YaRN factor=8 (vs UltraLong's =256)**: Qwen3.5 starts from 256K native (Llama starts from 8K). So our "8x extension to 2M" is smaller than UltraLong's "128x extension to 1M". The model might extrapolate better; needs measurement.
- **Does Qwen3.5-4B + YaRN factor=8 without CPT already give usable 2M NIAH?** If yes, our CPT scope shrinks.
- **What's our minimum viable CPT data scale?** Their 1B was empirically chosen. We may get away with 100M for "first signs of 2M usability".
- **Activation memory at 2M dense even with CP=4**: 2M tokens / 4 ranks = 500K per rank. With grad ckpt that's roughly our 128K dense memory. Should fit on 4 H100s with CP=4. But this is untested for our model.
