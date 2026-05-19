# Qwen3.5-4B + YaRN Factor=4 — Definitive 1M Long-Context Capability Report

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Model | `Qwen/Qwen3.5-4B` (local `/home/user01/Minko/models/Qwen3.5-4B/`) with YaRN factor=4 baked in (max_pos 1010000) |
| Inference | vLLM 0.21.0 TP=4 on H100 cards 0-3, bf16, FlashInfer, chunked prefill |
| Coverage | NIAH 32K-1M, RULER 32K-1M, LongBench-v2 long subset, InfiniteBench longbook_choice + leakage control |
| Wall-time | ~2 h synthetic + ~2.5 h real-doc total |

## TL;DR

**Qwen3.5-4B + YaRN supports 1M positionally and retrieves well at 1M, but cannot reliably reason over the retrieved content.** Real long-doc QA performance (LongBench-v2 long) is at chance (25%). Book-based QA shows 80% but the leakage-control reveals ~45 pp of that comes from actually reading the context — the other ~35 pp is pretraining memorization. **The capability gap is reasoning, not retrieval, and CPT should target reasoning data, not more long-context tokens.**

## Headline numbers

| Test | n | accuracy | random baseline | Δ vs random |
|---|---|---|---|---|
| **NIAH 1M** | 10 | **100%** | 0% (open-ended) | high signal |
| **RULER 1M overall** | 50 | **86%** | n/a (4 task-100% + 1 vt-30% / 5) | high on retrieval, low on vt |
| **RULER 1M vt only** | 10 | **30%** | varies by chain length (~12-25%) | weak |
| **LongBench-v2 long** | 71 (28 skipped) | **25.4%** | 25% (4-choice) | **at chance** |
| **InfBench longbook_choice (w/ ctx)** | 225 (4 skipped) | **80.4%** | 25% | strong |
| **InfBench longbook_choice (NO ctx)** | 229 | **35.4%** | 25% | **leakage signal** |
| InfBench longbook context-gain | — | **+45 pp** | — | real long-ctx contribution |

## Detail by capability axis

### 1. Position handling & basic retrieval — ✅ Strong

| ctx | NIAH | RULER niah_s | RULER niah_multikey | RULER niah_multiquery |
|---|---|---|---|---|
| 32K | 100% | 1.00 | 1.00 | 1.00 |
| 128K | 100% | 1.00 | 1.00 | 1.00 |
| 256K | 100% | 1.00 | 1.00 | 0.90 |
| 512K | 100% | 1.00 | 1.00 | 1.00 |
| 1M | 100% | 1.00 | 1.00 | 1.00 |

Conclusion: **YaRN factor=4 works.** The model maintains 100% single- and multi-needle retrieval at 1M tokens. No length-dependent degradation observed.

### 2. Synthetic single-fact QA (RULER qa_1) — ✅ Strong

Constant 100% from 32K to 1M.

### 3. Synthetic multi-hop reasoning (RULER vt) — ❌ Weak across ALL lengths

| ctx | vt accuracy |
|---|---|
| 32K | 0.10 |
| 128K | 0.10 |
| 256K | 0.00 |
| 512K | 0.10 |
| 1M | 0.30 |

`vt` asks: `v0=v1; v1=v2; v2=42; what is v0?` — a 3-5 hop variable chain. n=10/length → significant noise, but the qualitative picture is constant: **the model cannot reliably trace assignment chains, and ctx length is not the cause.** Even at 32K (well within the original Qwen3.5-4B 256K base ctx, no YaRN scaling) it scores 10%.

### 4. Real long-doc multi-domain QA — ❌ At chance

LongBench-v2 `long` subset (all >128K, mean ~500K real tokens):

| domain | n | accuracy |
|---|---|---|
| Single-Document QA | 23 | **0.087** (worse than random!) |
| Multi-Document QA | 21 | 0.238 |
| Long In-context Learning | 18 | 0.278 |
| Code Repository Understanding | 7 | 0.571 (small n) |
| Long Structured Data Understanding | 2 | 1.000 (tiny n) |
| **overall** | **71** | **0.254** |

LongBench-v2 is adversarially constructed: each question is verified to require multi-hop reasoning over the document. **The base model is unable to do this at any meaningful rate.** 28/108 samples (26%) were skipped because their contexts exceeded our 800K-token filter — these are samples where YaRN ratio=4 is unable to fit, not a capability limit per se.

### 5. Book-based QA with provided context — 🟡 Mixed (real signal + leakage)

InfiniteBench `longbook_choice_eng`, full 225 of 229 samples (4 skipped for length):

| ctx bucket (tokens, approx) | n | accuracy |
|---|---|---|
| <50K | 0 (all samples ≥ 50K) | — |
| 50K-180K | 131 | **0.817** |
| 180K-500K | 86 | **0.767** |
| >500K | 8 | 1.000 (tiny n) |
| **overall** | **225** | **0.804** |

**Leakage control** (same 229 questions, NO context provided): **35.4%**.

Interpretation:
- ~35 pp baseline is memorization: these books (Adam Bede, Aeneid, etc.) are in pretraining; the model knows characters and plots without re-reading.
- ~45 pp additional comes from actually reading the supplied context. That **IS** real long-context reading comprehension.
- Mild ~5 pp drop from 50-180K to 180-500K bucket suggests context utilization degrades modestly at length, but the >500K bucket is too small (n=8) to draw conclusions.

### 6. Implicit position interpolation correctness (PG-19 style PPL) — NOT MEASURED

We did not run a PPL baseline on real long documents. This is the missing piece — should be added pre/post CPT to detect language-modeling regressions.

## Synthesis

**What works at 1M:**
- Position handling (no rotational degeneration up to 1M)
- Single-needle and few-needle retrieval (RULER niah variants, NIAH)
- Single-fact QA grounded in retrieved span (RULER qa_1)
- Reading-grounded answer extraction on familiar text (InfBench longbook +45pp context gain)

**What does NOT work at 1M (or anywhere, since the bottleneck is reasoning not ctx):**
- Multi-hop variable tracking (RULER vt: 10-30%)
- Adversarial real-doc QA (LongBench-v2: 25%, at chance)
- Single-doc QA when adversarially constructed (LongBench-v2 single-doc: 8.7%)

**Inflated by pretraining memorization:**
- InfBench longbook_choice 80% surface — but 35 pp of that is just "model has read the book in pretraining". Be careful using this benchmark as a CPT target without context-removal controls.

## What this means for Phase 3b/3c training

**Original framing (revised):**
- ~~"Extend ctx from 1M to 2M+"~~ → Already supports 1M positionally with 100% retrieval. Pushing to 2M is a separate research project (YaRN re-fit). Not the immediate priority.
- ~~"Stage A: 250M tokens of arXiv + PG-19 to make 1M usable"~~ → The model can already attend to 1M and pull single answers out. More long-text data without targeted reasoning supervision will improve PPL but not move LongBench-v2 or vt.

**Revised priorities:**

1. **Synthetic reasoning data is the lever** — variable chains, multi-hop deductions, code execution traces, math derivations. The brainstorm `ttt.md` Section 5 ("high-level long-context tasks") is now the primary Phase 3b data plan, not a stretch goal.

2. **Real long-doc Q&A SFT mix** — use real long-doc QA datasets (Qasper, NarrativeQA, MuSiQue) as part of Stage D SFT. LongBench-v2 25% means there's enormous room to grow even at the 256K range.

3. **Length is a sufficient context, not a goal in itself.** Train at 128K-256K (which we can do with FSDP2 on 4 H100s as proven in Phase 3a) and verify generalization to 1M via eval. Don't burn tokens at 512K-1M training contexts without a reason.

4. **Quality gates revised:**
   - LongBench-v2 long: 25% → **≥ 45%** (move off chance toward weak reasoning)
   - RULER vt @ 1M: 30% → **≥ 60%**
   - InfBench longbook context gain: 45 pp → **≥ 55 pp** (i.e. more answers driven by reading, not memorization)
   - NIAH 1M and RULER non-vt: must stay at current levels (no regression)
   - LongBench-v2 should be the **headline overall metric** going forward.

## What we did NOT measure yet (Phase 4 work)

- **Real long-doc PPL on PG-19 holdout** — language-modeling quality at 32K/128K/512K/1M. Needed pre/post-CPT.
- **Open-ended long-doc QA** (Qasper, NarrativeQA, MuSiQue) — F1/ROUGE scores on real benchmarks beyond multiple choice.
- **Long-doc summarization** — coherence/faithfulness of multi-paragraph generation at long ctx.
- **Long code reasoning** (InfBench code_run, code_debug) — these subsets exist but were not run tonight.
- **Math at long ctx** (InfBench math_find).
- **InfBench kv_retrieval, passkey** — multi-key precision at ~134K-200K (RULER multikey covers similar ground; would triangulate).
- **YaRN-2M re-fit + 2M benchmarks** — explicitly out of scope; defer to Phase 4.

## Repo state

- 9 result dirs added: `eval/results/base_{ruler,niah,longbench_v2_long,infbench_longbook_choice,infbench_longbook_choice_NOCTX}`
- New runners (real implementations replacing stubs): `eval/runners/run_longbench_v2.py`, `eval/runners/run_infinitebench.py`, `eval/runners/_no_context_control.py`
- Batch helper: `scripts/run_base_1m_benchmarks.sh`
