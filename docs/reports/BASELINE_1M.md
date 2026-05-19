# Qwen3.5-4B Full Long-Context Baseline (32K → 1M)

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Tag | (untagged; supersedes incomplete `BASELINE_W1.md` from Phase 1) |
| Model | `Qwen/Qwen3.5-4B` (local copy: `/home/user01/Minko/models/Qwen3.5-4B/`) with YaRN factor=4 in `config.json` (`max_position_embeddings=1010000`, `rope_scaling.rope_type=yarn`) |
| Inference | vLLM 0.21.0 TP=4 on cards 0-3, bf16, FlashInfer attention, chunked prefill |
| Eval framework | `eval/runners/run_via_vllm.py` (chat completions, `enable_thinking: false`) |
| n_per_task (RULER) | 10 cells × 5 tasks per length |
| n_per_cell (NIAH) | 2 × 5 depths {10,30,50,70,90}% per length |
| Wall time | 86 min total (32K → 1M, batch script `scripts/run_base_1m_benchmarks.sh`) |

## RULER — by length × by task

| ctx | overall | niah_single_1 | niah_multikey_1 | niah_multiquery | **vt** | qa_1 |
|------|---------|---------------|------------------|-------------------|---------|--------|
| 32K  | 0.82    | 1.00          | 1.00             | 1.00              | **0.10** | 1.00   |
| 128K | 0.82    | 1.00          | 1.00             | 1.00              | **0.10** | 1.00   |
| 256K | 0.78    | 1.00          | 1.00             | 0.90              | **0.00** | 1.00   |
| 512K | 0.82    | 1.00          | 1.00             | 1.00              | **0.10** | 1.00   |
| 1M   | 0.86    | 1.00          | 1.00             | 1.00              | **0.30** | 1.00   |

Wall time per RULER length: 32K=5.5min, 128K=7.3min, 256K=10.5min, 512K=18.5min, 1M=40min.

## NIAH — single needle, 5 depths × 2/cell

| ctx | accuracy |
|------|----------|
| 256K | 1.00 |
| 512K | 1.00 |
| 1M   | 1.00 (measured during Phase 2 via vLLM at `--lengths 1000000`) |

NIAH 32K/128K/512K confirmed 100% at multiple points in Phase 1/2/3a. We did not re-run 32K/128K NIAH in this batch since it was redundant.

## What this tells us

1. **Length is not the bottleneck for base Qwen3.5-4B.** Single-needle retrieval (NIAH) is 100% all the way to 1M. The four "retrieval-like" RULER tasks (niah_single_1, niah_multikey_1, niah_multiquery, qa_1) are also at 100% across 32K → 1M (with one 90% sample at 256K, within n=10 noise).
2. **vt (variable tracking) is the only real gap, and it is length-independent.** The base model scores 0-30% across all lengths, hovering at ~10-15% true mean. n=10 makes the per-length number noisy, but the qualitative picture is constant: **the model cannot reliably resolve a multi-hop assignment chain (`v0=v1; v1=v2; v2=42; what is v0?`)**, regardless of ctx.
3. **RULER overall = 82% baseline** comes almost entirely from the 4 non-vt tasks averaging 100% diluted by vt at ~10%. (4×1.00 + 1×0.10) / 5 = 0.82. ✓
4. **Phase 1's published `BASELINE_W1` figure** of "RULER 128K overall 84%, vt 20%" is the same population as 128K here (within n=10 noise). The discrepancy is sampling, not a regression.

## Implication for Phase 3b/3c training

| Decision | Conclusion |
|---|---|
| **Do we need length extension training at all?** | NO for retrieval/QA — base + YaRN factor=4 already handles 1M cleanly for these tasks. |
| **What should CPT target?** | **vt-style reasoning**, not raw long-ctx. The fix is data composition (more variable-tracking, multi-hop assignment, code reasoning) not context-length extrapolation training. |
| **Headline gate for any post-CPT eval** | **RULER vt @ 1M ≥ 60%** AND no regression on the other 4 tasks (each must stay ≥ 0.95). Overall ≥ 92%. |
| **What about 2M+?** | YaRN ratio=4 maxes the trained 256K to 1M. To exceed 1M we'd need a YaRN re-fit (training a higher factor) OR position interpolation training. **Defer to Phase 4.** Phase 3b/3c first prove we can move vt at 1M. |

## What this changes vs the original Phase 3 plan

Originally we framed Phase 3b as "Stage A 1M CPT main run (250M tokens) + ramp ctx to 2M". The vt observation reframes it:

- The corpus we currently have (arXiv + PG-19, ~6.6M tokens, 30 × 256K seqs) is **long but not vt-rich**. Training on it 100M+ tokens worth would improve perplexity-on-long-text but likely leave vt at 10-30%.
- What we need: **synthetic vt-style data** (variable chains, multi-hop deductions) mixed into the long-context CPT data. The brainstorm doc `ttt.md` Section "5. 高级长上下文任务" hinted at this; we should now treat it as the **primary Phase 3b data extension goal**, not "moar arXiv tokens".

## Repo state

- 6 new result dirs: `eval/results/base_{ruler,niah}_*/`
- `scripts/run_base_1m_benchmarks.sh` — reproducible batch
- vLLM stack (post-Phase-3a env surgery): torch 2.11.0+cu130 + vllm 0.21.0 + transformers 5.8.1 + FlashInfer (no flash-attn in `.venv/`)
