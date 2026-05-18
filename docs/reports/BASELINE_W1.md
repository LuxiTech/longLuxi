# W1 Baseline Report — Qwen3.5-4B

| Field | Value |
|---|---|
| Date | 2026-05-18 |
| Model | `Qwen/Qwen3.5-4B` (256K native + YaRN-static to 1M) |
| Hardware | 1× H100 80GB (cards 0-3 budget; eval needs only one) |
| Eval engine | HF Transformers + SDPA, FLA kernels for GDN, `enable_thinking=False` |
| Tag | `phase1-baseline` |

## Summary

Phase 1 establishes the **post-trained-but-uncalibrated-for-long-context** baseline that
Phase 2 CPT (Stage A 1M, Stage B 2M) must improve against. Two clear signals:

1. **NIAH alone at ≤ 512K is saturated** — Qwen3.5-4B with static YaRN factor=4 gets 100%
   across all 30 cells we ran (32K / 128K / 512K × 5 depths × 2). NIAH is not where the
   CPT delta will be visible. *(1M dense inference is blocked on single-GPU memory; see
   "Known blockers" below.)*
2. **RULER 128K reveals the real weakness — variable tracking (VT) at 20%**. Even at
   native context, the model fails 8/10 chain-of-aliases lookups, returning the variable
   index ('1', '0') instead of resolving the chain. This is the gap Stage A CPT should
   target.

## NIAH

| Length     | YaRN config        | Accuracy | Cells |
|------------|--------------------|----------|-------|
| 32,768     | none               | **100%** | 10    |
| 131,072    | none               | **100%** | 10    |
| 524,288    | `yarn_1m.json` f=4 | **100%** | 10    |
| 1,048,576  | `yarn_1m.json` f=4 | *blocked* | 0    |

### By-depth (all three runs)

Every depth × length cell scored 100%:

| Depth | 32K | 128K | 512K (YaRN) |
|-------|-----|------|-------------|
| 10%   | 2/2 | 2/2  | 2/2 |
| 30%   | 2/2 | 2/2  | 2/2 |
| 50%   | 2/2 | 2/2  | 2/2 |
| 70%   | 2/2 | 2/2  | 2/2 |
| 90%   | 2/2 | 2/2  | 2/2 |

## RULER 128K (5-task MVP subset)

| Task              | Accuracy | n  | Comment |
|-------------------|----------|----|---------|
| `niah_single_1`   | **100%** | 10 | Trivial — same as NIAH |
| `niah_multikey_1` | **100%** | 10 | 1 real key + 3 distractor "tracking ids" — handled cleanly |
| `niah_multiquery` | **100%** | 10 | 4 labeled needles, asked for one by label |
| **`vt`**          | **20%**  | 10 | Variable-tracking chain — failure mode below |
| `qa_1`            | **100%** | 10 | Single-hop synthetic QA |
| **Overall**       | **84%**  | 50 | |

### `vt` failure pattern

For chains like `v0=v1`, `v1=v2`, `v2=v3`, `v3=54473` (shuffled, embedded in
~131K tokens of haystack), Qwen3.5-4B consistently outputs the variable index
('1', '0', '2') instead of resolving to the final number `54473`. Three example
predictions vs expected:

```
expected='54473'  pred='1'
expected='68800'  pred='0'
expected='42742'  pred='1'
```

The model appears to read `v0=v1` as `v0=1` (taking the index of `v1` as the value).
Even at native 128K with no YaRN extrapolation, multi-hop chain resolution over a long
context is the headline weakness. This is precisely what the project's Stage A 1M CPT
(memory-format SFT in Phase 6 too) is positioned to improve.

## Known blockers

### NIAH @ 1M is blocked on single-H100 memory

Three attempts (SDPA torch fallback, FLA kernels, expandable_segments allocator) all
OOM in the GDN intra-chunk forward at sequence length 1,048,576 on a single 80GB H100
(see [`phase1_artifacts/niah_1m_yarn_BLOCKED.md`](phase1_artifacts/niah_1m_yarn_BLOCKED.md)).

Two viable resolution paths (must pick before Phase 2 1M eval):
- **Install flash-attention 2** — pre-built wheels for torch 2.11+cu130 unavailable in
  current environment; source build is ~3hr. Once built, single-H100 1M inference is
  expected to fit (flash-attn avoids materializing the attention matrix).
- **vLLM serving with TP=4** — splits the model across cards 0-3, paged-attention KV
  cache. More work to wire up, but the right long-term answer for any reader serving.

The 512K result is reported as the upper-bound YaRN baseline that fits on the current
eval rig.

## Quality gates for Phase 2 (Stage A 1M CPT)

To declare Stage A a success, the post-CPT model must show *measurable* improvement on
the headline weakness:

- **RULER 128K `vt`**: from 20% → **≥ 60%** (this is the headline metric, not NIAH)
- **RULER 128K overall**: from 84% → **≥ 92%** (lifted by VT)
- **NIAH @ 1M**: from "blocked" → ≥ 80% (requires resolving the 1M blocker first)
- **Short benchmarks** (MMLU/GSM8K/HumanEval, to be measured in Phase 2): within 0.02 of base.

## Repo state at tag

- 21 commits on `main`
- 29 tests passing (`uv run pytest tests/ -q`)
- ruff clean across `src/`, `tests/`, `eval/runners/`
- External: `flame`, `flash-linear-attention`, `LLaMA-Factory`, `RULER` cloned to `external/`
- `Qwen/Qwen3.5-4B` cached in `~/.cache/huggingface/hub/`

## Artifacts

Raw eval outputs in [`phase1_artifacts/`](phase1_artifacts/):

- `niah_32768_native.{md,json}`
- `niah_131072_native.{md,json}`
- `niah_512k_yarn.{md,json}`
- `niah_1m_yarn_BLOCKED.md` *(no JSON — run did not complete)*
- `ruler_128k_native.{md,json}`

Predictions-level data lives under `eval/_outputs/baseline/` and `eval/_outputs/ruler/`
(gitignored). To regenerate:

```bash
uv run python scripts/baseline_eval.py --task niah --max-len 32768  --lengths 32768  --depths 10 30 50 70 90 --n-per-cell 2
uv run python scripts/baseline_eval.py --task niah --max-len 131072 --lengths 131072 --depths 10 30 50 70 90 --n-per-cell 2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/baseline_eval.py \
    --task niah --max-len 524288 --lengths 524288 --depths 10 30 50 70 90 --n-per-cell 2 \
    --yarn-config configs/yarn/yarn_1m.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python eval/runners/run_ruler.py \
    --lengths 131072 --n-per-task 10
```

## Phase 1 done

All 14 tasks of `docs/superpowers/plans/2026-05-18-phase1-env-baseline.md` complete (Task 11
is DONE_WITH_CONCERNS due to the documented 1M blocker; otherwise green). Tagged
`phase1-baseline`. Phase 2 plan to be written by next `writing-plans` pass.
