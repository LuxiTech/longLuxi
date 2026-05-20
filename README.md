# longluxi — Qwen3.5-4B with 2M Context

**TL;DR.** This project extends `Qwen/Qwen3.5-4B` to a **2 M-token usable context** via YaRN factor=8 RoPE scaling, validated by a comprehensive long-context benchmark suite (NIAH, RULER, LongBench-v2, InfiniteBench). The model achieves **100% NIAH at 1 M / 1.5 M / 2 M**, with no architecture changes — only a config patch plus targeted continued pretraining.

Built and benchmarked on 4 × H100 80 GB (cards 0–3 only) over Phase 1–4 (May 2026).

---

## What this repo contains

```
longluxi/
├── docs/
│   ├── reports/         # measured benchmark results + retrospectives
│   ├── notes/           # methodology distill (UltraLong-8B)
│   └── superpowers/specs/  # original design doc
├── configs/
│   ├── accelerate/      # FSDP2 config (cards 0-3)
│   ├── deepspeed/       # ZeRO-3 config (alternative path)
│   └── training/        # current stage YAMLs (LF format)
├── data/
│   ├── scripts/         # ingestion (arXiv, PG-19) + synthesis (vt_chain, entity, code) + packer
│   └── processed/       # gitignored — packed jsonls live here
├── eval/
│   ├── runners/         # NIAH, RULER, LongBench-v2, InfiniteBench drivers
│   └── results/         # committed measured metrics + predictions
├── scripts/             # installer, baseline-eval, batch benchmarks, post-train eval
├── src/longluxi/        # path constants, shared utilities
├── tests/               # 35+ tests passing
├── external/LLaMA-Factory/  # training framework (pinned)
└── checkpoints/         # gitignored — Stage A & B training artifacts (20 GB each)
```

---

## Headline measurements (base Qwen3.5-4B + YaRN factor=8)

### NIAH (single-needle retrieval, all-depth-all-length)

| context | accuracy | wall-time |
|---|---|---|
| 1 M  | **100%** | 6.5 min |
| 1.5 M | **100%** | 12.7 min |
| 2 M  | **100%** | 20.8 min |

This is the **primary verification** of the 2 M context claim.

### RULER (5 synthetic tasks × 10 cells per length)

| ctx | overall | niah_single | niah_multikey | niah_multiquery | vt | qa_1 |
|---|---|---|---|---|---|---|
| 1 M | 0.82 | 1.00 | 1.00 | 1.00 | 0.10 | 1.00 |
| 1.5 M | 0.72 | 1.00 | 0.90 | 1.00 | 0.10 | 0.60 |
| 2 M | 0.74 | 1.00 | 1.00 | 0.80 | 0.00 | 0.90 |

`vt` (variable tracking) is length-independent at 0–10 % — a reasoning gap inherited from the base model, not a context-length issue. See `docs/reports/1M_CAPABILITY_REPORT.md`.

### Real-doc QA at long context

| benchmark | metric | base @ factor=8 | n |
|---|---|---|---|
| LongBench-v2 long | accuracy (4-choice) | **0.254** | 71 |
| InfiniteBench longbook_choice (w/ ctx) | accuracy (4-choice) | **0.804** | 225 |
| InfiniteBench longbook_choice (NO ctx, leakage control) | accuracy | 0.354 | 229 |
| → real context contribution | gain | **+45 pp** | — |

The leakage control confirms the model **genuinely uses the long context** for ~45 pp of its answers (the remaining ~35 pp comes from pretraining memorization of public-domain books).

Full numbers: `docs/reports/BASELINE_2M.md`, `docs/reports/1M_CAPABILITY_REPORT.md`.

---

## How the 2M capability was achieved

### 1. RoPE-only config patch (no architecture change)

`/home/user01/Minko/models/Qwen3.5-4B/config.json` is patched to:

```jsonc
"text_config": {
  "max_position_embeddings": 2097152,          // 2 M
  "rope_parameters": {
    "rope_type": "yarn",
    "factor": 8.0,                              // was 4.0
    "original_max_position_embeddings": 262144,
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10]
  }
}
```

Original factor=4 config is preserved at `config.json.yarn4.bak`.

### 2. Continued pretraining (optional, see Stage A/B retros)

Two CPT runs were attempted to push reasoning + real-doc QA:

| Stage | ctx | data | LR | steps | outcome |
|---|---|---|---|---|---|
| A | 128K | v2 synthetic + arXiv + PG-19 (228 M tok) | 3e-5 | 500 | NIAH 2M dropped 100→80 %; LongBench-v2 long +7 pp |
| B | 128K | v3 synthetic with COT traces + multi-target (192 M tok) | 1e-5 | 250 | NIAH 2M held at 100 %; RULER no improvement |

**Conclusion** (see `STAGE_A_RETROSPECTIVE.md` and `STAGE_B_RETROSPECTIVE.md`): PT on synthetic Q+A data does **not** teach the model to execute the test-time algorithm — it only learns the joint distribution of the doc. The base model's reasoning ceiling is what it is; SFT is the right tool to lift it. The trained Stage B checkpoint preserves NIAH 2M = 100 % while remaining within the base model's reasoning band.

### 3. Inference

vLLM 0.21.0 with TP=4 on cards 0–3:

```bash
MODEL=/home/user01/Minko/models/Qwen3.5-4B \
  PORT=8001 \
  MAX_LEN=2097152 \
  GPU_MEM_UTIL=0.95 \
  bash eval/runners/vllm_server.sh
```

Single 2 M prompt fits in ~80 GiB / card (KV cache + model). NIAH 2 M wall-time ≈ 2 min per cell after warm-up.

---

## Repro

### Environment

```bash
# Eval venv (vLLM, transformers 5.8, FlashInfer, no flash-attn)
uv sync --extra eval

# Training venv (separate; LF + accelerate 1.11 + transformers 5.6 + flash-attn 2.8.3)
bash scripts/install_lf.sh    # builds .venv-lf/
```

Two venvs are intentionally isolated — vLLM 0.21 and LLaMA-Factory have incompatible pins. See `docs/notes/ultralong_methodology.md` for context.

### Base-model benchmarks (no training)

```bash
# Launch vLLM at 2M (one-time)
MAX_LEN=2097152 bash eval/runners/vllm_server.sh > /tmp/vllm.log 2>&1 &

# Wait for ready, then run the full batch
bash scripts/run_base_2m_benchmarks.sh
```

Results land under `eval/results/base_yarn8_*` and `eval/results/base_*`.

### Continued pretraining (Stage A/B style)

```bash
# Data
.venv/bin/python data/scripts/prepare_long_docs_v2.py --source arxiv --max-docs 5000
.venv/bin/python data/scripts/prepare_long_docs_v2.py --source pg19  --max-docs 800
.venv/bin/python data/scripts/synthesize_reasoning_v2.py --gen vt_chain_v2 --n-docs 3000
.venv/bin/python data/scripts/synthesize_reasoning_v2.py --gen entity_kv_v2 --n-docs 2000
.venv/bin/python data/scripts/synthesize_reasoning_v2.py --gen code_trace_v2 --n-docs 2000
.venv/bin/python data/scripts/pack_v2.py --target-ctx 131072 --input-jsonls data/processed/v2/*.jsonl data/processed/v3/*.jsonl --out data/processed/v3/packed_128k_v3.jsonl

# Training (Stage B reference)
bash scripts/launch_lf_smoke.sh configs/training/stage_b_v3_128k_fsdp2.yaml
# Hard rule: cards 0-3 only — every launch script sets CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## Project history

| Phase | Goal | Tag | Status |
|---|---|---|---|
| 1 | Env + Phase-1 baseline NIAH/RULER @ 128K | phase1-baseline | done |
| 2 | 1 M NIAH via vLLM, data pipeline v0, single-GPU smoke | phase2-smoke | done |
| 3a | Multi-GPU FSDP2 through 128K, ckpt round-trip eval | phase3a-lf-smoke | done |
| 3b | Base 1 M characterization (NIAH/RULER/LongBench-v2/InfiniteBench) | (none) | done |
| 4 | YaRN factor=8 → 2 M, base benchmarks, Stage A/B CPT | (current) | training experiments concluded |

Detailed reports in `docs/reports/`:

- **`1M_CAPABILITY_REPORT.md`** — comprehensive characterization of Qwen3.5-4B at 1M (NIAH, RULER, LongBench-v2, InfBench + leakage control).
- **`BASELINE_1M.md`** — synthetic NIAH/RULER baseline at YaRN factor=4 (32K → 1M).
- **`BASELINE_2M.md`** — synthetic NIAH/RULER baseline at YaRN factor=8 (1M → 2M).
- **`STAGE_A_RETROSPECTIVE.md`** — Stage A 128K CPT, what worked + what failed.
- **`STAGE_B_RETROSPECTIVE.md`** — Stage B v3-data + lower-LR CPT, partial eval + lessons.

---

## Known limits & next steps

**Verified at 2M:**
- 100 % single-needle retrieval across all depths
- Real-doc context utilization (+45 pp gain over leakage baseline on InfBench longbook)

**Not improved (base-model reasoning ceiling):**
- RULER vt (variable tracking): 0–30 %, length-independent
- LongBench-v2 long: ~0.25 (4-choice random baseline)

**Future work (not in current scope):**
- **Reasoning SFT** on the Stage B checkpoint with explicit answer-only loss masking (ShareGPT/OrcaMath/Magicoder mix per the UltraLong recipe).
- **Sequence-parallel CPT** (DeepSpeed Ulysses CP=4) if direct training at 256 K–1 M context is needed.
- **Open-ended long-doc QA evals** beyond multiple choice (Qasper, NarrativeQA, MuSiQue F1).

See `docs/notes/ultralong_methodology.md` for the reference recipe.

---

## Hardware & operating rules

- 4 × H100 80 GB, cards 0–3 only. Cards 4–7 are reserved for other workloads on the same host.
- bf16 throughout; flash-attn 2.8.3 in training, FlashInfer in serving.
- Two isolated venvs (`.venv/` for eval, `.venv-lf/` for training) — see `scripts/install_lf.sh`.

---

## License

Apache-2.0.
