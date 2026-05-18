# Phase 2 Report — 1M Eval Unblocked + Data v0 + Single-GPU CPT Smoke

| Field | Value |
|---|---|
| Date | 2026-05-18 |
| Tag | `phase2-smoke` |
| Hardware | 4× H100 80GB (cards 0-3); single node |
| Inference engine | vLLM TP=4 (1M ctx), transformers (≤256K eval) |
| Training | HF transformers + accelerate, single GPU (multi-GPU FSDP deferred to Phase 3) |
| Data | arXiv + PG-19, 7M tokens, packed into 30 sequences @ 256K |

## What Phase 2 delivered

1. **1M NIAH baseline measured** via vLLM TP=4 — closes Phase 1's BLOCKER.
   - `Qwen3.5-4B` + YaRN factor=4 → **NIAH 1M = 100% (10/10)** across depths 10/30/50/70/90.
   - Confirms NIAH alone is saturated at ≤1M for this model. **RULER VT remains the headline gap.**
2. **vLLM serving stack** (cards 0-3, TP=4, max_model_len=1M) — `eval/runners/vllm_server.sh`. YaRN baked into the local model `config.json#text_config.rope_parameters`.
3. **vLLM HTTP client + driver** — `eval/runners/_vllm_client.py` + `run_via_vllm.py`. Reuses Phase 1 cell builders for NIAH/RULER.
4. **Data pipeline v0** — `data/scripts/prepare_long_docs.py` (arXiv via `ccdv/arxiv-summarization` + PG-19 via `emozilla/pg19-test`) + `pack_docs.py` (multi-doc `<doc>` wrapper, token-level chunking, cross-doc boundary metadata). Output: 30 × 256K sequences, 6.58M tokens total.
5. **Training stack proven end-to-end on single H100**:
   - torch 2.8.0+cu128 + flash-attn 2.8.3 (local wheel) + bitsandbytes PagedAdamW8bit + FLA `FusedLinearCrossEntropyLoss` + gradient checkpointing.
   - Smoke run: 3 steps × 8K seq, loss 1.70 → 1.21 → 1.69. ~1s/step after warm-up.
   - HF-format checkpoint at `checkpoints/stage_a_smoke/step_3/`, `latest` symlink in place.
6. **Ckpt round-trip eval** — RULER 128K via transformers on the smoke ckpt = **84% overall, VT 20%** (matches `phase1-baseline` exactly; no regression).
7. **flame infra explored** but **not used for the smoke** (CPT-from-HF would need DCP conversion + torchtitan install + custom dataset adapter; deferred to Phase 3).

## Headline numbers

| Eval | phase1-baseline | After Phase 2 smoke | Δ |
|---|---|---|---|
| NIAH 32K | 100% | (n/a) | — |
| NIAH 128K | 100% | (n/a) | — |
| NIAH 512K (YaRN) | 100% | (n/a) | — |
| **NIAH 1M (YaRN, vLLM)** | **BLOCKED** | **100%** | (now measured) |
| RULER 128K overall | 84% | **84%** | **0** |
| RULER 128K `vt` | 20% | **20%** | **0** |

The single-GPU smoke (3 steps × 8K) is too small to move any metric. The point of this milestone was to prove the **end-to-end loop**: data → training → checkpoint → eval. Phase 3 runs the real training budget.

## What changed mechanically

**Code & configs (committed):**
- `pyproject.toml` — pinned `torch>=2.8,<2.9` + dropped `flash-attn` from auto-deps (uses local wheel).
- `scripts/install_flash_attn_local.sh` — helper to (re)install the prebuilt wheel after any `uv sync`.
- `eval/runners/vllm_server.sh` — single-command vLLM TP=4 launcher with cards 0-3.
- `eval/runners/_vllm_client.py` — minimal httpx-based OpenAI-compatible client.
- `eval/runners/run_via_vllm.py` — NIAH/RULER driver reusing Phase 1 cell builders.
- `data/scripts/prepare_long_docs.py` — arXiv + PG-19 ingestion (was stub).
- `data/scripts/pack_docs.py` — multi-doc 256K packer (was stub).
- `training/scripts/cpt_smoke_hf.py` — minimal CPT loop (bf16 + flash-attn + 8-bit Adam + fused/chunked CE).
- `training/flame_wrapper/toml_to_flame_args.py` — TOML → flame args translation (Phase 3 will use it).
- `configs/training/stage_a_1m_cpt.toml` — CP=4 (was 16) + `[smoke]` block.
- `Makefile` — `vllm-serve`, `vllm-stop`, `setup-ruler` targets.
- `tests/test_vllm_client.py`, `test_data_pipeline.py`, `test_flame_wrapper.py` — 7 new tests.

**Environment surgery (path that worked):**
1. Start: torch 2.11+cu130 (initial `uv sync`).
2. vllm 0.21.0 transitively bumps torch to 2.11+cu13; **flash-attn 2.8 wheel ABI clashes with this torch**.
3. Force-downgrade to torch 2.8.0+cu128 via `pytorch.org/whl/cu121` index; reinstall flash-attn local wheel + accelerate; vllm 0.21.0 re-resolves to a torch-2.8-compatible build.
4. Final stable stack: torch 2.8.0+cu128, flash-attn 2.8.3, transformers 5.8.1, vllm 0.21.0, accelerate 1.13.0, fla 0.5.1, bitsandbytes 0.49.2.

## Known limits (Phase 3 must address)

### 1. Multi-GPU FSDP doesn't shard cleanly with our HF model + accelerate

We tried (in increasing escalation):
- `FULL_SHARD` + TRANSFORMER_BASED_WRAP + `use_orig_params=True` → "`weight' must be 2-D`" in `embed_tokens.forward` (FSDP flattened the param even with use_orig_params; our forward was reaching weights without going through the FSDP gather hook).
- `SHARD_GRAD_OP` (ZeRO-2) — same 1-D weight error.
- Monkey-patching `Qwen3_5ForCausalLM.forward` to do chunked CE so the entire forward runs inside the FSDP wrap — moved past the 1-D weight issue but OOMs at 32K-64K (~79GB allocated per card before any activations of size). Strongly suggests optimizer state isn't actually sharding.
- Plain `multi_gpu` (DDP) + 8-bit Adam at 16K — also OOM (DDP keeps full state per card; bnb 8-bit Adam still has fp32 master weights).

### 2. `causal-conv1d` cannot be installed for torch 2.8 from PyPI

Every `uv pip install causal-conv1d` (or with `--no-binary`) resolves to a wheel built against a different torch ABI than what's in our venv. Without it, FLA's Gated Delta Rule falls back to torch — functional but materializes larger intermediate tensors per layer.

### 3. flash-attn wheel pins us to torch 2.8

Newer torch (2.11+cu13) → flash-attn must be rebuilt from source (~3hr). The prebuilt wheel locks us to torch 2.8 + cu12. vLLM 0.21.0 happens to work in this combo, but any upstream torch bump cascades.

## Phase 3 prerequisites

Before launching Stage A main (1M CPT, 250M tokens) and Stage B (2M):

1. **Adopt LLaMA-Factory's `pt` mode** for CPT, OR **DeepSpeed ZeRO-3** in our custom script.
   - LF is already cloned at `external/LLaMA-Factory`. Its `pt` stage handles the FSDP/sharding setup cleanly, reads our packed_256k.jsonl directly via a `dataset_info.json` registration, and supports DeepSpeed Ulysses for sequence parallelism (needed for 1M+).
   - Estimated effort: 1-2 days to register the dataset + wire stage_a TOML → LF yaml.
2. **Build flash-attn from source against current torch** in a separate venv, OR keep the torch 2.8 pin and treat flash-attn as immovable. The Phase 2 path (torch 2.8 + flash-attn wheel) works; sticking with it is the path of least resistance for Phase 3.
3. **Resolve `causal-conv1d`** OR accept the FLA torch-fallback cost for Stage A/B. At 1M-2M with FSDP, the GDN buffers are a real cost but not blocking.

## Quality gate carry-overs into Phase 3

From `phase1-baseline`:
- **RULER 128K `vt`: 20% → ≥ 60%** (the headline metric).
- RULER 128K overall: 84% → ≥ 92%.
- NIAH 1M: now measured (100% via vLLM) — stays at 100% post-CPT.
- Short benchmarks (MMLU/GSM8K/HumanEval): within 0.02 of base — measured in Phase 4 with Stage D short SFT.

New Phase-3 specific:
- 2M context **must work end-to-end** (training ckpt + vLLM eval at 2M).
- Stage A 1M main: minimum 100M tokens trained (full 250M plan target).

## Repo state at tag

- 38 commits on `main` (push to origin already up to commit pre-report).
- 33 tests passing (`uv run pytest tests/ -q`).
- ruff clean across source.
- External cloned: `flame`, `flash-linear-attention`, `LLaMA-Factory`, `RULER`.
- Local model: `/home/user01/Minko/models/Qwen3.5-4B/` (config patched with YaRN-1M; backup at `config.json.bak`).
- Smoke ckpt: `checkpoints/stage_a_smoke/{step_3, latest -> step_3}/`.

## Phase 2 done

Tag `phase2-smoke`. Phase 3 plan to be written next — primary scope is **Stage A 1M main run via LLaMA-Factory `pt` mode** (or DeepSpeed ZeRO-3) on cards 0-3, plus the path to **2M** via DeepSpeed Ulysses sequence parallelism.
