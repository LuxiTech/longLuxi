# NIAH @ 1M YaRN — BLOCKED on OOM

**Status:** Not produced. 1M dense inference OOMs on single H100 80GB.

## Attempts made

1. SDPA + torch fallback for GDN → OOM in `torch_chunk_gated_delta_rule` (fp32 promotion).
2. Installed `flash-linear-attention` + `causal-conv1d` (FLA kernel path active) → still OOM in
   `chunk_gated_delta_rule_fwd_intra` allocating `(B, T, HV, BT)` tensor.
3. Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → no help; OOM persists.

Single GPU 80GB cannot fit:
- 4B model weights (~8GB)
- Standard attention KV cache @ 1M for ~half the layers
- GDN intra-chunk activation buffers @ 1M
without flash-attention or model sharding.

## Workaround used for Phase 1 baseline

Reported 512K (still 2× over native 262144 with YaRN factor=4.0) as the upper-bound
YaRN baseline that is feasible on this hardware. See `niah_512k_yarn.md`.

## Resolution path for Phase 2+

Two viable paths, pick before Phase 2 1M CPT eval:
- **Install flash-attention 2**: ~3hr build from source against torch 2.11+cu130.
  Once built, single-H100 1M inference should fit (flash-attn avoids materializing
  the full attention matrix).
- **Use vLLM serving with TP=4**: production-grade engine, paged attention, YaRN
  support, splits model across cards 0-3. Cleaner long-term but requires standing
  up a serving process per eval.

Tracking: open this as Phase 2 prerequisite.

---

## RESOLVED in Phase 2

Resolution path taken: **vLLM TP=4 serving** (cards 0-3) with YaRN factor=4 baked
into the local model's `config.json#text_config.rope_parameters`. flash-attn was
NOT needed — vLLM uses flashinfer kernels. Tested at length=1000000 (not 2^20=1048576
since that exceeded vLLM's max_model_len=1010000 by ~5%).

Result: **NIAH 1M @ YaRN factor=4 = 100% (10/10 cells)** across depths 10/30/50/70/90.

Phase 1 had hypothesized accuracy 0.4-0.9; actual is 1.0. Implication: NIAH alone at
1M is **not** the metric that will show Stage A CPT improvement on this model. RULER
variants (especially `vt`) remain the headline gap to close.
