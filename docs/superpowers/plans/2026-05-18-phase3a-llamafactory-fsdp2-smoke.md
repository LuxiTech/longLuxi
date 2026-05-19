# Phase 3a Implementation Plan — LLaMA-Factory FSDP2 Smoke (cards 0-3, up to 256K ctx)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken accelerate-FSDP path from Phase 2 by running real distributed CPT on 4 H100s via LLaMA-Factory's `pt` stage with FSDP2 sharding. End state: 256K-ctx smoke training step proven to truly shard (per-card memory < 30 GB), ckpt round-trips through existing vLLM-based eval with no regression vs `phase1-baseline`.

**Architecture:**
- Isolated venv (`.venv-lf/`) for LF training with transformers ≤ 5.6 / accelerate ≤ 1.11 per LF's pyproject pin. Existing main venv keeps vLLM 0.21 + transformers 5.8.1 for serving and eval.
- LF reads our packed jsonl directly (`columns: {prompt: text}`); no re-packing.
- FSDP2 with `TRANSFORMER_BASED_WRAP` on `Qwen3_5DecoderLayer` — LF ships this exact policy in `examples/accelerate/fsdp2_config_qwen35.yaml`.
- Cards 0-3 only (`num_processes: 4`); bf16 throughout; flash-attn 2 via local wheel; gradient checkpointing on.
- Smoke runs ramp ctx 32K → 64K → 128K → 256K, each 3 steps, primarily verifying memory profile (true sharding) before pushing to Phase 3b main training.

**Tech Stack:** LLaMA-Factory PT stage, accelerate FSDP2, transformers ≤ 5.6, torch 2.8.0 + flash-attn 2.8.3 (local wheel), bf16, Qwen3.5-4B with YaRN factor=4 baked in.

---

## Pre-flight context (read before starting)

**Existing repo state (Phase 2 outputs):**
- Model: `/home/user01/Minko/models/Qwen3.5-4B/` — YaRN factor=4 patched into `config.json`, backup at `config.json.bak`. `max_position_embeddings=1010000`.
- Data: `data/processed/packed_256k.jsonl` — 30 sequences × ~260K tokens each, `{seq_id, target_ctx, docs, text, n_tokens, mask_doc_boundaries}` schema. **LF only consumes `text` field via `columns: {prompt: text}`.**
- LF clone: `external/LLaMA-Factory/` (installed but not yet usable from our main venv due to dep pin conflicts).
- flash-attn wheel: `/home/user01/Minko/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl` — must be installed AFTER torch 2.8 is pinned.
- Phase1 baseline metrics (do not regress): RULER 128K overall = 84%, RULER 128K vt = 20%, NIAH 32K/128K/512K = 100%, NIAH 1M (vLLM) = 100%.
- Main venv (`.venv/` at project root): keep untouched — needed by `eval/runners/vllm_server.sh` for ckpt round-trip eval.

**LF quirks already discovered:**
- LF's `data/dataset_info.json` registers datasets by name; relative `file_name` resolves under `external/LLaMA-Factory/data/`. We register our packed jsonl by SYMLINK (avoid duplicating 6.6MB of data).
- `examples/accelerate/fsdp2_config_qwen35.yaml` is the **only** built-in config with the exact wrap class our model needs (`Qwen3_5DecoderLayer`). Use it as the template; just change `num_processes` to 4.
- LF pretrain example (`examples/train_lora/qwen3_lora_pretrain.yaml`) is LoRA — for full CPT, set `finetuning_type: full` and drop `lora_*` keys.
- `stage: pt` is the right stage; LF will not apply chat template, just concatenates and shifts for next-token CE.

**Constraints (user-mandated, do not violate):**
- Cards 0-3 ONLY. Set `CUDA_VISIBLE_DEVICES=0,1,2,3` in every launch.
- bf16 mixed precision (no 8-bit Adam).
- All commits must be on `main` branch in `/home/user01/Minko/longluxi/`.

---

## File Structure

**Create:**
- `.venv-lf/` — isolated venv for LF training (sibling of `.venv/`).
- `scripts/install_lf.sh` — reproducible LF venv setup (torch 2.8 → LF → flash-attn local wheel).
- `configs/accelerate/fsdp2_cards0_3.yaml` — accelerate config (FSDP2, 4 cards).
- `configs/training/lf_qwen35_4b_smoke_{32k,64k,128k,256k}.yaml` — four LF training YAMLs, one per ctx step.
- `scripts/launch_lf_smoke.sh` — wrapper that activates `.venv-lf` and runs `accelerate launch ... llamafactory-cli train ...`.
- `scripts/probe_gpu_mem.sh` — backgrounded `nvidia-smi` polling during step 1 to capture peak per-card memory.
- `tests/test_lf_dataset_register.py` — unit test that `dataset_info.json` parses and `longluxi_packed_256k` entry resolves to our symlinked file.
- `docs/reports/PHASE3A_REPORT.md` — close-out report.

**Modify:**
- `external/LLaMA-Factory/data/dataset_info.json` — add one entry. (LF clone is in our repo as a submodule-like checkout; this edit is committed.)
- `external/LLaMA-Factory/data/longluxi_packed_256k.jsonl` — symlink to `data/processed/packed_256k.jsonl`.

**Untouched:**
- `.venv/` — main venv stays at current versions (vLLM + transformers 5.8.1).
- `/home/user01/Minko/models/Qwen3.5-4B/` — model files. No edits.
- `data/processed/packed_256k.jsonl` — source of truth.
- `training/scripts/cpt_smoke_hf.py` — Phase 2 single-GPU smoke; keep for reproducibility, do not delete.

---

## Task 1: Create isolated LF venv with pinned deps

**Files:**
- Create: `scripts/install_lf.sh`
- Verify: `.venv-lf/bin/python` exists and is Python 3.11

- [ ] **Step 1: Write the installer script**

```bash
#!/usr/bin/env bash
# Build isolated LF training venv. Idempotent: re-run after edits.
# Outputs: .venv-lf/ with torch 2.8 + flash-attn 2.8.3 + LF + accelerate 1.11 + transformers 5.6.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d .venv-lf ]]; then
  uv venv .venv-lf --python 3.11
fi

# Activate by setting VIRTUAL_ENV so `uv pip` targets the right venv.
export VIRTUAL_ENV="$(pwd)/.venv-lf"

# 1) Pin torch 2.8.0+cu128 first (flash-attn local wheel was built against this).
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# 2) LF editable + its training extras (datasets, peft, trl, deepspeed kept off-train path).
uv pip install -e ./external/LLaMA-Factory \
    --extra-index-url https://download.pytorch.org/whl/cu128

# 3) Force LF-compatible pins (LF's pyproject allows ranges; lock at upper bound that we know works).
uv pip install \
    "transformers>=4.55,<=5.6.0" \
    "accelerate>=1.10,<=1.11.0" \
    "datasets>=2.14,<4.0" \
    "trl>=0.8,<0.18"

# 4) Flash-attn local wheel (must come AFTER torch is pinned).
uv pip install /home/user01/Minko/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-deps

# 5) Smoke imports.
python -c "
import torch, transformers, accelerate, llamafactory, flash_attn
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} ngpu={torch.cuda.device_count()}')
print(f'transformers={transformers.__version__}')
print(f'accelerate={accelerate.__version__}')
print(f'llamafactory={llamafactory.__version__}')
print(f'flash_attn={flash_attn.__version__}')
"
echo '[install_lf] OK'
```

- [ ] **Step 2: Run installer**

Run: `bash scripts/install_lf.sh`
Expected last line: `[install_lf] OK` with all versions printed, `cuda=True ngpu>=4`.

If `flash_attn` import fails: most likely torch got bumped by a transitive dep. Re-pin torch 2.8 and re-install flash-attn wheel.

- [ ] **Step 3: Commit**

```bash
git add scripts/install_lf.sh .gitignore
# Add .venv-lf/ to .gitignore if not already covered by .venv pattern.
git commit -m "feat(phase3a): isolated LF venv installer with torch 2.8 + flash-attn pin"
```

Confirm `.venv-lf/` is gitignored (existing `.gitignore` should cover via `*venv*` or add `.venv-lf/`).

---

## Task 2: Verify Qwen3.5-4B loads cleanly in LF's transformers

**Files:**
- Create: `tests/test_lf_qwen35_loads.py`

- [ ] **Step 1: Write the failing test**

```python
"""Verify Qwen3.5-4B loads in the LF venv with YaRN config intact.

This catches: (a) transformers version too old to know Qwen3_5ForCausalLM,
(b) trust_remote_code path not wired, (c) YaRN config dropped on load.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LF_VENV_PY = REPO_ROOT / ".venv-lf" / "bin" / "python"
MODEL_PATH = "/home/user01/Minko/models/Qwen3.5-4B"


def test_qwen35_loads_in_lf_venv():
    assert LF_VENV_PY.exists(), f"LF venv missing — run scripts/install_lf.sh first ({LF_VENV_PY})"
    script = f"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM

cfg = AutoConfig.from_pretrained({MODEL_PATH!r}, trust_remote_code=True)
assert cfg.__class__.__name__ in ('Qwen3_5Config', 'Qwen3Config'), f'unexpected config class: {{cfg.__class__.__name__}}'

# YaRN: factor=4 should be present in text_config.rope_parameters or rope_scaling.
text_cfg = getattr(cfg, 'text_config', cfg)
rope = getattr(text_cfg, 'rope_parameters', None) or getattr(text_cfg, 'rope_scaling', None)
assert rope is not None, 'YaRN config missing — model config.json not patched'
factor = rope.get('factor') if isinstance(rope, dict) else None
assert factor == 4, f'YaRN factor != 4 ({{factor}})'

# Load weights to CPU to keep test fast; just need the class to resolve.
model = AutoModelForCausalLM.from_pretrained(
    {MODEL_PATH!r}, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='meta'
)
assert model.__class__.__name__ in ('Qwen3_5ForCausalLM', 'Qwen3ForCausalLM'), \\
    f'unexpected model class: {{model.__class__.__name__}}'
print('OK', model.__class__.__name__, 'YaRN factor=', factor)
"""
    proc = subprocess.run(
        [str(LF_VENV_PY), "-c", script],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert "OK" in proc.stdout, proc.stdout
```

- [ ] **Step 2: Run test to verify it fails (if venv not built yet) or passes**

Run: `uv run --no-project pytest tests/test_lf_qwen35_loads.py -v`
Expected: PASS if Task 1 done; FAIL with clear "LF venv missing" if Task 1 not yet done.

- [ ] **Step 3: If FAIL with config class mismatch, investigate**

Likely causes:
- transformers < 4.55 doesn't have Qwen3.5 model registration → bump LF pin and re-install
- YaRN factor not in config → re-check `/home/user01/Minko/models/Qwen3.5-4B/config.json` shows `text_config.rope_parameters.factor: 4`

Do NOT proceed to next task until this passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_lf_qwen35_loads.py
git commit -m "test(phase3a): verify Qwen3.5-4B + YaRN config loads in LF venv"
```

---

## Task 3: Register packed dataset with LF

**Files:**
- Create: `external/LLaMA-Factory/data/longluxi_packed_256k.jsonl` (symlink)
- Modify: `external/LLaMA-Factory/data/dataset_info.json` (add one entry)
- Create: `tests/test_lf_dataset_register.py`

- [ ] **Step 1: Create the symlink**

```bash
ln -sfn "$(pwd)/data/processed/packed_256k.jsonl" \
    external/LLaMA-Factory/data/longluxi_packed_256k.jsonl
ls -la external/LLaMA-Factory/data/longluxi_packed_256k.jsonl
```

Expected: symlink shown pointing at our packed jsonl.

- [ ] **Step 2: Edit `dataset_info.json` — add entry**

Use Edit tool. Insert new key right after `"c4_demo"` entry. The exact JSON to add:

```json
  "longluxi_packed_256k": {
    "file_name": "longluxi_packed_256k.jsonl",
    "columns": {
      "prompt": "text"
    }
  },
```

The `columns: {prompt: text}` mapping tells LF that the `text` field in our jsonl is the document body for `stage: pt`. Extra fields (`seq_id`, `target_ctx`, `docs`, `n_tokens`, `mask_doc_boundaries`) are ignored.

- [ ] **Step 3: Write the failing test**

```python
"""Verify LF dataset registration: longluxi_packed_256k resolves and parses."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LF_DATA = REPO_ROOT / "external" / "LLaMA-Factory" / "data"


def test_longluxi_packed_256k_registered():
    info = json.loads((LF_DATA / "dataset_info.json").read_text())
    assert "longluxi_packed_256k" in info, "dataset not registered in dataset_info.json"
    entry = info["longluxi_packed_256k"]
    assert entry["file_name"] == "longluxi_packed_256k.jsonl"
    assert entry["columns"]["prompt"] == "text"


def test_longluxi_packed_256k_file_resolves():
    f = LF_DATA / "longluxi_packed_256k.jsonl"
    assert f.exists(), f"data file missing: {f}"
    # Should resolve to our packed jsonl with 30 sequences.
    n = sum(1 for _ in f.open())
    assert n == 30, f"expected 30 packed sequences, got {n}"
    rec = json.loads(f.open().readline())
    assert "text" in rec
    assert len(rec["text"]) > 100_000, "first sequence text too short — wrong file?"
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_lf_dataset_register.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add external/LLaMA-Factory/data/dataset_info.json \
        external/LLaMA-Factory/data/longluxi_packed_256k.jsonl \
        tests/test_lf_dataset_register.py
git commit -m "feat(phase3a): register longluxi_packed_256k as LF pt dataset"
```

Note: committing the symlink is intentional. Git stores it as a symlink object; it resolves correctly on any clone where `data/processed/packed_256k.jsonl` exists.

---

## Task 4: FSDP2 accelerate config tuned for cards 0-3

**Files:**
- Create: `configs/accelerate/fsdp2_cards0_3.yaml`

- [ ] **Step 1: Write the config**

```yaml
# Adapted from external/LLaMA-Factory/examples/accelerate/fsdp2_config_qwen35.yaml.
# Changes from upstream:
#   - num_processes: 8 → 4 (cards 0-3 only)
#   - reshard_after_forward stays true (memory)
#   - fsdp_offload_params stays false (we have 80 GB cards; offload only if OOM)
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_version: 2
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen3_5DecoderLayer
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: FULL_STATE_DICT
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
use_cpu: false
```

- [ ] **Step 2: Validate YAML parses**

Run: `python -c "import yaml; print(yaml.safe_load(open('configs/accelerate/fsdp2_cards0_3.yaml')))"`
Expected: dict with `num_processes: 4`, `fsdp_version: 2`.

- [ ] **Step 3: Commit**

```bash
git add configs/accelerate/fsdp2_cards0_3.yaml
git commit -m "feat(phase3a): FSDP2 accelerate config for cards 0-3"
```

---

## Task 5: LF training YAML at 32K (first smoke target)

**Files:**
- Create: `configs/training/lf_qwen35_4b_smoke_32k.yaml`

- [ ] **Step 1: Write the YAML**

```yaml
### model
model_name_or_path: /home/user01/Minko/models/Qwen3.5-4B
trust_remote_code: true
flash_attn: fa2

### method
stage: pt
do_train: true
finetuning_type: full

### dataset
dataset: longluxi_packed_256k
cutoff_len: 32768
max_samples: 30          # all 30 packed sequences
preprocessing_num_workers: 8
dataloader_num_workers: 2
packing: false           # already packed upstream
neat_packing: false

### output
output_dir: checkpoints/phase3a_smoke_32k
logging_steps: 1
save_steps: 3
save_total_limit: 1
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
max_steps: 3
learning_rate: 5.0e-6
warmup_ratio: 0.0
lr_scheduler_type: constant
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
```

Rationale for the values:
- `cutoff_len: 32768` — first smoke ctx; small enough that OOM here means our FSDP config itself is broken.
- `per_device_train_batch_size: 1` × 4 cards × `gradient_accumulation_steps: 1` = effective batch of 4 sequences/step.
- `max_steps: 3` + `save_steps: 3` — train exactly 3 steps and save once.
- `learning_rate: 5.0e-6` — small enough that loss doesn't spike; we only care that the loop runs.
- `lr_scheduler_type: constant` + `warmup_ratio: 0` — avoid warmup math in a 3-step run.
- `gradient_checkpointing: true` — required at 32K for activation memory.
- `flash_attn: fa2` — wire flash-attn 2 into LF's attention path.
- `packing: false` — our sequences are already packed; LF must NOT re-pack.

- [ ] **Step 2: Commit**

```bash
git add configs/training/lf_qwen35_4b_smoke_32k.yaml
git commit -m "feat(phase3a): LF stage_a smoke YAML at 32K ctx"
```

---

## Task 6: Smoke run @ 32K — verify FSDP2 actually shards

**Files:**
- Create: `scripts/launch_lf_smoke.sh`
- Create: `scripts/probe_gpu_mem.sh`

- [ ] **Step 1: Write launch wrapper**

```bash
#!/usr/bin/env bash
# Launch LF pt training under FSDP2 on cards 0-3.
# Usage: bash scripts/launch_lf_smoke.sh configs/training/lf_qwen35_4b_smoke_32k.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="${1:?usage: $0 <training.yaml>}"
ACC_CFG="${ACC_CFG:-configs/accelerate/fsdp2_cards0_3.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VIRTUAL_ENV="$(pwd)/.venv-lf"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

echo "[launch_lf] cfg=$CFG acc=$ACC_CFG gpus=$CUDA_VISIBLE_DEVICES"

exec accelerate launch \
    --config_file "$ACC_CFG" \
    --num_processes 4 \
    -m llamafactory.cli train "$CFG"
```

- [ ] **Step 2: Write GPU memory probe**

```bash
#!/usr/bin/env bash
# Poll nvidia-smi every 2s for `${DURATION:-120}` seconds, log per-card used MB.
# Usage: DURATION=120 bash scripts/probe_gpu_mem.sh > /tmp/gpu_probe.csv
set -euo pipefail
DURATION="${DURATION:-120}"
INTERVAL="${INTERVAL:-2}"
END=$(( $(date +%s) + DURATION ))
echo "timestamp,gpu0_mb,gpu1_mb,gpu2_mb,gpu3_mb"
while [[ $(date +%s) -lt $END ]]; do
  ts=$(date +%s)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 | paste -sd,)
  echo "$ts,$mem"
  sleep "$INTERVAL"
done
```

- [ ] **Step 3: Run the 32K smoke with memory probe in parallel**

```bash
chmod +x scripts/launch_lf_smoke.sh scripts/probe_gpu_mem.sh
DURATION=600 bash scripts/probe_gpu_mem.sh > /tmp/gpu_probe_32k.csv 2>&1 &
PROBE_PID=$!
bash scripts/launch_lf_smoke.sh configs/training/lf_qwen35_4b_smoke_32k.yaml 2>&1 | tee /tmp/smoke_32k.log
wait $PROBE_PID || true
```

- [ ] **Step 4: Inspect outcomes**

Three things to check, IN ORDER:

1. **Did training complete 3 steps?** Search `/tmp/smoke_32k.log` for `{'loss':` lines.
   Expected: 3 loss lines, each finite (no NaN), values typically 1.5–3.0 range for cold pt step.

2. **Did ckpt save?** Check `checkpoints/phase3a_smoke_32k/checkpoint-3/` exists with `model.safetensors*` files.

3. **Did FSDP actually shard?** Parse `/tmp/gpu_probe_32k.csv` for peak per-card memory:
   ```bash
   python3 -c "
   import csv
   peaks = [0]*4
   for row in csv.DictReader(open('/tmp/gpu_probe_32k.csv')):
       for i in range(4):
           peaks[i] = max(peaks[i], int(row[f'gpu{i}_mb']))
   print('peak MB per card:', peaks)
   print('max:', max(peaks), 'min:', min(peaks))
   "
   ```
   Expected: max per-card peak **< 30000 MB** (~30 GB). If any card hits > 60 GB, FSDP is not sharding optimizer state — same Phase 2 failure mode; STOP and debug before ramping ctx.

- [ ] **Step 5: If FSDP not sharding, debug**

Likely fixes (try in order):
1. Verify `fsdp_version: 2` is being read by accelerate — log lines should mention "FSDP2" or "FullyShardedDataParallel v2".
2. Confirm wrap class — `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer` must be the actual class name. If transformers exposes it as `Qwen3DecoderLayer` instead, add that to the wrap list.
3. Check that LF isn't disabling FSDP. Some LF code paths force DDP if `deepspeed` config is set — verify our YAML has no `deepspeed:` key.

Do NOT proceed to Task 7 until peak per-card < 30 GB.

- [ ] **Step 6: Commit launch scripts**

```bash
git add scripts/launch_lf_smoke.sh scripts/probe_gpu_mem.sh
git commit -m "feat(phase3a): LF training launch wrapper + GPU mem probe"
```

---

## Task 7: Ramp to 64K

**Files:**
- Create: `configs/training/lf_qwen35_4b_smoke_64k.yaml`

- [ ] **Step 1: Write 64K config (copy 32K, change ctx + output dir)**

```yaml
### model
model_name_or_path: /home/user01/Minko/models/Qwen3.5-4B
trust_remote_code: true
flash_attn: fa2

### method
stage: pt
do_train: true
finetuning_type: full

### dataset
dataset: longluxi_packed_256k
cutoff_len: 65536
max_samples: 30
preprocessing_num_workers: 8
dataloader_num_workers: 2
packing: false
neat_packing: false

### output
output_dir: checkpoints/phase3a_smoke_64k
logging_steps: 1
save_steps: 3
save_total_limit: 1
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
max_steps: 3
learning_rate: 5.0e-6
warmup_ratio: 0.0
lr_scheduler_type: constant
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
```

- [ ] **Step 2: Run + probe**

```bash
DURATION=900 bash scripts/probe_gpu_mem.sh > /tmp/gpu_probe_64k.csv 2>&1 &
PROBE_PID=$!
bash scripts/launch_lf_smoke.sh configs/training/lf_qwen35_4b_smoke_64k.yaml 2>&1 | tee /tmp/smoke_64k.log
wait $PROBE_PID || true
```

- [ ] **Step 3: Check (same three checks as Task 6 Step 4)**

Expected peak per-card: < 45 GB at 64K. Three loss lines, ckpt at `checkpoints/phase3a_smoke_64k/checkpoint-3/`.

- [ ] **Step 4: Commit config**

```bash
git add configs/training/lf_qwen35_4b_smoke_64k.yaml
git commit -m "feat(phase3a): LF smoke @ 64K ctx — verifies ctx ramp"
```

---

## Task 8: Ramp to 128K

**Files:**
- Create: `configs/training/lf_qwen35_4b_smoke_128k.yaml`

- [ ] **Step 1: Write 128K config**

Same as Task 7's YAML, with these two values changed:
- `cutoff_len: 131072`
- `output_dir: checkpoints/phase3a_smoke_128k`

- [ ] **Step 2: Run + probe**

```bash
DURATION=1500 bash scripts/probe_gpu_mem.sh > /tmp/gpu_probe_128k.csv 2>&1 &
PROBE_PID=$!
bash scripts/launch_lf_smoke.sh configs/training/lf_qwen35_4b_smoke_128k.yaml 2>&1 | tee /tmp/smoke_128k.log
wait $PROBE_PID || true
```

- [ ] **Step 3: Check**

Expected peak per-card: < 65 GB at 128K. If we exceed 70 GB, set `fsdp_offload_params: true` in accelerate config and re-run (offload trades speed for memory).

- [ ] **Step 4: Commit**

```bash
git add configs/training/lf_qwen35_4b_smoke_128k.yaml
git commit -m "feat(phase3a): LF smoke @ 128K ctx"
```

---

## Task 9: Ramp to 256K (headline target)

**Files:**
- Create: `configs/training/lf_qwen35_4b_smoke_256k.yaml`

- [ ] **Step 1: Write 256K config**

Same as Task 8's YAML, with:
- `cutoff_len: 262144`
- `output_dir: checkpoints/phase3a_smoke_256k`
- `save_steps: 1` (save after first step in case OOM hits later steps)

- [ ] **Step 2: Run + probe**

```bash
DURATION=3000 bash scripts/probe_gpu_mem.sh > /tmp/gpu_probe_256k.csv 2>&1 &
PROBE_PID=$!
bash scripts/launch_lf_smoke.sh configs/training/lf_qwen35_4b_smoke_256k.yaml 2>&1 | tee /tmp/smoke_256k.log
wait $PROBE_PID || true
```

- [ ] **Step 3: Check**

Expected peak per-card: < 78 GB at 256K. This is tight; if it OOMs:
1. Try `fsdp_offload_params: true` in accelerate config.
2. If still OOM, this is a real signal that Phase 3b will need sequence parallelism (Ulysses) for 256K+. Note the finding in PHASE3A_REPORT and stop at the largest ctx that works.

- [ ] **Step 4: Confirm ckpt saved**

`ls -la checkpoints/phase3a_smoke_256k/checkpoint-1/` should show `model.safetensors*`, `config.json`, tokenizer files.

- [ ] **Step 5: Commit**

```bash
git add configs/training/lf_qwen35_4b_smoke_256k.yaml
git commit -m "feat(phase3a): LF smoke @ 256K ctx — headline Phase 3a target"
```

---

## Task 10: Round-trip eval via vLLM

The 256K smoke ckpt must serve via vLLM and not regress on RULER 128K. We switch back to the main venv for eval.

**Files:**
- No new files. Reuse `eval/runners/vllm_server.sh` and `eval/runners/run_via_vllm.py`.

- [ ] **Step 1: Stop any running vLLM**

```bash
pkill -f "vllm serve" 2>/dev/null || true
sleep 3
```

- [ ] **Step 2: Verify ckpt is HF-format complete**

```bash
ls -la checkpoints/phase3a_smoke_256k/checkpoint-1/
# Expected: model.safetensors* shards, config.json, generation_config.json, tokenizer*, special_tokens_map.json
```

If `config.json` is missing the YaRN rope_scaling section, copy it from the source model:

```bash
python3 -c "
import json
src = json.load(open('/home/user01/Minko/models/Qwen3.5-4B/config.json'))
dst_path = 'checkpoints/phase3a_smoke_256k/checkpoint-1/config.json'
dst = json.load(open(dst_path))
# Patch YaRN if missing.
tc = dst.get('text_config', dst)
if 'rope_parameters' not in tc and 'rope_scaling' not in tc:
    src_tc = src.get('text_config', src)
    rope = src_tc.get('rope_parameters') or src_tc.get('rope_scaling')
    tc['rope_parameters'] = rope
    json.dump(dst, open(dst_path, 'w'), indent=2)
    print('patched YaRN into', dst_path)
else:
    print('YaRN already present in ckpt config')
"
```

- [ ] **Step 3: Launch vLLM against the smoke ckpt**

```bash
MODEL="$(pwd)/checkpoints/phase3a_smoke_256k/checkpoint-1" \
PORT=8001 \
MAX_LEN=131072 \
bash eval/runners/vllm_server.sh > /tmp/vllm_phase3a.log 2>&1 &
VLLM_PID=$!

# Wait for ready (poll /v1/models).
for i in {1..60}; do
  if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "vLLM ready after ${i}0s"
    break
  fi
  sleep 10
done
curl -s http://localhost:8001/v1/models | python3 -m json.tool
```

Expected: model entry with the ckpt path; no errors in `/tmp/vllm_phase3a.log` about config or weight mismatch.

- [ ] **Step 4: Run RULER 128K via vLLM**

```bash
uv run python eval/runners/run_via_vllm.py \
    --task ruler --ctx 131072 --port 8001 \
    --out-dir eval/results/phase3a_ruler_128k 2>&1 | tee /tmp/ruler_phase3a.log
```

Expected: overall ≥ 84% (phase1-baseline), vt ≥ 20%. Loss of more than 2 points on overall is a regression — investigate before tagging.

- [ ] **Step 5: Run NIAH 32K + 128K**

```bash
uv run python eval/runners/run_via_vllm.py \
    --task niah --ctx 32768 --port 8001 \
    --out-dir eval/results/phase3a_niah_32k

uv run python eval/runners/run_via_vllm.py \
    --task niah --ctx 131072 --port 8001 \
    --out-dir eval/results/phase3a_niah_128k
```

Expected: 100% / 100% (matches phase1-baseline).

- [ ] **Step 6: Stop vLLM**

```bash
kill $VLLM_PID 2>/dev/null || pkill -f "vllm serve" 2>/dev/null || true
```

- [ ] **Step 7: Commit eval results**

```bash
git add eval/results/phase3a_*/
git commit -m "test(phase3a): ckpt round-trip — RULER 128K + NIAH 32K/128K via vLLM"
```

---

## Task 11: Phase 3a close-out report + tag

**Files:**
- Create: `docs/reports/PHASE3A_REPORT.md`

- [ ] **Step 1: Gather numbers**

```bash
# Pull peak GPU memory at each ctx step.
for ctx in 32k 64k 128k 256k; do
  if [[ -f /tmp/gpu_probe_${ctx}.csv ]]; then
    python3 -c "
import csv
peaks=[0]*4
for r in csv.DictReader(open('/tmp/gpu_probe_${ctx}.csv')):
    for i in range(4): peaks[i]=max(peaks[i], int(r[f'gpu{i}_mb']))
print(f'${ctx}: peak per-card MB =', peaks, 'max:', max(peaks))
"
  fi
done

# Pull eval numbers from phase3a results.
ls eval/results/phase3a_*/
```

- [ ] **Step 2: Write the report**

Template (fill numbers from Step 1):

```markdown
# Phase 3a Report — LLaMA-Factory FSDP2 Smoke Through 256K

| Field | Value |
|---|---|
| Date | 2026-05-XX |
| Tag | `phase3a-lf-smoke` |
| Hardware | 4× H100 80GB (cards 0-3) |
| Training | LF pt stage + accelerate FSDP2 (TRANSFORMER_BASED_WRAP) |
| Eval | vLLM TP=4 (existing main-venv stack) |
| Data | longluxi_packed_256k (30 × ~260K tokens) |

## What Phase 3a delivered
1. Isolated LF venv (`.venv-lf/`) at torch 2.8 + transformers ≤ 5.6 + accelerate ≤ 1.11 + flash-attn 2.8.3.
2. `longluxi_packed_256k` registered as LF pt dataset via symlink + `dataset_info.json` entry.
3. FSDP2 config for cards 0-3 (`configs/accelerate/fsdp2_cards0_3.yaml`).
4. Four LF training YAMLs (32K, 64K, 128K, 256K), each producing a 3-step ckpt.
5. **FSDP2 truly shards** — peak per-card memory at each ctx:
   - 32K: <FILL> MB
   - 64K: <FILL> MB
   - 128K: <FILL> MB
   - 256K: <FILL> MB
6. 256K ckpt round-trips through vLLM with no eval regression:
   - RULER 128K overall: <FILL>% (baseline 84%)
   - RULER 128K vt: <FILL>% (baseline 20%)
   - NIAH 32K: <FILL>% / NIAH 128K: <FILL>%

## Closes the Phase 2 blocker
Phase 2 documented "accelerate FSDP doesn't shard cleanly with HF Qwen3.5 + accelerate." Phase 3a confirms: with FSDP2 (`fsdp_version: 2`) + the correct wrap class (`Qwen3_5DecoderLayer`) wired through LF, sharding works. The Phase 2 failure was the FSDP1 wrap policy, not the model.

## Phase 3b prerequisites
1. Stage A main 1M CPT — same stack, ramp ctx to 512K then 1M via YaRN. Estimated training compute: 100M–250M tokens.
2. Beyond 256K may need `fsdp_offload_params: true` if peak per-card > 70 GB at 512K.
3. For 2M (Phase 4), evaluate DeepSpeed Ulysses or Ring-Attention sequence parallelism — FSDP2 alone won't fit 2M activations on 4 H100s.
```

- [ ] **Step 3: Commit + tag**

```bash
git add docs/reports/PHASE3A_REPORT.md
git commit -m "docs(phase3a): close-out report — FSDP2 smoke through 256K verified"
git tag phase3a-lf-smoke
git push origin main --tags
```

---

## Self-Review Checklist (do not skip)

After completing all tasks, the implementing agent runs this checklist:

1. **All four smoke ctx levels (32K, 64K, 128K, 256K) completed with peak per-card memory recorded.** If 256K OOMed, the report must say so and stop there — do NOT skip ctx levels and claim 256K worked.
2. **Eval round-trip ran on the 256K ckpt** (or largest working ctx) — RULER 128K overall ≥ 82% (within 2 points of baseline), vt ≥ 18%. Wider deviation = regression = investigate.
3. **vLLM venv was NOT modified** during Phase 3a. `pip list` in `.venv/` matches what `PHASE2_REPORT.md` documented.
4. **No accelerate FSDP1 references slipped into configs** — all `fsdp_version: 2` everywhere.
5. **`packing: false` set in every LF training YAML** — our jsonl is already packed.

If any item fails: fix and re-run the affected task; do NOT tag `phase3a-lf-smoke` until clean.
