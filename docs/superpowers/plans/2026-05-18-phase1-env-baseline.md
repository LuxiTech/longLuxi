# Phase 1 — Environment + Baseline Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish reproducible baseline numbers for Qwen3.5-4B-Instruct on NIAH (32K / 128K / 1M) and RULER 128K, with a reusable eval harness in `src/longluxi/eval/` that all later phases extend. Lock in the YaRN config application path. Deliver `BASELINE_W1.md`.

**Architecture:** Refactor the monolithic `scripts/baseline_eval.py` into focused modules (`haystack`, `niah`, `model_loader`, `results`, `runner`) under `src/longluxi/eval/`. Same modules back the production `eval/runners/run_niah.py` and `run_ruler.py`. CLI scripts stay thin. RULER tasks loaded from upstream `NVIDIA/RULER` repo (clone into `external/`). Real-model eval gated behind GPU; everything else unit-tested without GPU.

**Tech Stack:** Python 3.11, PyTorch 2.4+, Transformers (HF), FlashAttention 2, Qwen3.5-4B-Instruct, HF `datasets` (PG-19 haystack), pytest, uv.

---

## File Structure

**New files:**
- `src/longluxi/eval/__init__.py`
- `src/longluxi/eval/haystack.py` — haystack corpus loaders (PG-19 + offline fallback)
- `src/longluxi/eval/niah.py` — NIAH cell construction + grading
- `src/longluxi/eval/model_loader.py` — model+tokenizer load with YaRN config
- `src/longluxi/eval/results.py` — metrics.json / predictions.jsonl / summary.md writer
- `src/longluxi/eval/runner.py` — generic grid evaluation loop
- `tests/test_haystack.py`
- `tests/test_niah.py`
- `tests/test_model_loader.py`
- `tests/test_results.py`
- `eval/runners/_shared.py` — shared CLI helpers (parse_len, output dir naming)
- `docs/reports/BASELINE_W1.md` — final report (Task 14)

**Modified files:**
- `scripts/baseline_eval.py` — strip body, wire to `src/longluxi/eval/`
- `eval/runners/run_niah.py` — implement using `src/longluxi/eval/`
- `eval/runners/run_ruler.py` — implement using `src/longluxi/eval/`
- `pyproject.toml` — already lists deps; no change unless missing pulls discovered
- `README.md` — add baseline-report row to milestones table after Task 14

**External clones (gitignored, via Makefile):**
- `external/flame`
- `external/flash-linear-attention`
- `external/LLaMA-Factory`
- `external/RULER` (NVIDIA/RULER, for Task 12)

---

### Task 1: Sync eval + dev dependencies

**Files:** none (uses existing `pyproject.toml`).

- [ ] **Step 1: Sync the venv with eval + dev extras**

Run from repo root:
```bash
uv sync --extra eval --extra dev
```

Expected: uv resolves a fresh `.venv/` with `torch`, `transformers>=4.45`, `datasets`, `evaluate`, `pytest`, `ruff`, etc. Heavy wheels (`torch`) may take several minutes on first run.

- [ ] **Step 2: Verify imports work**

```bash
uv run python -c "import torch, transformers, datasets; print(torch.__version__, transformers.__version__)"
```

Expected: prints two version strings, exits 0. If `torch` import fails on a machine without CUDA, that's OK for Tasks 1–8 (they don't need GPU), but Tasks 9–13 require a CUDA-capable box. Note this on the executor's environment doc.

- [ ] **Step 3: Re-run existing smoke tests**

```bash
uv run pytest tests/ -q
```

Expected: `8 passed`. No new failures from the dep upgrade.

- [ ] **Step 4: Commit (only if pyproject changed)**

If you had to add a missing dep:
```bash
git add pyproject.toml
git commit -m "deps: add <missing-pkg> to eval extra"
```

Otherwise skip commit — `uv.lock` is in `.gitignore`.

---

### Task 2: Clone external repos (flame, FLA, LLaMA-Factory, RULER)

**Files:**
- Modify: `Makefile` (add `setup-ruler` target)

- [ ] **Step 1: Add RULER setup target to Makefile**

Open `Makefile`. Find the `setup-external: setup-flame setup-llamafactory` line. Replace the entire `setup-*` block with:

```makefile
.PHONY: setup-flame setup-llamafactory setup-ruler setup-external
setup-flame:
	bash training/flame_wrapper/setup_flame.sh

setup-llamafactory:
	bash training/llamafactory_wrapper/setup_llamafactory.sh

setup-ruler:
	mkdir -p external
	[ -d external/RULER ] || git clone --depth 1 https://github.com/NVIDIA/RULER.git external/RULER
	cd external/RULER && git pull --ff-only

setup-external: setup-flame setup-llamafactory setup-ruler
```

Also add a help line near the other `setup-*` lines:
```
	@echo "  setup-ruler           clone NVIDIA/RULER into external/"
```

- [ ] **Step 2: Run setup-external**

```bash
make setup-external
```

Expected: clones `external/flame`, `external/flash-linear-attention`, `external/LLaMA-Factory`, `external/RULER`. Each `git clone --depth 1` is ~50-500MB. No install yet.

- [ ] **Step 3: Verify directories exist**

```bash
ls external/
```

Expected output contains:
```
flame
flash-linear-attention
LLaMA-Factory
RULER
```

- [ ] **Step 4: Commit Makefile change**

```bash
git add Makefile
git commit -m "build: add setup-ruler target for NVIDIA/RULER clone"
```

---

### Task 3: Create empty eval package + smoke import test

**Files:**
- Create: `src/longluxi/eval/__init__.py`
- Create: `tests/test_eval_package.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_eval_package.py`:

```python
"""Phase 1: ensure src/longluxi/eval/ is an importable package."""
import importlib


def test_eval_package_importable():
    mod = importlib.import_module("longluxi.eval")
    assert mod is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_eval_package.py -v
```

Expected: `ModuleNotFoundError: No module named 'longluxi.eval'`.

- [ ] **Step 3: Create the package init file**

Create `src/longluxi/eval/__init__.py`:

```python
"""Evaluation harness — modules shared by scripts/baseline_eval.py and eval/runners/."""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_eval_package.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/longluxi/eval/__init__.py tests/test_eval_package.py
git commit -m "feat(eval): create longluxi.eval package skeleton"
```

---

### Task 4: Haystack corpus loader (TDD)

**Files:**
- Create: `src/longluxi/eval/haystack.py`
- Create: `tests/test_haystack.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_haystack.py`:

```python
"""Haystack corpus loader tests. No network or HF download required."""
from longluxi.eval import haystack


def test_fallback_corpus_long_enough_for_small_target():
    text = haystack.load_haystack_corpus(target_tokens=2_000, allow_network=False)
    # Heuristic: at ~1.3 tokens/word, 2000 tokens ≈ 1540 words. Fallback repeats PG snippet to fill.
    assert len(text.split()) >= 1500


def test_fallback_corpus_scales_to_request():
    small = haystack.load_haystack_corpus(target_tokens=1_000, allow_network=False)
    big = haystack.load_haystack_corpus(target_tokens=50_000, allow_network=False)
    assert len(big) > len(small) * 10


def test_pg_snippet_constant_present():
    assert "writer" in haystack.PAUL_GRAHAM_FALLBACK.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_haystack.py -v
```

Expected: `ImportError` / `AttributeError: module 'longluxi.eval.haystack' has no attribute 'load_haystack_corpus'`.

- [ ] **Step 3: Implement the module**

Create `src/longluxi/eval/haystack.py`:

```python
"""Haystack corpus loaders for NIAH/RULER.

Production path: stream PG-19 from HF datasets.
Offline fallback: repeat a hard-coded Paul Graham snippet to fill the target.
"""
from __future__ import annotations

PAUL_GRAHAM_FALLBACK = """\
The most surprising thing I've learned from being a writer is how rarely the way I think
something is going to come out matches the way it actually does. Whenever you write
something, you imagine how it'll read. But the imagined version and the actual version
diverge in unpredictable ways. The act of writing forces you to think more rigorously than
you would just imagining; you can't paper over weaknesses with fluent prose when each
sentence sits on the page and has to defend itself.
"""

_APPROX_TOKEN_PER_WORD = 1.3


def load_haystack_corpus(target_tokens: int, allow_network: bool = True) -> str:
    """Return a string with at least ~target_tokens worth of English prose."""
    if allow_network:
        text = _try_pg19(target_tokens)
        if text is not None:
            return text
    return _fallback_fill(target_tokens)


def _try_pg19(target_tokens: int) -> str | None:
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
    except Exception:  # noqa: BLE001
        return None
    chunks: list[str] = []
    words = 0
    for ex in ds:
        txt = ex.get("text", "") or ""
        if not txt:
            continue
        chunks.append(txt)
        words += len(txt.split())
        if words * _APPROX_TOKEN_PER_WORD > target_tokens * 1.2:
            break
    return "\n\n".join(chunks) if chunks else None


def _fallback_fill(target_tokens: int) -> str:
    target_words = max(1, int(target_tokens / _APPROX_TOKEN_PER_WORD))
    snippet = PAUL_GRAHAM_FALLBACK
    snippet_words = len(snippet.split())
    n_repeat = max(1, (target_words // snippet_words) + 2)
    return (snippet + "\n") * n_repeat
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_haystack.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/longluxi/eval/haystack.py tests/test_haystack.py
git commit -m "feat(eval): haystack corpus loader with PG-19 + offline fallback"
```

---

### Task 5: NIAH cell construction + grader (TDD)

**Files:**
- Create: `src/longluxi/eval/niah.py`
- Create: `tests/test_niah.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_niah.py`:

```python
"""NIAH cell construction + grading tests. Uses a fake whitespace tokenizer (no HF dep)."""
import random

from longluxi.eval.niah import NiahCell, build_cell, grade


class _WhitespaceTokenizer:
    """Minimal tokenizer for unit tests: split on whitespace, no special tokens."""
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


def test_build_cell_inserts_needle_at_depth():
    tok = _WhitespaceTokenizer()
    hay = " ".join(["word"] * 1000)
    cell = build_cell(hay, length_tokens=500, depth_pct=50, tokenizer=tok,
                      rng=random.Random(0), margin_tokens=50)
    assert isinstance(cell, NiahCell)
    assert cell.expected.isdigit()
    assert "magic number" in cell.haystack.lower()
    # The needle should sit near the midpoint of the haystack region.
    words = cell.haystack.split()
    midword_idx = words.index("magic")
    # depth_pct=50 of (length_tokens - margin) = 50% of 450 = 225-ish; allow ±50.
    assert 175 <= midword_idx <= 275


def test_grade_substring_match():
    cell = NiahCell(length_tokens=1, depth_pct=50, needle="The magic number is 12345.",
                    expected="12345", haystack="x", prompt="x")
    assert grade(cell, "12345") is True
    assert grade(cell, "The number is 12345 indeed.") is True
    assert grade(cell, "98765") is False
    assert grade(cell, "") is False


def test_build_cell_deterministic_with_seed():
    tok = _WhitespaceTokenizer()
    hay = " ".join(["word"] * 1000)
    c1 = build_cell(hay, 500, 50, tok, random.Random(42), margin_tokens=50)
    c2 = build_cell(hay, 500, 50, tok, random.Random(42), margin_tokens=50)
    assert c1.expected == c2.expected
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_niah.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the module**

Create `src/longluxi/eval/niah.py`:

```python
"""NIAH cell construction + grading.

A "cell" = one (length, depth) evaluation instance: haystack with a needle
inserted at a specific depth percentage, plus prompt + expected answer.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class NiahCell:
    length_tokens: int
    depth_pct: int
    needle: str
    expected: str       # the secret number, used for grading
    haystack: str       # haystack including needle
    prompt: str         # full prompt sent to model


_PROMPT_TMPL = (
    "You are a helpful assistant. Read the text and answer the question precisely.\n\n"
    "<text>\n{haystack}\n</text>\n\n"
    "Question: What is the magic number mentioned in the text?\n"
    "Answer with just the number, nothing else."
)


def build_cell(haystack_text: str, length_tokens: int, depth_pct: int,
               tokenizer, rng: random.Random, margin_tokens: int = 200) -> NiahCell:
    """Embed a random 10-digit number at depth_pct of the haystack and build the prompt."""
    secret = f"{rng.randint(10**9, 10**10 - 1)}"
    needle = f"The magic number is {secret}."

    target = max(length_tokens - margin_tokens, 64)
    hay_ids = tokenizer(haystack_text, return_tensors=None, add_special_tokens=False)["input_ids"]
    if len(hay_ids) < target:
        rep = (target // max(len(hay_ids), 1)) + 1
        hay_ids = hay_ids * rep
    hay_ids = hay_ids[:target]

    needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]
    insert_pos = int(len(hay_ids) * depth_pct / 100)
    full_ids = list(hay_ids[:insert_pos]) + list(needle_ids) + list(hay_ids[insert_pos:])
    haystack = tokenizer.decode(full_ids)

    prompt = _PROMPT_TMPL.format(haystack=haystack)
    return NiahCell(
        length_tokens=length_tokens,
        depth_pct=depth_pct,
        needle=needle,
        expected=secret,
        haystack=haystack,
        prompt=prompt,
    )


def grade(cell: NiahCell, prediction: str) -> bool:
    """Pass iff the secret string appears anywhere in the prediction."""
    if not prediction:
        return False
    return cell.expected in prediction
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_niah.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/longluxi/eval/niah.py tests/test_niah.py
git commit -m "feat(eval): NIAH cell builder + substring grader"
```

---

### Task 6: Model loader with YaRN config application (TDD)

**Files:**
- Create: `src/longluxi/eval/model_loader.py`
- Create: `tests/test_model_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model_loader.py`:

```python
"""Model-loader unit tests. Mock the HF config object — no real model load."""
import json
from pathlib import Path
from types import SimpleNamespace

from longluxi.eval.model_loader import apply_yarn_to_config, load_yarn_json


def test_load_yarn_json_returns_rope_params(tmp_path):
    p = tmp_path / "y.json"
    p.write_text(json.dumps({
        "rope_parameters": {
            "rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 262144,
            "mrope_interleaved": True, "mrope_section": [11, 11, 10],
            "rope_theta": 10000000, "partial_rotary_factor": 0.25,
        }
    }))
    rp = load_yarn_json(p)
    assert rp["rope_type"] == "yarn"
    assert rp["factor"] == 4.0


def test_apply_yarn_to_config_with_rope_parameters_attr():
    cfg = SimpleNamespace(rope_parameters={"placeholder": True})
    rp = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}
    apply_yarn_to_config(cfg, rp)
    assert cfg.rope_parameters == rp


def test_apply_yarn_to_config_falls_back_to_rope_scaling():
    cfg = SimpleNamespace()  # no rope_parameters attr
    rp = {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 262144}
    apply_yarn_to_config(cfg, rp)
    assert cfg.rope_scaling["factor"] == 8.0
    assert cfg.rope_scaling["original_max_position_embeddings"] == 262144


def test_repo_yarn_configs_loadable():
    repo_root = Path(__file__).resolve().parent.parent
    for name in ("yarn_1m.json", "yarn_2m.json", "yarn_4m.json"):
        rp = load_yarn_json(repo_root / "configs" / "yarn" / name)
        assert rp["rope_type"] == "yarn"
        assert rp["original_max_position_embeddings"] == 262144
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_model_loader.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the module**

Create `src/longluxi/eval/model_loader.py`:

```python
"""Load HF causal LM + tokenizer with optional YaRN config override.

Heavy imports (torch, transformers) are gated to keep --dry-run lightweight.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yarn_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    return data["rope_parameters"]


def apply_yarn_to_config(model_config, rope_params: dict[str, Any]) -> None:
    """Patch a HF AutoConfig in place.

    Qwen3.5 uses `rope_parameters` (multi-RoPE);
    older HF models use `rope_scaling`.
    """
    if hasattr(model_config, "rope_parameters"):
        model_config.rope_parameters = rope_params
    else:
        model_config.rope_scaling = {
            "rope_type": rope_params["rope_type"],
            "factor": rope_params["factor"],
            "original_max_position_embeddings": rope_params["original_max_position_embeddings"],
        }


def load_model_and_tokenizer(model_id: str, yarn_path: str | Path | None = None,
                             dtype: str = "bfloat16", attn_impl: str = "flash_attention_2"):
    """Heavy path: actually load model. Requires torch+transformers+flash-attn.

    Returns (model, tokenizer, config).
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if yarn_path:
        apply_yarn_to_config(config, load_yarn_json(yarn_path))

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, config
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_model_loader.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/longluxi/eval/model_loader.py tests/test_model_loader.py
git commit -m "feat(eval): model loader with YaRN config patching"
```

---

### Task 7: Results writer (TDD)

**Files:**
- Create: `src/longluxi/eval/results.py`
- Create: `tests/test_results.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_results.py`:

```python
"""Results writer tests — verify metrics.json / predictions.jsonl / summary.md emitted."""
import json
from pathlib import Path

from longluxi.eval.results import write_results


def test_write_results_creates_all_three_files(tmp_path):
    preds = [
        {"length_tokens": 1024, "depth_pct": 50, "expected": "12345", "pred": "12345", "ok": True, "input_tokens": 1000},
        {"length_tokens": 1024, "depth_pct": 50, "expected": "67890", "pred": "wrong", "ok": False, "input_tokens": 1000},
        {"length_tokens": 8192, "depth_pct": 10, "expected": "11111", "pred": "11111", "ok": True, "input_tokens": 8000},
    ]
    out = write_results(tmp_path, model_id="m/x", yarn_config="y.json", predictions=preds)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "predictions.jsonl").exists()
    assert (tmp_path / "summary.md").exists()
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["accuracy"] == 2 / 3
    assert m["by_length"]["1024"] == 0.5
    assert m["by_length"]["8192"] == 1.0
    assert out["accuracy"] == 2 / 3


def test_write_results_empty_predictions(tmp_path):
    out = write_results(tmp_path, model_id="m/x", yarn_config=None, predictions=[])
    assert out["accuracy"] == 0.0
    assert (tmp_path / "metrics.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_results.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the module**

Create `src/longluxi/eval/results.py`:

```python
"""Write metrics.json + predictions.jsonl + summary.md for an eval run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_results(out_dir: Path, *, model_id: str, yarn_config: str | None,
                  predictions: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(predictions)
    correct = sum(1 for p in predictions if p.get("ok"))
    acc = correct / n if n else 0.0

    by_len: dict[int, list[bool]] = {}
    for p in predictions:
        by_len.setdefault(int(p["length_tokens"]), []).append(bool(p["ok"]))

    by_len_acc = {str(L): (sum(oks) / len(oks)) for L, oks in by_len.items()}

    metrics = {
        "model_id": model_id,
        "yarn_config": yarn_config,
        "n_cells": n,
        "accuracy": acc,
        "by_length": by_len_acc,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with (out_dir / "predictions.jsonl").open("w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    lines = [f"# NIAH Results — {model_id}", ""]
    lines.append(f"- Accuracy: **{acc:.2%}** ({correct}/{n})")
    lines.append(f"- YaRN: `{yarn_config}`" if yarn_config else "- YaRN: none")
    lines.append("")
    if by_len:
        lines.append("## By length")
        for L in sorted(by_len.keys()):
            oks = by_len[L]
            lines.append(f"- {L:>10} tokens — {sum(oks)/len(oks):.2%} ({sum(oks)}/{len(oks)})")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_results.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/longluxi/eval/results.py tests/test_results.py
git commit -m "feat(eval): results writer (metrics.json + predictions.jsonl + summary.md)"
```

---

### Task 8: Generic grid runner + rewire `scripts/baseline_eval.py`

**Files:**
- Create: `src/longluxi/eval/runner.py`
- Modify: `scripts/baseline_eval.py` (full rewrite, thinner)

- [ ] **Step 1: Implement the runner**

Create `src/longluxi/eval/runner.py`:

```python
"""Generic grid runner for NIAH-style evals."""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from longluxi.eval.haystack import load_haystack_corpus
from longluxi.eval.niah import NiahCell, build_cell, grade
from longluxi.eval.results import write_results


def build_grid(lengths: list[int], depths: list[int], n_per_cell: int,
               tokenizer, seed: int = 42, max_cells: int | None = None) -> list[NiahCell]:
    rng = random.Random(seed)
    hay = load_haystack_corpus(max(lengths), allow_network=True)
    cells: list[NiahCell] = []
    for L in lengths:
        for d in depths:
            for _ in range(n_per_cell):
                cells.append(build_cell(hay, L, d, tokenizer, rng))
            if max_cells and len(cells) >= max_cells:
                return cells[:max_cells]
    return cells


def evaluate_cells(cells: list[NiahCell], model, tokenizer, max_new_tokens: int = 16,
                   verbose: bool = True) -> list[dict[str, Any]]:
    """Iterate cells, run greedy decode, return prediction records."""
    import torch  # heavy

    predictions: list[dict[str, Any]] = []
    for i, cell in enumerate(cells):
        inputs = tokenizer(cell.prompt, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        pred = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        ok = grade(cell, pred)
        predictions.append({
            "length_tokens": cell.length_tokens,
            "depth_pct": cell.depth_pct,
            "expected": cell.expected,
            "pred": pred,
            "ok": ok,
            "input_tokens": n_in,
        })
        if verbose:
            print(f"[eval] cell {i+1}/{len(cells)} L={cell.length_tokens} d={cell.depth_pct}% "
                  f"expected={cell.expected} pred={pred!r} ok={ok}", flush=True)
    return predictions


def run_niah(model_id: str, lengths: list[int], depths: list[int], n_per_cell: int,
             out_dir: Path, yarn_path: str | None = None, seed: int = 42,
             max_cells: int | None = None) -> dict[str, Any]:
    from longluxi.eval.model_loader import load_model_and_tokenizer

    model, tokenizer, _cfg = load_model_and_tokenizer(model_id, yarn_path)
    cells = build_grid(lengths, depths, n_per_cell, tokenizer, seed=seed, max_cells=max_cells)
    preds = evaluate_cells(cells, model, tokenizer)
    return write_results(out_dir, model_id=model_id, yarn_config=str(yarn_path) if yarn_path else None,
                         predictions=preds)
```

- [ ] **Step 2: Rewrite scripts/baseline_eval.py as thin CLI**

Open `scripts/baseline_eval.py` and replace its full contents with:

```python
"""W1 baseline NIAH eval — thin CLI wrapping longluxi.eval.runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from longluxi.paths import EVAL_OUTPUTS_DIR  # noqa: E402


def parse_len(s: str) -> int:
    s = s.strip().lower()
    mult = 1
    if s.endswith("k"):
        mult, s = 1024, s[:-1]
    elif s.endswith("m"):
        mult, s = 1024 * 1024, s[:-1]
    return int(float(s) * mult)


def _hash(s: str, n: int = 8) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B-Instruct")
    p.add_argument("--task", choices=["niah"], default="niah")
    p.add_argument("--max-len", type=parse_len, default=131072)
    p.add_argument("--lengths", nargs="+", type=parse_len, default=None)
    p.add_argument("--depths", nargs="+", type=int, default=[10, 50, 90])
    p.add_argument("--n-per-cell", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--yarn-config", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    lengths = args.lengths or [args.max_len]
    out_dir = EVAL_OUTPUTS_DIR / "baseline" / f"{_hash(args.model_id)}_{args.task}_{args.max_len}"

    plan = {
        "model_id": args.model_id, "task": args.task, "max_len": args.max_len,
        "yarn_config": args.yarn_config, "lengths": lengths, "depths": args.depths,
        "n_per_cell": args.n_per_cell, "limit": args.limit, "out_dir": str(out_dir),
    }
    print(json.dumps({"plan": plan}, indent=2, ensure_ascii=False))
    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    yarn_path = (REPO_ROOT / args.yarn_config) if (args.yarn_config and not Path(args.yarn_config).is_absolute()) else args.yarn_config

    from longluxi.eval.runner import run_niah
    metrics = run_niah(
        model_id=args.model_id,
        lengths=lengths,
        depths=args.depths,
        n_per_cell=args.n_per_cell,
        out_dir=out_dir,
        yarn_path=yarn_path,
        seed=args.seed,
        max_cells=args.limit,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify dry-run + existing tests still pass**

```bash
uv run python scripts/baseline_eval.py --task niah --max-len 131072 --limit 1 --dry-run
uv run pytest tests/ -q
```

Expected: dry-run prints `plan` JSON and exits 0; all tests pass (existing baseline-dry-run test in `tests/test_smoke.py` continues to work).

- [ ] **Step 4: Commit**

```bash
git add src/longluxi/eval/runner.py scripts/baseline_eval.py
git commit -m "refactor(eval): rewire baseline_eval.py over longluxi.eval modules"
```

---

### Task 9: Pull-through smoke at 4K tokens, 1 cell, real model

**Files:** none new; this is a real-model smoke run that requires a CUDA box.

**Pre-requisite check:**
```bash
nvidia-smi -L 2>&1 | head -1
```
Expected: at least one H100. If none, defer Tasks 9–13 to a GPU box and finish 1–8 / 14 on CPU.

- [ ] **Step 1: Run a tiny 1-cell smoke**

```bash
uv run python scripts/baseline_eval.py \
    --model-id Qwen/Qwen3.5-4B-Instruct \
    --task niah \
    --max-len 4096 \
    --depths 50 \
    --n-per-cell 1 \
    --limit 1
```

Expected behavior:
- First run downloads the model (~8GB) into HF cache. May take 5-10 min.
- Loads with FlashAttention 2 on a single GPU (device_map="auto").
- Builds one NIAH cell at 4K tokens, runs generate.
- Prints a single cell line and final metrics JSON.
- Accuracy should be **1.0** — Qwen3.5-4B has no trouble retrieving a needle at 4K.

- [ ] **Step 2: Inspect the output dir**

```bash
ls eval/_outputs/baseline/*_niah_4096/
cat eval/_outputs/baseline/*_niah_4096/summary.md
```

Expected: `metrics.json`, `predictions.jsonl`, `summary.md` all present. summary.md shows 100%.

- [ ] **Step 3: No commit** — output artifacts are gitignored.

---

### Task 10: Baseline NIAH @ 32K and 128K native

**Files:** none new; produces eval artifacts.

- [ ] **Step 1: Run NIAH @ 32K, depths 10/30/50/70/90, 2/cell**

```bash
uv run python scripts/baseline_eval.py \
    --model-id Qwen/Qwen3.5-4B-Instruct \
    --task niah \
    --lengths 32768 \
    --depths 10 30 50 70 90 \
    --n-per-cell 2
```

Expected: 10 cells, accuracy near 1.0. Output dir: `eval/_outputs/baseline/<hash>_niah_32768/`.

- [ ] **Step 2: Run NIAH @ 128K, same depths/budget**

```bash
uv run python scripts/baseline_eval.py \
    --model-id Qwen/Qwen3.5-4B-Instruct \
    --task niah \
    --lengths 131072 \
    --depths 10 30 50 70 90 \
    --n-per-cell 2
```

Expected: 10 cells. Accuracy still high (Qwen3.5 is native 256K). Output dir: `<hash>_niah_131072/`.

- [ ] **Step 3: Stash these two reports under a stable name**

```bash
mkdir -p docs/reports/phase1_artifacts
cp eval/_outputs/baseline/*_niah_32768/summary.md docs/reports/phase1_artifacts/niah_32768_native.md
cp eval/_outputs/baseline/*_niah_131072/summary.md docs/reports/phase1_artifacts/niah_131072_native.md
cp eval/_outputs/baseline/*_niah_32768/metrics.json docs/reports/phase1_artifacts/niah_32768_native.json
cp eval/_outputs/baseline/*_niah_131072/metrics.json docs/reports/phase1_artifacts/niah_131072_native.json
```

- [ ] **Step 4: Commit the stashed artifacts**

```bash
git add docs/reports/phase1_artifacts/
git commit -m "eval(phase1): baseline NIAH @ 32K + 128K native (Qwen3.5-4B-Instruct)"
```

---

### Task 11: Baseline NIAH @ 1M with YaRN factor=4

**Files:** none new.

- [ ] **Step 1: Run NIAH @ 1M with static YaRN config**

```bash
uv run python scripts/baseline_eval.py \
    --model-id Qwen/Qwen3.5-4B-Instruct \
    --task niah \
    --lengths 1048576 \
    --depths 10 30 50 70 90 \
    --n-per-cell 2 \
    --yarn-config configs/yarn/yarn_1m.json
```

Expected:
- Model load uses `apply_yarn_to_config` to patch `rope_parameters.factor=4.0`.
- 10 cells at 1M tokens each. Each cell load can take ~30s of compute on a single H100.
- Accuracy should be **lower than native** — somewhere in 0.4–0.8 is typical for static YaRN at 4× extrapolation. This is the quality gap CPT should close.

- [ ] **Step 2: Stash report**

```bash
cp eval/_outputs/baseline/*_niah_1048576/summary.md docs/reports/phase1_artifacts/niah_1m_yarn.md
cp eval/_outputs/baseline/*_niah_1048576/metrics.json docs/reports/phase1_artifacts/niah_1m_yarn.json
```

- [ ] **Step 3: Commit**

```bash
git add docs/reports/phase1_artifacts/niah_1m_yarn*
git commit -m "eval(phase1): baseline NIAH @ 1M with static YaRN factor=4"
```

---

### Task 12: RULER runner (128K subset)

**Files:**
- Modify: `eval/runners/run_ruler.py`
- Create: `src/longluxi/eval/ruler.py`
- Create: `tests/test_ruler.py`

- [ ] **Step 1: Write a failing test for the task registry**

Create `tests/test_ruler.py`:

```python
"""RULER runner unit tests (no GPU)."""
from longluxi.eval.ruler import TASK_REGISTRY, RulerTask, build_ruler_cell


def test_task_registry_has_canonical_tasks():
    for name in ("niah_single_1", "niah_multikey_1", "niah_multiquery", "vt", "qa_1"):
        assert name in TASK_REGISTRY, f"missing task {name}"
        t = TASK_REGISTRY[name]
        assert isinstance(t, RulerTask)
        assert t.name == name
        assert callable(t.build_fn)
        assert callable(t.grade_fn)


def test_build_niah_single_cell_has_expected_field():
    class _Tok:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": text.split()}

        def decode(self, ids):
            return " ".join(ids)

    import random
    cell = build_ruler_cell("niah_single_1", length_tokens=200, tokenizer=_Tok(),
                              rng=random.Random(0))
    assert cell["task"] == "niah_single_1"
    assert "prompt" in cell
    assert "expected" in cell
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_ruler.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the RULER module**

Create `src/longluxi/eval/ruler.py`:

```python
"""RULER task registry and cell builder.

MVP scope (W1): 5 tasks at 128K to validate harness.
- niah_single_1 (1 needle)
- niah_multikey_1 (1 key, 3 distractors)
- niah_multiquery (4 needles, 1 query at a time)
- vt (variable tracking: foo=bar, bar=42, what is foo?)
- qa_1 (HotpotQA-style on synthetic doc)

Full RULER (13 tasks) handled in Phase 2 once we have ckpts to compare.

Upstream definitions: external/RULER/scripts/data/synthetic/
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from longluxi.eval.haystack import load_haystack_corpus
from longluxi.eval.niah import build_cell as _build_niah, grade as _grade_niah


@dataclass
class RulerTask:
    name: str
    build_fn: Callable[..., dict[str, Any]]
    grade_fn: Callable[[dict[str, Any], str], bool]


# ------------- task implementations -------------

def _build_niah_single(length_tokens, tokenizer, rng):
    haystack = load_haystack_corpus(length_tokens, allow_network=True)
    c = _build_niah(haystack, length_tokens, depth_pct=rng.choice([20, 50, 80]),
                    tokenizer=tokenizer, rng=rng)
    return {"task": "niah_single_1", "prompt": c.prompt, "expected": c.expected,
            "length_tokens": c.length_tokens, "depth_pct": c.depth_pct}


def _grade_niah(cell, pred):
    return cell["expected"] in (pred or "")


def _build_niah_multikey(length_tokens, tokenizer, rng):
    """1 real magic number + 3 distractor magic numbers; ask for THE magic number."""
    haystack = load_haystack_corpus(length_tokens, allow_network=True)
    real = f"{rng.randint(10**9, 10**10 - 1)}"
    distractors = [f"{rng.randint(10**9, 10**10 - 1)}" for _ in range(3)]
    hay_ids = tokenizer(haystack, add_special_tokens=False)["input_ids"]
    # Tokenize the labelled key strings.
    real_line = f" The magic number is {real}. "
    real_ids = tokenizer(real_line, add_special_tokens=False)["input_ids"]
    distract_ids = [tokenizer(f" Note: {d} is a random tracking id. ",
                              add_special_tokens=False)["input_ids"] for d in distractors]
    target = max(length_tokens - 300, 64)
    if len(hay_ids) < target:
        hay_ids = (hay_ids * ((target // len(hay_ids)) + 1))[:target]
    # Insert distractors at 20/40/60 %, real at 80 %.
    positions = sorted([int(target * p) for p in (0.2, 0.4, 0.6, 0.8)])
    inserts = list(zip(positions, distract_ids + [real_ids]))
    out_ids: list = []
    last = 0
    for pos, ids in inserts:
        out_ids.extend(hay_ids[last:pos])
        out_ids.extend(ids)
        last = pos
    out_ids.extend(hay_ids[last:])
    haystack_full = tokenizer.decode(out_ids)
    prompt = (
        "Read the text and answer.\n\n<text>\n" + haystack_full + "\n</text>\n\n"
        "Question: What is THE magic number mentioned in the text? "
        "(Ignore any tracking ids.)\nAnswer with just the number."
    )
    return {"task": "niah_multikey_1", "prompt": prompt, "expected": real,
            "length_tokens": length_tokens}


def _build_niah_multiquery(length_tokens, tokenizer, rng):
    """4 needles, one question asks for one specific needle by label."""
    haystack = load_haystack_corpus(length_tokens, allow_network=True)
    labels = ["alpha", "beta", "gamma", "delta"]
    values = [f"{rng.randint(10**9, 10**10 - 1)}" for _ in labels]
    target_idx = rng.randint(0, 3)
    needles = [tokenizer(f" The {lab} number is {val}. ", add_special_tokens=False)["input_ids"]
               for lab, val in zip(labels, values)]
    hay_ids = tokenizer(haystack, add_special_tokens=False)["input_ids"]
    target = max(length_tokens - 400, 64)
    if len(hay_ids) < target:
        hay_ids = (hay_ids * ((target // len(hay_ids)) + 1))[:target]
    positions = sorted([int(target * p) for p in (0.15, 0.35, 0.55, 0.85)])
    out_ids: list = []
    last = 0
    for pos, ids in zip(positions, needles):
        out_ids.extend(hay_ids[last:pos])
        out_ids.extend(ids)
        last = pos
    out_ids.extend(hay_ids[last:])
    haystack_full = tokenizer.decode(out_ids)
    prompt = (
        "Read the text and answer.\n\n<text>\n" + haystack_full + "\n</text>\n\n"
        f"Question: What is the {labels[target_idx]} number? "
        "Answer with just the number."
    )
    return {"task": "niah_multiquery", "prompt": prompt, "expected": values[target_idx],
            "length_tokens": length_tokens}


def _build_vt(length_tokens, tokenizer, rng):
    """Variable tracking. foo=A; A=B; B=42; what is foo? Expected 42."""
    chain_len = rng.randint(3, 5)
    chain = [f"v{i}" for i in range(chain_len)]
    final = f"{rng.randint(10000, 99999)}"
    assignments = [f"{chain[i]} = {chain[i+1]}." for i in range(chain_len - 1)]
    assignments.append(f"{chain[-1]} = {final}.")
    rng.shuffle(assignments)
    haystack = load_haystack_corpus(length_tokens, allow_network=True)
    hay_words = haystack.split()
    # Sprinkle assignments evenly through haystack words.
    n_words = max(len(hay_words), length_tokens)
    out_parts: list[str] = []
    step = max(1, n_words // (len(assignments) + 1))
    next_assign = iter(assignments)
    for i, w in enumerate(hay_words[:n_words]):
        out_parts.append(w)
        if (i + 1) % step == 0:
            try:
                out_parts.append(next(next_assign))
            except StopIteration:
                pass
    # Append any leftover assignments
    for left in next_assign:
        out_parts.append(left)
    text = " ".join(out_parts)
    prompt = (
        "Read the text and trace the variable chain.\n\n<text>\n" + text + "\n</text>\n\n"
        f"Question: What is the final value of {chain[0]}?\nAnswer with just the value."
    )
    return {"task": "vt", "prompt": prompt, "expected": final,
            "length_tokens": length_tokens}


def _build_qa_1(length_tokens, tokenizer, rng):
    """Single-hop synthetic QA: insert a unique fact, ask about it."""
    subject = f"Subject_{rng.randint(1000, 9999)}"
    fact_val = f"{rng.randint(100, 999)}"
    fact = f" {subject} won the {fact_val}-meter race. "
    haystack = load_haystack_corpus(length_tokens, allow_network=True)
    hay_ids = tokenizer(haystack, add_special_tokens=False)["input_ids"]
    fact_ids = tokenizer(fact, add_special_tokens=False)["input_ids"]
    target = max(length_tokens - len(fact_ids) - 50, 64)
    if len(hay_ids) < target:
        hay_ids = (hay_ids * ((target // len(hay_ids)) + 1))[:target]
    pos = int(len(hay_ids) * rng.choice([0.3, 0.6, 0.85]))
    out_ids = hay_ids[:pos] + fact_ids + hay_ids[pos:]
    text = tokenizer.decode(out_ids)
    prompt = (
        "Read the text and answer.\n\n<text>\n" + text + "\n</text>\n\n"
        f"Question: How many meters did {subject} race? Answer with just the number."
    )
    return {"task": "qa_1", "prompt": prompt, "expected": fact_val,
            "length_tokens": length_tokens}


TASK_REGISTRY: dict[str, RulerTask] = {
    "niah_single_1":    RulerTask("niah_single_1",    _build_niah_single,    _grade_niah),
    "niah_multikey_1":  RulerTask("niah_multikey_1",  _build_niah_multikey,  _grade_niah),
    "niah_multiquery":  RulerTask("niah_multiquery",  _build_niah_multiquery, _grade_niah),
    "vt":               RulerTask("vt",               _build_vt,             _grade_niah),
    "qa_1":             RulerTask("qa_1",             _build_qa_1,           _grade_niah),
}


def build_ruler_cell(task_name: str, length_tokens: int, tokenizer,
                     rng: random.Random) -> dict[str, Any]:
    task = TASK_REGISTRY[task_name]
    return task.build_fn(length_tokens, tokenizer, rng)
```

- [ ] **Step 4: Run unit tests to verify they pass**

```bash
uv run pytest tests/test_ruler.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Implement the runner script**

Open `eval/runners/run_ruler.py` and replace contents with:

```python
"""RULER benchmark runner (MVP subset: 5 tasks)."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from longluxi.eval.results import write_results  # noqa: E402
from longluxi.eval.ruler import TASK_REGISTRY, build_ruler_cell  # noqa: E402


def parse_len(s: str) -> int:
    s = s.strip().lower()
    mult = 1
    if s.endswith("k"):
        mult, s = 1024, s[:-1]
    elif s.endswith("m"):
        mult, s = 1024 * 1024, s[:-1]
    return int(float(s) * mult)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B-Instruct")
    p.add_argument("--lengths", nargs="+", type=parse_len, default=[131072])
    p.add_argument("--tasks", nargs="+", default=list(TASK_REGISTRY.keys()))
    p.add_argument("--n-per-task", type=int, default=10)
    p.add_argument("--yarn-config", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="eval/_outputs/ruler")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir) / f"{hashlib.sha256(args.model_id.encode()).hexdigest()[:8]}"
    plan = {
        "model_id": args.model_id, "lengths": args.lengths, "tasks": args.tasks,
        "n_per_task": args.n_per_task, "yarn_config": args.yarn_config, "out_dir": str(out_dir),
    }
    print(json.dumps({"plan": plan}, indent=2, ensure_ascii=False))
    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
        return

    from longluxi.eval.model_loader import load_model_and_tokenizer
    yarn_path = (REPO_ROOT / args.yarn_config) if (args.yarn_config and not Path(args.yarn_config).is_absolute()) else args.yarn_config
    model, tokenizer, _cfg = load_model_and_tokenizer(args.model_id, yarn_path)

    import torch

    rng = random.Random(args.seed)
    predictions: list[dict] = []
    for L in args.lengths:
        for task_name in args.tasks:
            for _ in range(args.n_per_task):
                cell = build_ruler_cell(task_name, L, tokenizer, rng)
                inputs = tokenizer(cell["prompt"], return_tensors="pt").to(model.device)
                n_in = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                pred = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
                ok = TASK_REGISTRY[task_name].grade_fn(cell, pred)
                predictions.append({
                    "length_tokens": cell["length_tokens"],
                    "depth_pct": cell.get("depth_pct", -1),
                    "task": task_name,
                    "expected": cell["expected"],
                    "pred": pred,
                    "ok": ok,
                    "input_tokens": n_in,
                })
                print(f"[ruler] L={L} task={task_name} ok={ok} expected={cell['expected']} pred={pred!r}", flush=True)

    metrics = write_results(out_dir, model_id=args.model_id,
                            yarn_config=str(yarn_path) if yarn_path else None,
                            predictions=predictions)
    # Add per-task breakdown
    by_task: dict[str, list[bool]] = {}
    for p_ in predictions:
        by_task.setdefault(p_["task"], []).append(p_["ok"])
    metrics["by_task"] = {t: (sum(o) / len(o)) for t, o in by_task.items()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify dry-run**

```bash
uv run python eval/runners/run_ruler.py --model-id Qwen/Qwen3.5-4B-Instruct --lengths 131072 --n-per-task 1 --dry-run
```

Expected: prints plan JSON, exits 0. Creates `eval/_outputs/ruler/<hash>/plan.json`.

- [ ] **Step 7: Run full tests**

```bash
uv run pytest tests/ -q
```

Expected: all green (previous 8 + 1 ruler test = at least 9 pass).

- [ ] **Step 8: Commit**

```bash
git add src/longluxi/eval/ruler.py eval/runners/run_ruler.py tests/test_ruler.py
git commit -m "feat(eval): RULER runner with 5-task MVP subset (niah_single/multikey/multiquery, vt, qa_1)"
```

---

### Task 13: RULER 128K baseline (real model, on GPU)

**Files:** none new.

- [ ] **Step 1: Run RULER 128K, 10/task**

```bash
uv run python eval/runners/run_ruler.py \
    --model-id Qwen/Qwen3.5-4B-Instruct \
    --lengths 131072 \
    --tasks niah_single_1 niah_multikey_1 niah_multiquery vt qa_1 \
    --n-per-task 10
```

Expected: 50 cells (5 tasks × 10 each). Wall clock ~20-40 min on single H100. Output dir under `eval/_outputs/ruler/<hash>/`.

- [ ] **Step 2: Stash report**

```bash
mkdir -p docs/reports/phase1_artifacts
cp eval/_outputs/ruler/*/summary.md docs/reports/phase1_artifacts/ruler_128k_native.md
cp eval/_outputs/ruler/*/metrics.json docs/reports/phase1_artifacts/ruler_128k_native.json
```

- [ ] **Step 3: Commit**

```bash
git add docs/reports/phase1_artifacts/ruler_128k_native*
git commit -m "eval(phase1): baseline RULER 128K 5-task subset"
```

---

### Task 14: Write `BASELINE_W1.md` report

**Files:**
- Create: `docs/reports/BASELINE_W1.md`
- Modify: `README.md` (link to report)

- [ ] **Step 1: Write the report**

Create `docs/reports/BASELINE_W1.md` with the following structure (fill numeric tables from the actual metrics.json files in `docs/reports/phase1_artifacts/`):

```markdown
# W1 Baseline Report — Qwen3.5-4B-Instruct

**Date:** 2026-05-XX
**Model:** Qwen/Qwen3.5-4B-Instruct
**Hardware:** 1×H100 80GB (eval-only; CPT/SFT runs in later phases use the full 2×8 IB cluster)

## Summary

Baseline numbers for Phase 2-7 CPT/SFT to compare against. The headline gap is
**NIAH @ 1M with static YaRN vs native 128K** — this delta is what Stage A 1M CPT
must close.

## NIAH

| Length     | YaRN factor | Depths covered | Accuracy | n_cells |
|------------|-------------|----------------|----------|---------|
| 32,768     | none        | 10/30/50/70/90 | TODO     | 10      |
| 131,072    | none        | 10/30/50/70/90 | TODO     | 10      |
| 1,048,576  | 4.0         | 10/30/50/70/90 | TODO     | 10      |

(Fill TODO from `docs/reports/phase1_artifacts/niah_*.json#accuracy`.)

### By-depth breakdown @ 1M (where we expect the cliff)

| Depth | Accuracy |
|-------|----------|
| 10%   | TODO     |
| 30%   | TODO     |
| 50%   | TODO     |
| 70%   | TODO     |
| 90%   | TODO     |

(Compute from `niah_1m_yarn.json#predictions` by grouping on depth_pct.)

## RULER 128K (5-task MVP subset)

| Task             | Accuracy | n |
|------------------|----------|---|
| niah_single_1    | TODO     | 10 |
| niah_multikey_1  | TODO     | 10 |
| niah_multiquery  | TODO     | 10 |
| vt               | TODO     | 10 |
| qa_1             | TODO     | 10 |

## Observations

(Fill 2-4 bullets after looking at the numbers. Expected pattern:)
- Native 128K NIAH near-perfect; static YaRN @ 1M shows significant drop, especially
  at edge depths (10%/90%).
- multikey_1 and multiquery weaker than single — confirms ttt.md §13 assumption that
  multi-needle stress tests are needed.
- vt likely the hardest of the 5 RULER tasks at 128K.

## Quality gates for Phase 2 (Stage A 1M CPT)

To declare Stage A a success, the post-CPT model must:
- NIAH @ 1M accuracy ≥ baseline_yarn_1m_accuracy + 0.10 (≥ +10pp)
- RULER 128K mean accuracy not degraded vs baseline (within 0.02)
- Short benchmarks (MMLU/GSM8K — measured in Phase 2 Task X) within 0.02 of base

## Artifacts

Raw outputs in `docs/reports/phase1_artifacts/`:
- `niah_32768_native.{md,json}`
- `niah_131072_native.{md,json}`
- `niah_1m_yarn.{md,json}`
- `ruler_128k_native.{md,json}`
```

- [ ] **Step 2: Fill in the numeric TODOs**

Read each `docs/reports/phase1_artifacts/*.json`, extract `accuracy` and `by_length`, write them into the markdown tables.

For the by-depth breakdown at 1M, post-process the predictions:

```bash
uv run python - <<'PY'
import glob, json, collections
paths = glob.glob("eval/_outputs/baseline/*_niah_1048576/predictions.jsonl")
assert paths, "no 1M predictions.jsonl found — did Task 11 run?"
preds = [json.loads(l) for l in open(paths[0])]
by = collections.defaultdict(list)
for p in preds:
    by[p["depth_pct"]].append(p["ok"])
for d in sorted(by):
    print(f"{d:3d}% : {sum(by[d])/len(by[d]):.2%}  ({sum(by[d])}/{len(by[d])})")
PY
```

Paste each line into the "By-depth breakdown @ 1M" table. Do the same shape of one-liner against `*_ruler/predictions.jsonl` for the RULER per-task table (group on `task` instead of `depth_pct`):

```bash
uv run python - <<'PY'
import glob, json, collections
paths = glob.glob("eval/_outputs/ruler/*/predictions.jsonl")
assert paths, "no ruler predictions.jsonl found — did Task 13 run?"
preds = [json.loads(l) for l in open(paths[0])]
by = collections.defaultdict(list)
for p in preds:
    by[p["task"]].append(p["ok"])
for t in sorted(by):
    print(f"{t:20s} : {sum(by[t])/len(by[t]):.2%}  ({sum(by[t])}/{len(by[t])})")
PY
```

- [ ] **Step 3: Add a link from README**

Open `README.md`, find the "12 周里程碑" section (the milestones table). Append a row right under the W1 entry:

```markdown
| W1 报告 | [`docs/reports/BASELINE_W1.md`](docs/reports/BASELINE_W1.md) | — |
```

- [ ] **Step 4: Commit**

```bash
git add docs/reports/BASELINE_W1.md README.md
git commit -m "docs(phase1): W1 baseline report — NIAH 32K/128K/1M + RULER 128K"
git tag phase1-baseline -m "Phase 1 baseline established"
```

---

## Phase 1 Done Definition

- [ ] All tests pass: `uv run pytest tests/ -q` → ≥ 13 passed
- [ ] All commits land on `main`
- [ ] Tag `phase1-baseline` exists
- [ ] `docs/reports/BASELINE_W1.md` filled (no TODOs)
- [ ] `eval/_outputs/baseline/*/` directories exist for 4K / 32K / 128K / 1M runs
- [ ] `external/flame`, `external/flash-linear-attention`, `external/LLaMA-Factory`, `external/RULER` cloned

Phase 2 (`docs/superpowers/plans/<date>-phase2-data-stage-a.md`, written by next `writing-plans` pass) starts from this tag.
