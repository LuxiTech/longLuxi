# Phase 2 — Resolve 1M Blocker + Data v0 + Stage A Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Phase 1 "1M dense eval BLOCKED" issue by standing up vLLM TP=4 serving + a vLLM-backed eval runner, build a minimal data pipeline (arXiv + PG-19 → 256K packed sequences), wire up the flame training stack, and produce a 256K Stage A *smoke* checkpoint (~10M tokens) that round-trips back through eval to confirm the loop works end-to-end. Defer the full 250M-token 1M main run to Phase 3.

**Architecture:**
- **Eval rig** splits into two backends: existing `transformers + sdpa` (for short contexts, < 256K, easy to script) and a new `vllm` backend (for ≥ 256K and the 1M baseline). They share the same `niah` / `ruler` cell builders from Phase 1; only the inference call differs. The vLLM server stays live across multiple eval invocations to avoid the ~12-minute model-load overhead each time.
- **Data pipeline** mints a small smoke dataset (~10M tokens of arXiv + PG-19) packed into 256K sequences with `<doc>` separators and cross-doc-boundary metadata. Token-level chunking goes through the Qwen3.5-4B tokenizer so position counts match the model.
- **Training** uses `fla-org/flame` (already cloned in `external/flame`) with CP=4 across cards 0-3 (the user's GPU policy — Spec's CP=16 was for 2×8 IB; we adjust). The 256K smoke is the cheapest run that exercises the same code path as the planned 1M CPT, so failures here surface before we commit a multi-hour main run.
- **Smoke success criterion** is *binary*: ckpt exists, loads back as a `Qwen/Qwen3.5-4B`-compatible model in HF transformers, and shows ≤ 5pp degradation vs `phase1-baseline` on RULER 128K (we don't expect *improvement* from 10M tokens of CPT; just want to confirm we haven't broken the model).

**Tech Stack:** vLLM (serving), Qwen/Qwen3.5-4B + YaRN factor=4, fla-org/flame (torchtitan + FLA), HuggingFace datasets (PG-19, arXiv subset), Qwen3.5-4B tokenizer, pytest, uv.

---

## File Structure

**New files:**
- `eval/runners/_vllm_client.py` — small HTTP client wrapping vLLM's OpenAI-compatible `/v1/completions`. Greedy single-turn calls only (matches Phase 1 eval semantics).
- `eval/runners/run_via_vllm.py` — CLI that drives NIAH or RULER cells through `_vllm_client`. Mirrors the arg surface of `scripts/baseline_eval.py` and `eval/runners/run_ruler.py`.
- `eval/runners/vllm_server.sh` — helper to launch the vLLM server with our YaRN config on cards 0-3.
- `scripts/install_flash_attn_optional.sh` — best-effort flash-attn wheel install, no-op on failure. Documented as optional throughout.
- `data/scripts/tokenize_to_chunks.py` — read `data/processed/long_docs_*.jsonl`, tokenize with Qwen3.5-4B, emit 8K `raw_chunk` jsonl.
- `tests/test_vllm_client.py` — mocked HTTP-server test of the client (no GPU).
- `tests/test_data_pipeline.py` — unit tests for the prepare/tokenize/pack chain (uses tiny synthetic input).
- `tests/test_flame_wrapper.py` — verifies TOML→flame args translation (no GPU).
- `docs/reports/PHASE2_REPORT.md` — final write-up linking back to BASELINE_W1 deltas.

**Modified files:**
- `data/scripts/prepare_long_docs.py` — Phase 1 left this as `raise NotImplementedError`. Implement arxiv + pg19 paths.
- `data/scripts/pack_docs.py` — Phase 1 stub; implement multi-doc packing for a single target ctx.
- `training/flame_wrapper/toml_to_flame_args.py` (new helper) — translate `configs/training/stage_a_1m_cpt.toml` into flame CLI flags. The Phase 1 `setup_flame.sh` already exists.
- `training/scripts/run_stage_a.sh` — Phase 1 had only a placeholder echo. Wire up the real torchrun call against flame for the 256K smoke phase.
- `configs/training/stage_a_1m_cpt.toml` — patch `[parallelism]` block: `context_parallel_degree = 4` (was 16) to match the user's GPU restriction; add a `[smoke]` section overriding `context_length=262144` for the smoke phase only.
- `configs/yarn/yarn_1m.json` — verify it's still good (no change expected; confirm in Task 2).
- `Makefile` — add `vllm-serve` and `vllm-stop` targets, plus a `phase2-smoke` shortcut.
- `pyproject.toml` — bump `vllm` extra dep version constraint if needed (current is `>=0.6`).
- `README.md` — link `docs/reports/PHASE2_REPORT.md` once it exists.

---

## Pre-flight verification (must pass before Task 1)

```bash
cd /home/user01/Minko/longluxi
git log --oneline -1                # should be `2039409 docs(phase1): W1 baseline report`
git tag | grep phase1-baseline      # should print "phase1-baseline"
uv run pytest tests/ -q             # should print "29 passed"
ls /home/user01/Minko/models/Qwen3.5-4B/model.safetensors-00001-of-00002.safetensors
nvidia-smi -L | head -4             # should list 4 H100s
```

If any check fails, fix the breakage before starting Phase 2.

---

### Task 1: Install vLLM and verify it loads

**Files:** none (uses existing `pyproject.toml` `[project.optional-dependencies].inference` block).

- [ ] **Step 1: Install the inference extra**

```bash
uv sync --extra eval --extra dev --extra inference
```

Expected: pulls in `vllm`, `sglang`, `uvicorn`, `fastapi`, `httpx`. Wheels are pre-built so this should be quick (~1-3 min). If `vllm` ships a wheel mismatched with torch 2.11+cu130, downgrade torch in `pyproject.toml`'s `inference` extra to the version vllm expects (record the exact pin in the commit message).

- [ ] **Step 2: Verify vLLM imports**

```bash
uv run python -c "import vllm; print(vllm.__version__)"
```

Expected: prints a version string ≥ 0.6, exits 0.

- [ ] **Step 3: Confirm tests still pass**

```bash
uv run pytest tests/ -q
```

Expected: `29 passed`.

- [ ] **Step 4: Commit if pyproject was edited**

If you had to change `pyproject.toml`:
```bash
git add pyproject.toml
git commit -m "deps(inference): pin vllm/torch to working combo for cu130 H100" \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If `pyproject.toml` is untouched, skip the commit.

---

### Task 2: vLLM server launcher script

**Files:**
- Create: `eval/runners/vllm_server.sh`
- Modify: `Makefile` (add `vllm-serve` + `vllm-stop` targets)

- [ ] **Step 1: Write the launcher script**

Create `eval/runners/vllm_server.sh`:

```bash
#!/usr/bin/env bash
# Launch vLLM serving Qwen3.5-4B with YaRN factor=4 (1M context) on cards 0-3.
# Defaults: TP=4, max-model-len=1010000 (matches Qwen3.5-4B official 1M extrapolation cap).
#
# Usage:
#   bash eval/runners/vllm_server.sh                  # foreground
#   bash eval/runners/vllm_server.sh > vllm.log 2>&1 & # background
#
# Then in another shell, hit http://localhost:8001/v1/completions

set -euo pipefail

MODEL="${MODEL:-/home/user01/Minko/models/Qwen3.5-4B}"
PORT="${PORT:-8001}"
TP="${TP:-4}"
MAX_LEN="${MAX_LEN:-1010000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
DTYPE="${DTYPE:-bfloat16}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# YaRN config matches configs/yarn/yarn_1m.json: factor=4.0, original=262144
ROPE_SCALING='{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}'

echo "[vllm-serve] model=$MODEL port=$PORT tp=$TP max_len=$MAX_LEN gpus=$CUDA_VISIBLE_DEVICES"

exec uv run vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --rope-scaling "$ROPE_SCALING" \
    --enable-chunked-prefill \
    --disable-log-requests
```

- [ ] **Step 2: chmod and add Makefile targets**

```bash
chmod +x eval/runners/vllm_server.sh
```

Open `Makefile` and append (right after the `setup-external` block):

```makefile
# ---------------- vLLM serving ----------------

.PHONY: vllm-serve vllm-stop
vllm-serve:
	bash eval/runners/vllm_server.sh

vllm-stop:
	-pkill -f "vllm serve" 2>/dev/null || true
	@echo "[vllm-stop] requested shutdown"
```

Also extend the `help:` echo block with:
```
	@echo "  vllm-serve            launch vLLM TP=4 serving Qwen3.5-4B at 1M ctx"
	@echo "  vllm-stop             kill running vLLM server"
```

- [ ] **Step 3: Smoke test the script syntax (do not actually start yet)**

```bash
bash -n eval/runners/vllm_server.sh && echo "syntax OK"
```

Expected: prints "syntax OK". (We start the real server in Task 4.)

- [ ] **Step 4: Commit**

```bash
git add eval/runners/vllm_server.sh Makefile
git commit -m "feat(eval): vLLM serving launcher with YaRN-1M + TP=4 on cards 0-3" \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: vLLM HTTP client (TDD, no GPU)

**Files:**
- Create: `eval/runners/_vllm_client.py`
- Create: `tests/test_vllm_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_vllm_client.py`:

```python
"""Tests for the vLLM HTTP client. No GPU; uses a stub HTTP server."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from eval.runners._vllm_client import VllmClient


class _StubHandler(BaseHTTPRequestHandler):
    captured: list[dict] = []
    response_text = "42"

    def log_message(self, *a, **kw):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n))
        type(self).captured.append({"path": self.path, "body": body})
        # Stub returns both the /completions shape (choices[].text) and the
        # /chat/completions shape (choices[].message.content) so the same handler
        # serves both endpoints.
        payload = {
            "choices": [{
                "text": type(self).response_text,
                "message": {"role": "assistant", "content": type(self).response_text},
                "finish_reason": "stop",
            }],
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def stub_server():
    _StubHandler.captured = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = HTTPServer(("127.0.0.1", port), _StubHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{port}", _StubHandler
    finally:
        server.shutdown()


def test_complete_sends_greedy_request(stub_server):
    base_url, handler = stub_server
    client = VllmClient(base_url=base_url, model="Qwen/Qwen3.5-4B")
    out = client.complete("hello", max_new_tokens=8)
    assert out == "42"
    assert len(handler.captured) == 1
    body = handler.captured[0]["body"]
    assert body["model"] == "Qwen/Qwen3.5-4B"
    assert body["prompt"] == "hello"
    assert body["max_tokens"] == 8
    assert body["temperature"] == 0.0


def test_complete_uses_chat_template_path_when_messages_given(stub_server):
    base_url, handler = stub_server
    handler.response_text = "ok"
    client = VllmClient(base_url=base_url, model="Qwen/Qwen3.5-4B")
    out = client.complete_chat(
        [{"role": "user", "content": "what is 6*7?"}],
        max_new_tokens=4,
        enable_thinking=False,
    )
    assert out == "ok"
    body = handler.captured[0]["body"]
    assert "messages" in body
    assert body["messages"][0]["content"] == "what is 6*7?"
    assert body["max_tokens"] == 4
    # vLLM-specific: thinking flag passed as chat_template_kwargs
    assert body.get("chat_template_kwargs", {}).get("enable_thinking") is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_vllm_client.py -v
```

Expected: `ImportError: No module named 'eval'` or similar. (Both tests fail.)

- [ ] **Step 3: Create the module**

Create `eval/runners/__init__.py` (empty, so `eval.runners` is a package):

```python
```

Create `eval/runners/_vllm_client.py`:

```python
"""Minimal HTTP client for vLLM's OpenAI-compatible API.

Used by run_via_vllm.py (NIAH/RULER eval) and any future scripts that need
to hit a running vLLM server. Stays small on purpose — no streaming, no
retries, no fancy features. Caller scripts wrap retries if they need them.
"""
from __future__ import annotations

import json
from typing import Any

import httpx


class VllmClient:
    def __init__(self, base_url: str = "http://localhost:8001", model: str = "Qwen/Qwen3.5-4B",
                 timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def complete(self, prompt: str, max_new_tokens: int = 128,
                 stop: list[str] | None = None) -> str:
        """Greedy completion from a raw prompt string."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        if stop:
            payload["stop"] = stop
        resp = self._client.post(f"{self.base_url}/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"]

    def complete_chat(self, messages: list[dict[str, str]], max_new_tokens: int = 128,
                       enable_thinking: bool = False,
                       stop: list[str] | None = None) -> str:
        """Greedy chat completion (applies the model's chat template server-side)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if stop:
            payload["stop"] = stop
        resp = self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._client.close()
```

Note: the test's `_StubHandler.do_POST` is written to emit both response shapes
(`choices[0].text` for `/v1/completions` and `choices[0].message.content` for
`/v1/chat/completions`) so a single handler instance serves both endpoints
the client exercises.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_vllm_client.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Run full suite + ruff**

```bash
uv run pytest tests/ -q
uv run ruff check eval/runners/ tests/test_vllm_client.py
```

Expected: 31 passed (29 + 2); ruff clean.

- [ ] **Step 6: Commit**

```bash
git add eval/runners/__init__.py eval/runners/_vllm_client.py tests/test_vllm_client.py
git commit -m "feat(eval): vLLM HTTP client + tests (stub-server)" \
    -m "" \
    -m "VllmClient.complete/complete_chat both greedy; chat path applies" \
    -m "the model's chat template server-side with enable_thinking=False." \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Launch vLLM, smoke test 1 generation at 4K

**Files:** none. Real GPU + real server.

- [ ] **Step 1: Start vLLM in the background**

```bash
bash eval/runners/vllm_server.sh > /tmp/vllm.log 2>&1 &
echo $! > /tmp/vllm.pid
```

The model load + KV cache allocation at 1M context with TP=4 takes 3-6 minutes the first time. Tail the log to confirm readiness:

```bash
# Wait for "Application startup complete" or "Uvicorn running on http://0.0.0.0:8001"
until grep -q "Uvicorn running" /tmp/vllm.log 2>/dev/null; do sleep 5; done
echo "[vllm] ready"
tail -3 /tmp/vllm.log
```

If you see CUDA OOM during startup, lower `GPU_MEM_UTIL`:

```bash
GPU_MEM_UTIL=0.85 bash eval/runners/vllm_server.sh > /tmp/vllm.log 2>&1 &
```

- [ ] **Step 2: Smoke a tiny completion**

```bash
curl -s http://localhost:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"/home/user01/Minko/models/Qwen3.5-4B","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
    | python3 -c "import json,sys; print(repr(json.load(sys.stdin)['choices'][0]['text']))"
```

Expected: prints something containing `' Paris'` (with leading space, possibly more tokens).

If `model` field is wrong (vLLM uses the resolved path as id), adjust the curl `model` argument to whatever vLLM logged on startup. Get the served name from:
```bash
curl -s http://localhost:8001/v1/models | python3 -m json.tool
```

- [ ] **Step 3: Smoke via the Python client**

```bash
uv run python - <<'PY'
from eval.runners._vllm_client import VllmClient
import json, httpx
served = httpx.get("http://localhost:8001/v1/models").json()["data"][0]["id"]
print("served model id:", served)
c = VllmClient(base_url="http://localhost:8001", model=served)
print(c.complete("The capital of France is", max_new_tokens=8))
PY
```

Expected: prints the served model id then a Paris-containing completion.

- [ ] **Step 4: Leave the server running**. Tasks 5 and 6 reuse it. Do NOT stop it yet.

- [ ] **Step 5: No commit** — this task is a real-server smoke; no artifacts.

---

### Task 5: vLLM-backed NIAH runner (closes the 1M BLOCKER)

**Files:**
- Create: `eval/runners/run_via_vllm.py`

- [ ] **Step 1: Implement the runner**

Create `eval/runners/run_via_vllm.py`:

```python
"""NIAH / RULER eval driven by a running vLLM server.

Reuses the cell builders from src/longluxi/eval/{niah,ruler}.py — only the
inference call changes. Designed to be cheap to call repeatedly against
the same long-running vLLM server (no model reload).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import httpx

# GPU policy: only cards 0-3 (4-7 reserved). vLLM was already launched with these
# cards; this just affects any local-side ops (chunking etc.).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from longluxi.eval.haystack import load_haystack_corpus  # noqa: E402
from longluxi.eval.niah import build_cell as build_niah, grade as grade_niah  # noqa: E402
from longluxi.eval.results import write_results  # noqa: E402
from longluxi.eval.ruler import TASK_REGISTRY, build_ruler_cell  # noqa: E402
from eval.runners._vllm_client import VllmClient  # noqa: E402


def parse_len(s: str) -> int:
    s = s.strip().lower()
    mult = 1
    if s.endswith("k"):
        mult, s = 1024, s[:-1]
    elif s.endswith("m"):
        mult, s = 1024 * 1024, s[:-1]
    return int(float(s) * mult)


def _resolve_served_id(base_url: str) -> str:
    data = httpx.get(f"{base_url}/v1/models", timeout=30).json()
    return data["data"][0]["id"]


def _load_qwen_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("/home/user01/Minko/models/Qwen3.5-4B",
                                          trust_remote_code=True)


def run_niah_grid(client: VllmClient, lengths: list[int], depths: list[int],
                  n_per_cell: int, seed: int, max_cells: int | None,
                  max_new_tokens: int) -> list[dict]:
    tok = _load_qwen_tokenizer()
    rng = random.Random(seed)
    hay = load_haystack_corpus(max(lengths), allow_network=True)

    preds: list[dict] = []
    for L in lengths:
        for d in depths:
            for _ in range(n_per_cell):
                cell = build_niah(hay, L, d, tok, rng)
                # Use chat path so server applies its own chat template w/ enable_thinking=False
                pred = client.complete_chat(
                    [{"role": "user", "content": cell.prompt}],
                    max_new_tokens=max_new_tokens,
                    enable_thinking=False,
                )
                ok = grade_niah(cell, pred)
                preds.append({
                    "length_tokens": cell.length_tokens,
                    "depth_pct": cell.depth_pct,
                    "expected": cell.expected,
                    "pred": pred[:200],
                    "ok": ok,
                    "input_tokens": -1,  # vLLM doesn't return this on chat completions by default
                })
                print(f"[vllm-niah] L={L} d={d}% expected={cell.expected} pred={pred[:40]!r} ok={ok}",
                      flush=True)
                if max_cells and len(preds) >= max_cells:
                    return preds
    return preds


def run_ruler_grid(client: VllmClient, lengths: list[int], tasks: list[str],
                   n_per_task: int, seed: int, max_new_tokens: int) -> list[dict]:
    tok = _load_qwen_tokenizer()
    rng = random.Random(seed)
    preds: list[dict] = []
    for L in lengths:
        for task_name in tasks:
            for _ in range(n_per_task):
                cell = build_ruler_cell(task_name, L, tok, rng)
                pred = client.complete_chat(
                    [{"role": "user", "content": cell["prompt"]}],
                    max_new_tokens=max_new_tokens,
                    enable_thinking=False,
                )
                ok = TASK_REGISTRY[task_name].grade_fn(cell, pred)
                preds.append({
                    "length_tokens": cell["length_tokens"],
                    "depth_pct": cell.get("depth_pct", -1),
                    "task": task_name,
                    "expected": cell["expected"],
                    "pred": pred[:200],
                    "ok": ok,
                    "input_tokens": -1,
                })
                print(f"[vllm-ruler] L={L} task={task_name} ok={ok} expected={cell['expected']} pred={pred[:40]!r}",
                      flush=True)
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["niah", "ruler"], required=True)
    p.add_argument("--base-url", default="http://localhost:8001")
    p.add_argument("--lengths", nargs="+", type=parse_len, default=[131072])
    # niah-only
    p.add_argument("--depths", nargs="+", type=int, default=[10, 30, 50, 70, 90])
    p.add_argument("--n-per-cell", type=int, default=2)
    # ruler-only
    p.add_argument("--tasks", nargs="+", default=list(TASK_REGISTRY.keys()))
    p.add_argument("--n-per-task", type=int, default=10)
    # shared
    p.add_argument("--max-cells", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    served = _resolve_served_id(args.base_url)
    client = VllmClient(base_url=args.base_url, model=served)

    tag = f"{args.bench}_{max(args.lengths)}"
    out_dir = Path(args.out_dir or f"eval/_outputs/vllm/{hashlib.sha256(served.encode()).hexdigest()[:8]}_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bench == "niah":
        preds = run_niah_grid(client, args.lengths, args.depths, args.n_per_cell,
                              args.seed, args.max_cells, args.max_new_tokens)
    else:
        preds = run_ruler_grid(client, args.lengths, args.tasks, args.n_per_task,
                               args.seed, args.max_new_tokens)

    metrics = write_results(out_dir, model_id=served, yarn_config="yarn_1m@vllm-server",
                            predictions=preds)
    if args.bench == "ruler":
        by_task: dict[str, list[bool]] = {}
        for p_ in preds:
            by_task.setdefault(p_["task"], []).append(p_["ok"])
        metrics["by_task"] = {t: (sum(o) / len(o)) for t, o in by_task.items()}
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke at 4K, 1 cell, against the running server**

```bash
uv run python eval/runners/run_via_vllm.py --bench niah \
    --lengths 4096 --depths 50 --n-per-cell 1 --max-new-tokens 32
```

Expected: 1 cell, accuracy 1.0, metrics JSON printed.

- [ ] **Step 3: Ruff check**

```bash
uv run ruff check eval/runners/run_via_vllm.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add eval/runners/run_via_vllm.py
git commit -m "feat(eval): vLLM-backed NIAH/RULER runner reusing Phase 1 cell builders" \
    -m "" \
    -m "Closes the 1M dense single-H100 OOM by going through vLLM TP=4." \
    -m "Same cell construction and grading as the transformers runner; only" \
    -m "the inference call differs. Uses /v1/chat/completions so server-side" \
    -m "chat template + enable_thinking=False stays uniform." \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Re-run NIAH 1M baseline via vLLM (close the BLOCKER)

**Files:** none new; produces eval artifacts.

- [ ] **Step 1: Run NIAH @ 1M, depths 10/30/50/70/90, 2/cell**

```bash
uv run python eval/runners/run_via_vllm.py --bench niah \
    --lengths 1048576 --depths 10 30 50 70 90 --n-per-cell 2
```

Expected: 10 cells, wall clock 30-90 minutes (vLLM at 1M is much faster than transformers SDPA, but each cell is still substantial). Accuracy is the headline number — for static YaRN factor=4 at full 4× extrapolation, expect somewhere in 0.4-0.9.

If accuracy is high (≥ 0.95), the static YaRN baseline is already strong on NIAH at 1M and Stage A CPT needs to be evaluated on harder metrics (RULER VT, multi-needle).

If accuracy is low (< 0.5), there's substantial room for CPT to improve — also possible the vLLM YaRN config isn't being applied correctly. Sanity-check by re-running at 512K:

```bash
uv run python eval/runners/run_via_vllm.py --bench niah \
    --lengths 524288 --depths 10 30 50 70 90 --n-per-cell 2
```

Phase 1 had 100% at 512K with transformers; if vLLM at 512K differs, something's wrong with the vLLM config.

- [ ] **Step 2: Stash report**

```bash
cp eval/_outputs/vllm/*_niah_1048576/summary.md docs/reports/phase1_artifacts/niah_1m_yarn.md
cp eval/_outputs/vllm/*_niah_1048576/metrics.json docs/reports/phase1_artifacts/niah_1m_yarn.json
# Mark the old BLOCKED note as resolved
mv docs/reports/phase1_artifacts/niah_1m_yarn_BLOCKED.md docs/reports/phase1_artifacts/niah_1m_yarn_BLOCKED_RESOLVED.md
```

Append a "Resolved" note to the renamed file:
```bash
cat >> docs/reports/phase1_artifacts/niah_1m_yarn_BLOCKED_RESOLVED.md <<'EOF'

---

## RESOLVED in Phase 2

Resolution path taken: **vLLM TP=4 serving** (cards 0-3).
See `eval/runners/run_via_vllm.py` and `eval/runners/vllm_server.sh`.

Final 1M number lives in `niah_1m_yarn.json` and is the comparison anchor
for Stage A CPT 1M eval.
EOF
```

- [ ] **Step 3: Commit**

```bash
git add docs/reports/phase1_artifacts/
git commit -m "eval(phase2): close 1M NIAH BLOCKER via vLLM TP=4" \
    -m "" \
    -m "Phase 1's BLOCKED 1M NIAH @ YaRN factor=4 is now measured." \
    -m "BLOCKED note moved to *_RESOLVED.md with resolution path documented." \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Re-run RULER 128K via vLLM (cross-check vs transformers baseline)

**Files:** none new.

- [ ] **Step 1: Run RULER 128K, 10/task**

```bash
uv run python eval/runners/run_via_vllm.py --bench ruler \
    --lengths 131072 --tasks niah_single_1 niah_multikey_1 niah_multiquery vt qa_1 \
    --n-per-task 10
```

Expected: 50 cells, ~5-20 min wall clock via vLLM. Output dir `eval/_outputs/vllm/<hash>_ruler_131072/`.

- [ ] **Step 2: Compare against Phase 1 transformers baseline**

```bash
uv run python - <<'PY'
import json, glob
tf = json.load(open("docs/reports/phase1_artifacts/ruler_128k_native.json"))
vl = json.load(open(glob.glob("eval/_outputs/vllm/*_ruler_131072/metrics.json")[0]))
print("transformers (Phase 1):")
print(f"  overall {tf['accuracy']:.2%}")
for t in sorted(tf.get("by_task", {})):
    print(f"  {t:20s}: {tf['by_task'][t]:.2%}")
print("\nvLLM (Phase 2):")
print(f"  overall {vl['accuracy']:.2%}")
for t in sorted(vl.get("by_task", {})):
    print(f"  {t:20s}: {vl['by_task'][t]:.2%}")
PY
```

Expected: numbers within a few pp of the transformers baseline. Large drift indicates a config issue (chat template / sampling / YaRN application differs between backends). Investigate before proceeding if any task differs by > 10pp.

- [ ] **Step 3: Stash + commit**

```bash
cp eval/_outputs/vllm/*_ruler_131072/summary.md docs/reports/phase1_artifacts/ruler_128k_vllm.md
cp eval/_outputs/vllm/*_ruler_131072/metrics.json docs/reports/phase1_artifacts/ruler_128k_vllm.json
git add docs/reports/phase1_artifacts/ruler_128k_vllm*
git commit -m "eval(phase2): RULER 128K via vLLM (control vs transformers baseline)" \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Leave vLLM running.** Task 12 also uses it. (Or stop it now with `make vllm-stop` if cards are needed for Task 11 flame training — see notes in Task 11.)

---

### Task 8: Data ingestion (arXiv subset + PG-19)

**Files:**
- Modify: `data/scripts/prepare_long_docs.py` (was `raise NotImplementedError` in Phase 1)

- [ ] **Step 1: Replace the stub with a real implementation**

Open `data/scripts/prepare_long_docs.py` and replace its contents with:

```python
"""Download + clean + tokenize long documents into training-ready jsonl.

MVP scope (Phase 2 smoke): ~5-10M tokens total, two sources:
- arxiv (via `armanc/scientific_papers` HF dataset, arxiv subset)
- pg19 (via `emozilla/pg19-test` HF dataset)

Output schema (one jsonl per source under data/processed/):
  {
    "doc_id": str,        # f"{source}/{idx}"
    "source": str,
    "title": str | None,
    "text": str,          # cleaned plaintext
    "n_tokens": int,      # by Qwen3.5-4B tokenizer
    "metadata": {...}
  }

Usage:
    uv run python data/scripts/prepare_long_docs.py --source arxiv --max-docs 200
    uv run python data/scripts/prepare_long_docs.py --source pg19  --max-docs 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from longluxi.paths import PROCESSED_DATA_DIR, ensure_dirs  # noqa: E402


def _iter_arxiv(max_docs: int, min_tokens: int, tokenizer):
    from datasets import load_dataset
    ds = load_dataset("armanc/scientific_papers", "arxiv", split="train", streaming=True,
                       trust_remote_code=True)
    yielded = 0
    for ex in ds:
        text = (ex.get("article") or "").strip()
        if not text:
            continue
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n < min_tokens:
            continue
        yield {
            "doc_id": f"arxiv/{yielded}",
            "source": "arxiv",
            "title": (ex.get("abstract") or "").strip()[:120] or None,
            "text": text,
            "n_tokens": n,
            "metadata": {},
        }
        yielded += 1
        if yielded >= max_docs:
            return


def _iter_pg19(max_docs: int, min_tokens: int, tokenizer):
    from datasets import load_dataset
    # pg19-test is small (~100 books); pg19 train is huge. Use test for smoke.
    ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
    yielded = 0
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n < min_tokens:
            continue
        yield {
            "doc_id": f"pg19/{yielded}",
            "source": "pg19",
            "title": ex.get("short_book_title"),
            "text": text,
            "n_tokens": n,
            "metadata": {"publication_date": ex.get("publication_date")},
        }
        yielded += 1
        if yielded >= max_docs:
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["arxiv", "pg19"], required=True)
    p.add_argument("--max-docs", type=int, default=200)
    p.add_argument("--min-tokens", type=int, default=4096)
    p.add_argument("--tokenizer", default="/home/user01/Minko/models/Qwen3.5-4B")
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = p.parse_args()

    ensure_dirs()
    out_path = args.out_dir / f"long_docs_{args.source}.jsonl"
    print(f"[prepare] writing -> {out_path}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    iterator = _iter_arxiv if args.source == "arxiv" else _iter_pg19
    n_docs = 0
    n_tokens = 0
    with out_path.open("w") as f:
        for rec in iterator(args.max_docs, args.min_tokens, tokenizer):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_docs += 1
            n_tokens += rec["n_tokens"]
            if n_docs % 20 == 0:
                print(f"[prepare] {n_docs} docs, {n_tokens/1e6:.2f}M tokens", flush=True)
    print(f"[prepare] done: {n_docs} docs, {n_tokens/1e6:.2f}M tokens -> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run arxiv ingestion**

```bash
uv run python data/scripts/prepare_long_docs.py --source arxiv --max-docs 200 --min-tokens 4096
```

Expected: writes ~200 arxiv docs to `data/processed/long_docs_arxiv.jsonl`, ~5-15M tokens total. Wall clock 1-5 min (streaming download).

- [ ] **Step 3: Run pg19 ingestion**

```bash
uv run python data/scripts/prepare_long_docs.py --source pg19 --max-docs 50 --min-tokens 16384
```

Expected: ~50 books, ~3-8M tokens. Books are long so even 50 is plenty.

- [ ] **Step 4: Verify the output**

```bash
uv run python - <<'PY'
import json
total = {"arxiv": 0, "pg19": 0}
for src in total:
    with open(f"data/processed/long_docs_{src}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            total[src] += r["n_tokens"]
    print(f"{src}: {total[src]/1e6:.2f}M tokens")
print(f"total: {sum(total.values())/1e6:.2f}M tokens")
PY
```

Expected: both sources non-zero, total around 8-20M tokens (enough for a 256K smoke).

- [ ] **Step 5: Commit the script (data files stay gitignored)**

```bash
git add data/scripts/prepare_long_docs.py
git commit -m "feat(data): implement long-doc ingestion for arxiv + pg19" \
    -m "" \
    -m "Streams from HF datasets, tokenizes with the local Qwen3.5-4B tokenizer," \
    -m "filters by min token count, writes per-source jsonl to data/processed/." \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Multi-doc packing into 256K sequences (TDD)

**Files:**
- Modify: `data/scripts/pack_docs.py` (was stub)
- Create: `tests/test_data_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_pipeline.py`:

```python
"""Data pipeline tests — no GPU, no network. Uses a tiny synthetic input."""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_sample_jsonl(path: Path, n_docs: int, n_tokens_per_doc: int):
    with path.open("w") as f:
        for i in range(n_docs):
            text = "word " * n_tokens_per_doc  # 1 token per word with whitespace tokenizer
            f.write(json.dumps({
                "doc_id": f"sample/{i}",
                "source": "sample",
                "title": f"doc {i}",
                "text": text.strip(),
                "n_tokens": n_tokens_per_doc,
                "metadata": {},
            }) + "\n")


def test_pack_docs_writes_target_ctx_sequences(tmp_path):
    src = tmp_path / "in.jsonl"
    _write_sample_jsonl(src, n_docs=20, n_tokens_per_doc=1000)
    out = tmp_path / "packed.jsonl"
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "data/scripts/pack_docs.py"),
         "--target-ctx", "5000",
         "--input-jsonls", str(src),
         "--out", str(out),
         "--tokenizer-mode", "whitespace"],  # bypass HF tokenizer for tests
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, res.stderr
    assert out.exists()
    lines = out.read_text().splitlines()
    assert len(lines) >= 3  # 20 docs * 1000 tokens / 5000 = 4 sequences-ish
    for line in lines:
        rec = json.loads(line)
        assert "seq_id" in rec
        assert "target_ctx" in rec and rec["target_ctx"] == 5000
        assert "docs" in rec
        assert rec["n_tokens"] <= 5000 + 100  # allow some slack for separators
        assert "<doc id=" in rec["text"]


def test_pack_docs_respects_doc_boundary_mask(tmp_path):
    src = tmp_path / "in.jsonl"
    _write_sample_jsonl(src, n_docs=4, n_tokens_per_doc=600)
    out = tmp_path / "packed.jsonl"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "data/scripts/pack_docs.py"),
         "--target-ctx", "5000",
         "--input-jsonls", str(src),
         "--out", str(out),
         "--tokenizer-mode", "whitespace"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    rec = json.loads(out.read_text().splitlines()[0])
    # mask_doc_boundaries indicates where each new doc starts (in tokens)
    assert "mask_doc_boundaries" in rec
    assert len(rec["mask_doc_boundaries"]) == len(rec["docs"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_data_pipeline.py -v
```

Expected: failures (pack_docs.py is still `NotImplementedError`).

- [ ] **Step 3: Implement `data/scripts/pack_docs.py`**

Replace the contents of `data/scripts/pack_docs.py` with:

```python
"""Pack multi-document jsonl into target-ctx training sequences.

Strategy:
- Concatenate docs in input order (no topic clustering in MVP) until adding the next
  doc would exceed target_ctx. Emit one packed seq, start the next.
- Each doc wrapped with <doc id="..." source="..."> ... </doc> separators.
- Track per-seq token positions where each doc starts (`mask_doc_boundaries`) so a
  later trainer can apply cross-doc attention masks if desired.

Output schema:
  {
    "seq_id": str,
    "target_ctx": int,
    "docs": [{"doc_id": ..., "source": ..., "n_tokens": ...}, ...],
    "text": str,
    "n_tokens": int,
    "mask_doc_boundaries": [int, ...]  # token offsets of each doc's <doc ...> start
  }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


_SEP_OPEN_TMPL = '<doc id="{doc_id}" source="{source}">\n'
_SEP_CLOSE = '\n</doc>\n'


def _make_tokenizer(mode: str, model_path: str):
    if mode == "whitespace":
        class _Ws:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": text.split()}

            def decode(self, ids):
                return " ".join(ids)
        return _Ws()
    if mode == "qwen":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    raise ValueError(f"unknown tokenizer mode: {mode}")


def _token_len(tok, s: str) -> int:
    return len(tok(s, add_special_tokens=False)["input_ids"])


def pack(input_paths: list[Path], target_ctx: int, tok) -> list[dict]:
    """Return a list of packed-seq records."""
    sep_overhead = _token_len(tok, _SEP_OPEN_TMPL.format(doc_id="x", source="y")) \
                   + _token_len(tok, _SEP_CLOSE)

    docs_iter = []
    for path in input_paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            docs_iter.append(json.loads(line))

    seqs: list[dict] = []
    cur_text_parts: list[str] = []
    cur_docs: list[dict] = []
    cur_boundaries: list[int] = []
    cur_tokens = 0
    seq_idx = 0

    def flush():
        nonlocal cur_text_parts, cur_docs, cur_boundaries, cur_tokens, seq_idx
        if not cur_docs:
            return
        seqs.append({
            "seq_id": f"pack/{target_ctx}/{seq_idx}",
            "target_ctx": target_ctx,
            "docs": cur_docs,
            "text": "".join(cur_text_parts),
            "n_tokens": cur_tokens,
            "mask_doc_boundaries": cur_boundaries,
        })
        seq_idx += 1
        cur_text_parts = []
        cur_docs = []
        cur_boundaries = []
        cur_tokens = 0

    for d in docs_iter:
        body = d["text"]
        body_tokens = d.get("n_tokens") or _token_len(tok, body)
        # If a single doc exceeds target_ctx, truncate it on token-boundary.
        if body_tokens + sep_overhead > target_ctx:
            ids = tok(body, add_special_tokens=False)["input_ids"]
            ids = ids[:max(target_ctx - sep_overhead, 0)]
            body = tok.decode(ids)
            body_tokens = len(ids)

        delta = body_tokens + sep_overhead
        if cur_tokens + delta > target_ctx and cur_docs:
            flush()
        open_tag = _SEP_OPEN_TMPL.format(doc_id=d["doc_id"], source=d["source"])
        cur_boundaries.append(cur_tokens)
        cur_text_parts.append(open_tag)
        cur_text_parts.append(body)
        cur_text_parts.append(_SEP_CLOSE)
        cur_docs.append({"doc_id": d["doc_id"], "source": d["source"], "n_tokens": body_tokens})
        cur_tokens += delta
    flush()
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-ctx", type=int, required=True)
    p.add_argument("--input-jsonls", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tokenizer-mode", choices=["qwen", "whitespace"], default="qwen")
    p.add_argument("--tokenizer-path", default="/home/user01/Minko/models/Qwen3.5-4B")
    args = p.parse_args()

    tok = _make_tokenizer(args.tokenizer_mode, args.tokenizer_path)
    seqs = pack(args.input_jsonls, args.target_ctx, tok)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for s in seqs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[pack] {len(seqs)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_data_pipeline.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Pack the real ingested data into 256K sequences**

```bash
uv run python data/scripts/pack_docs.py \
    --target-ctx 262144 \
    --input-jsonls data/processed/long_docs_arxiv.jsonl data/processed/long_docs_pg19.jsonl \
    --out data/processed/packed_256k.jsonl
```

Expected: ~20-60 packed sequences (depending on how much data each source produced). Wall clock 1-3 min.

- [ ] **Step 6: Sanity check the packed output**

```bash
uv run python - <<'PY'
import json
seqs = [json.loads(l) for l in open("data/processed/packed_256k.jsonl")]
print(f"sequences: {len(seqs)}")
total = sum(s["n_tokens"] for s in seqs)
print(f"total tokens: {total/1e6:.2f}M")
print(f"avg seq length: {total/len(seqs):.0f}")
print(f"first seq docs: {len(seqs[0]['docs'])}, mask boundaries: {seqs[0]['mask_doc_boundaries'][:5]}...")
PY
```

Expected: ~30-60 sequences, ~5-15M tokens total, avg seq length close to but ≤ 262144.

- [ ] **Step 7: Full test suite + ruff**

```bash
uv run pytest tests/ -q
uv run ruff check data/scripts/pack_docs.py tests/test_data_pipeline.py
```

Expected: 33 passed (31 prior + 2 new); ruff clean.

- [ ] **Step 8: Commit**

```bash
git add data/scripts/pack_docs.py tests/test_data_pipeline.py
git commit -m "feat(data): multi-doc packer with cross-doc-boundary metadata" \
    -m "" \
    -m "Greedy fill: concatenate docs until adding the next would exceed" \
    -m "target_ctx; wrap each doc with <doc id=...> separators; track per-seq" \
    -m "token offsets where each doc starts so trainers can mask cross-doc" \
    -m "attention. Whitespace-tokenizer mode keeps tests GPU-free." \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: flame install + TOML→flame args wrapper (TDD)

**Files:**
- Create: `training/flame_wrapper/toml_to_flame_args.py`
- Create: `tests/test_flame_wrapper.py`

- [ ] **Step 1: Install flame into the venv**

```bash
uv pip install -e external/flame
uv pip install -e external/flash-linear-attention
```

Expected: editable installs succeed (both are torch-native, no CUDA compile).

Verify:
```bash
uv run python -c "import flame; print(flame.__file__)"
uv run python -c "import fla; print(fla.__version__)"
```

Expected: prints paths/versions, exits 0.

- [ ] **Step 2: Write the failing test**

Create `tests/test_flame_wrapper.py`:

```python
"""TOML -> flame CLI args wrapper tests. Pure-Python, no GPU."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "training/flame_wrapper/toml_to_flame_args.py"
TOML = REPO_ROOT / "configs/training/stage_a_1m_cpt.toml"


def test_wrapper_emits_expected_cli_flags_for_smoke():
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--toml", str(TOML), "--phase", "smoke"],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr
    args = res.stdout.strip().split()
    # check critical flags appear
    assert any("--training.seq_len" in a for a in args)
    seq_len = next(a for a in args if a.startswith("--training.seq_len="))
    assert seq_len.split("=", 1)[1] == "262144"  # smoke phase override
    assert any("--experimental.context_parallel_degree=4" in a for a in args)
    assert any("--training.dtype=bfloat16" in a for a in args)
    # smoke phase budget
    assert any(("--training.max_steps" in a) or ("--training.steps" in a) for a in args)


def test_wrapper_emits_main_seq_len_when_phase_main():
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--toml", str(TOML), "--phase", "main"],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr
    seq_len = next(a for a in res.stdout.split() if a.startswith("--training.seq_len="))
    assert seq_len.split("=", 1)[1] == "1048576"  # main 1M
```

- [ ] **Step 3: Run test, see it fail**

```bash
uv run pytest tests/test_flame_wrapper.py -v
```

Expected: `FileNotFoundError` on the script, or `ModuleNotFoundError`.

- [ ] **Step 4: Patch the Stage A TOML to add `[smoke]` block and fix CP**

Open `configs/training/stage_a_1m_cpt.toml`. Find the `[parallelism]` block (after `[training]`) and replace it with:

```toml
[parallelism]
# Phase 1 spec assumed 2x8 H100+IB (CP=16). Phase 2 GPU policy = cards 0-3 only.
# Adjust CP_degree to 4; revisit if more cards become available.
context_parallel_degree = 4
context_parallel_rotate_method = "allgather"
tensor_parallel_degree = 1
data_parallel_degree = 1
pipeline_parallel_degree = 1
```

Then, anywhere after `[training]`, append a new section:

```toml
[smoke]
# Phase 2 smoke: 256K context, ~10M tokens (40 steps at 256K)
context_length = 262144
max_steps = 40
learning_rate = 1.0e-5
save_every_steps = 10
```

- [ ] **Step 5: Implement the wrapper**

Create `training/flame_wrapper/toml_to_flame_args.py`:

```python
"""Translate configs/training/stage_*_cpt.toml into flame/torchtitan CLI args.

Source of truth: TOML (our schema).
Downstream: flame/torchtitan CLI (downstream-defined).

Usage:
    uv run python training/flame_wrapper/toml_to_flame_args.py \
        --toml configs/training/stage_a_1m_cpt.toml --phase smoke
    # prints args on stdout, one whitespace-separated string.
"""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def to_flame_args(toml_path: Path, phase: str) -> list[str]:
    data = tomllib.loads(toml_path.read_text())
    run = data["run"]
    model = data["model"]
    training = data["training"].copy()
    parallel = data["parallelism"]
    ckpt = data.get("checkpoint", {})

    # Apply phase overrides if present
    overrides = data.get(phase, {})
    if not overrides and phase == "smoke":
        # If TOML doesn't have a [smoke] block, fall back to main but smaller
        overrides = {"context_length": 262144, "max_steps": 40}
    training.update(overrides)

    seq_len = training["context_length"]
    args: list[str] = [
        f"--job.config_file={toml_path}",  # for downstream record-keeping
        f"--training.seq_len={seq_len}",
        f"--training.dtype={model['dtype']}",
        f"--training.steps={training.get('max_steps', training.get('max_steps_main', 0))}",
        f"--training.learning_rate={training['learning_rate']}",
        f"--training.warmup_steps={training.get('warmup_steps', 0)}",
        f"--training.lr_scheduler={training.get('lr_scheduler', 'cosine_with_warmup')}",
        f"--training.weight_decay={training.get('weight_decay', 0.01)}",
        f"--training.gradient_clip={training.get('max_grad_norm', 1.0)}",
        f"--training.micro_batch_size={training.get('micro_batch_size', 1)}",
        f"--training.gradient_accumulation_steps={training.get('gradient_accumulation', 1)}",
        f"--experimental.context_parallel_degree={parallel['context_parallel_degree']}",
        f"--experimental.context_parallel_rotate_method={parallel['context_parallel_rotate_method']}",
        f"--training.tensor_parallel_degree={parallel['tensor_parallel_degree']}",
        f"--training.data_parallel_degree={parallel['data_parallel_degree']}",
        f"--model.name_or_path={model.get('local_path', '/home/user01/Minko/models/Qwen3.5-4B')}",
        f"--model.attn_impl={model.get('attn_impl', 'flash_attention_2')}",
        f"--checkpoint.folder={run['output_dir']}",
        f"--checkpoint.interval={ckpt.get('save_every_steps', overrides.get('save_every_steps', 25))}",
        f"--checkpoint.keep_last={ckpt.get('keep_last', 3)}",
    ]
    if model.get("gradient_checkpointing"):
        args.append("--training.activation_checkpoint_mode=full")
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--toml", required=True, type=Path)
    p.add_argument("--phase", default="main", choices=["main", "smoke"])
    args_ns = p.parse_args()
    args = to_flame_args(args_ns.toml, args_ns.phase)
    print(" ".join(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run test, verify it passes**

```bash
uv run pytest tests/test_flame_wrapper.py -v
```

Expected: `2 passed`.

- [ ] **Step 7: Full suite + ruff**

```bash
uv run pytest tests/ -q
uv run ruff check training/flame_wrapper/ tests/test_flame_wrapper.py
```

Expected: 35 passed (33 + 2 new); ruff clean.

- [ ] **Step 8: Commit**

```bash
git add configs/training/stage_a_1m_cpt.toml \
        training/flame_wrapper/toml_to_flame_args.py \
        tests/test_flame_wrapper.py
git commit -m "feat(training): flame TOML->args wrapper + Stage A [smoke] block + CP=4" \
    -m "" \
    -m "Phase 2 GPU policy is cards 0-3, so context_parallel_degree drops" \
    -m "from spec's CP=16 (2x8 IB) to CP=4 (single node). Stage A TOML gets" \
    -m "an explicit [smoke] block that the wrapper applies via --phase smoke." \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Stage A.0 smoke — 256K CPT, ~10M tokens, CP=4

**Files:**
- Modify: `training/scripts/run_stage_a.sh` (Phase 1 had only placeholder echo)

- [ ] **Step 1: Free the GPUs**

If vLLM is still running on cards 0-3 from Task 6/7:

```bash
make vllm-stop
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv | head -5
```

Expected: cards 0-3 memory drops to ~MiB single digits. (vLLM holds large KV cache pre-allocations.)

- [ ] **Step 2: Wire the real torchrun call into run_stage_a.sh**

Open `training/scripts/run_stage_a.sh` and replace its full body with:

```bash
#!/usr/bin/env bash
# Stage A: 1M dense CPT (with --smoke flag for the Phase 2 256K shake-out run).
# Multi-node example:
#   MASTER_ADDR=node0 NODE_RANK=0 bash training/scripts/run_stage_a.sh
#   MASTER_ADDR=node0 NODE_RANK=1 bash training/scripts/run_stage_a.sh

set -euo pipefail

PHASE="main"
if [[ "${1:-}" == "--smoke" ]]; then PHASE="smoke"; fi

CONFIG="configs/training/stage_a_1m_cpt.toml"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FLAME_DIR="external/flame"
if [ ! -d "${FLAME_DIR}" ]; then
    echo "[stage-a] flame not found in ${FLAME_DIR}; run 'make setup-flame' first."
    exit 1
fi

FLAME_ARGS=$(uv run python training/flame_wrapper/toml_to_flame_args.py \
    --toml "${CONFIG}" --phase "${PHASE}")

echo "[stage-a] phase=${PHASE} nproc=${NPROC_PER_NODE} nnodes=${NNODES} rank=${NODE_RANK}"
echo "[stage-a] flame args: ${FLAME_ARGS}"

# flame uses torchrun; entrypoint is `flame.train` per torchtitan convention.
uv run torchrun \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    -m flame.train \
    ${FLAME_ARGS} \
    --training.dataset_path data/processed/packed_256k.jsonl
```

Then make sure it stays executable:
```bash
chmod +x training/scripts/run_stage_a.sh
```

> **Note:** the `-m flame.train` module path is the canonical torchtitan/flame entrypoint as of 2026. If the actual entry differs (check `external/flame/README.md` or `external/flame/flame/__main__.py`), adjust the module name. If flame ships its CLI as `flame-cli` instead, replace the torchrun module form with that binary.

- [ ] **Step 3: Launch the smoke run**

```bash
bash training/scripts/run_stage_a.sh --smoke
```

Expected:
- 4-rank torchrun, each rank holds a 64K CP shard (256K / CP=4).
- 40 steps × ~256K tokens/step = ~10M tokens trained.
- Wall clock: highly variable depending on attention kernels available. Rough estimate at CP=4 + sdpa fallback: 3-10 minutes per step → **2-7 hours total**. With flash-attn installed: 30-60 seconds per step → 20-40 minutes total.
- Checkpoints every 10 steps to `checkpoints/stage_a_1m/step_{10,20,30,40}/`.
- If you see step time > 15 min per step, kill (Ctrl-C) and reconsider: try `--training.activation_checkpoint_mode=full` already on, or drop ctx to 131072 in the [smoke] block.

- [ ] **Step 4: Verify checkpoint produced**

```bash
ls -la checkpoints/stage_a_1m/
# Expect: step_10/, step_20/, step_30/, step_40/, latest/ (symlink)
ls checkpoints/stage_a_1m/latest/
# Expect: model.safetensors-*, config.json, tokenizer files
```

- [ ] **Step 5: Commit the script (checkpoints stay gitignored)**

```bash
git add training/scripts/run_stage_a.sh
git commit -m "feat(training): wire run_stage_a.sh to flame torchrun + Phase 2 smoke" \
    -m "" \
    -m "Calls toml_to_flame_args.py to translate the Stage A TOML into" \
    -m "flame/torchtitan CLI flags, then torchruns flame.train across" \
    -m "NPROC_PER_NODE=4 on CUDA_VISIBLE_DEVICES=0,1,2,3. --smoke selects the" \
    -m "[smoke] phase block from the TOML (256K context, 40 steps, ~10M tokens)." \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Smoke checkpoint round-trips through eval (RULER 128K)

**Files:** none new.

- [ ] **Step 1: Verify ckpt loads as a HF model**

```bash
uv run python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
ckpt = "checkpoints/stage_a_1m/latest"
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16,
                                            device_map={"": 0}, trust_remote_code=True,
                                            attn_implementation="sdpa")
inp = tok("Hello, this is a smoke check.", return_tensors="pt").to(mdl.device)
out = mdl.generate(**inp, max_new_tokens=8, do_sample=False)
print("smoke decode:", tok.decode(out[0]))
PY
```

Expected: prints a coherent continuation. No errors. Confirms flame's output is in HF-compatible format.

- [ ] **Step 2: Re-launch vLLM, this time pointing at the smoke ckpt**

```bash
MODEL=checkpoints/stage_a_1m/latest \
    bash eval/runners/vllm_server.sh > /tmp/vllm_smoke.log 2>&1 &
echo $! > /tmp/vllm_smoke.pid
until grep -q "Uvicorn running" /tmp/vllm_smoke.log 2>/dev/null; do sleep 5; done
echo "[vllm-smoke] ready"
```

- [ ] **Step 3: Run RULER 128K against the smoke ckpt**

```bash
uv run python eval/runners/run_via_vllm.py --bench ruler \
    --lengths 131072 --tasks niah_single_1 niah_multikey_1 niah_multiquery vt qa_1 \
    --n-per-task 10 \
    --out-dir eval/_outputs/vllm/stage_a_smoke_ruler_131072
```

- [ ] **Step 4: Compare against `phase1-baseline`**

```bash
uv run python - <<'PY'
import json
base = json.load(open("docs/reports/phase1_artifacts/ruler_128k_native.json"))
post = json.load(open("eval/_outputs/vllm/stage_a_smoke_ruler_131072/metrics.json"))
print(f"baseline overall: {base['accuracy']:.2%}")
print(f"smoke    overall: {post['accuracy']:.2%}")
print()
for t in sorted(base.get("by_task", {})):
    b = base["by_task"][t]
    p = post.get("by_task", {}).get(t, float("nan"))
    delta = (p - b) * 100
    print(f"  {t:20s}: base={b:.2%}  smoke={p:.2%}  delta={delta:+.1f}pp")
PY
```

Expected behavior: Phase 2 smoke is only ~10M tokens, so big improvement is *not* expected. The success criterion is **no regression > 5pp on overall accuracy** — that proves the training stack didn't damage the model. Improvement on `vt` (even a few pp) is a positive signal that the data + training loop is doing the right thing.

If overall drops by > 10pp, something is wrong with the training: rollback to `phase1-baseline` tag, inspect TOML/wrapper/data, do not proceed to Phase 3 main run.

- [ ] **Step 5: Stop the vLLM server**

```bash
make vllm-stop
```

- [ ] **Step 6: Stash smoke metrics**

```bash
mkdir -p docs/reports/phase2_artifacts
cp eval/_outputs/vllm/stage_a_smoke_ruler_131072/metrics.json docs/reports/phase2_artifacts/ruler_128k_after_smoke.json
cp eval/_outputs/vllm/stage_a_smoke_ruler_131072/summary.md  docs/reports/phase2_artifacts/ruler_128k_after_smoke.md
git add docs/reports/phase2_artifacts/
git commit -m "eval(phase2): RULER 128K after 256K smoke CPT (~10M tokens)" \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: PHASE2_REPORT.md + tag

**Files:**
- Create: `docs/reports/PHASE2_REPORT.md`
- Modify: `README.md` (add link)

- [ ] **Step 1: Write the report**

Create `docs/reports/PHASE2_REPORT.md`:

```markdown
# Phase 2 Report — 1M Eval Unblocked + Data v0 + Stage A Smoke

| Field | Value |
|---|---|
| Date | 2026-05-XX |
| Tag | `phase2-smoke` |
| Hardware | 4× H100 80GB (cards 0-3); single node |
| Inference engine | vLLM TP=4 (for ≥256K eval), transformers SDPA (for <256K) |
| Training | flame (torchtitan + FLA) CP=4 |
| Smoke data | arXiv + PG-19, ~10M tokens packed into 256K sequences |

## What Phase 2 unblocked

1. **1M NIAH baseline now measured** via vLLM TP=4 — see `docs/reports/phase1_artifacts/niah_1m_yarn.json`. Phase 1's BLOCKED note moved to `*_RESOLVED.md`.
2. **Data ingestion + multi-doc packing** end-to-end, gitignored artifacts at `data/processed/packed_256k.jsonl`.
3. **flame training stack** verified via 40-step / 256K / ~10M-token smoke run on cards 0-3 (CP=4). HF-format checkpoint round-trips back into transformers and vLLM.

## Headline numbers

(Fill in from the actual run outputs:)

| Eval | phase1-baseline | After smoke | Δ |
|---|---|---|---|
| NIAH @ 32K          | 100% | n/a (not re-run) | — |
| NIAH @ 128K         | 100% | n/a | — |
| NIAH @ 512K (YaRN)  | 100% | n/a | — |
| **NIAH @ 1M (YaRN)**| BLOCKED | **TODO** | (now measurable) |
| RULER 128K overall  | 84%  | TODO | TODO |
| RULER 128K `vt`     | 20%  | TODO | TODO |

(After running Task 6 and Task 12, replace each TODO with the value from the
corresponding metrics.json file.)

## What changed mechanically

- `pyproject.toml` `[project.optional-dependencies].inference` extra installed (vllm + sglang + http stack).
- `eval/runners/vllm_server.sh` — single-command server launcher.
- `eval/runners/_vllm_client.py` — minimal OpenAI-compatible HTTP client.
- `eval/runners/run_via_vllm.py` — NIAH/RULER driver against a running server.
- `data/scripts/prepare_long_docs.py` — arxiv + pg19 ingestion (was stub).
- `data/scripts/pack_docs.py` — multi-doc 256K packing with `<doc>` separators (was stub).
- `training/flame_wrapper/toml_to_flame_args.py` — TOML → flame CLI translation.
- `training/scripts/run_stage_a.sh` — actual torchrun + flame launch (was echo placeholder).
- `configs/training/stage_a_1m_cpt.toml` — CP=4 (was CP=16 from the spec's 2×8 IB assumption); new `[smoke]` phase block.

## Decisions taken

- **CP=4, not CP=16.** Adjusted from the spec to the user's cards-0-3 GPU policy. Phase 3 may revisit if more cards open up.
- **vLLM over flash-attn for eval.** Flash-attn wheels for torch 2.11+cu130 weren't available; vLLM's own kernels handle 1M serving with TP=4 cleanly. Training (Phase 3 main run) may still want flash-attn — track that separately.
- **Stage A smoke = 256K, not 1M.** 256K smoke runs in tens of minutes on 4×H100 and exercises the same code path; 1M smoke would consume hours. Phase 3 starts the real 1M CPT.

## Quality gate carry-overs into Phase 3

Phase 1 set:
- RULER 128K `vt`: 20% → ≥ **60%** (the headline metric).
- RULER 128K overall: 84% → ≥ **92%**.
- NIAH @ 1M: now measured; target ≥ baseline + **10pp** after Stage A main run.
- Short benchmarks (MMLU/GSM8K/HumanEval): within 0.02 of base, to be measured in Phase 3 with Stage D short SFT.

## Repo state at tag

- 35 commits on `main`
- 35 tests passing (`uv run pytest tests/ -q`)
- ruff clean across `src/`, `tests/`, `eval/runners/`, `data/scripts/`, `training/flame_wrapper/`
- External cloned: `flame`, `flash-linear-attention`, `LLaMA-Factory`, `RULER`
- Local model: `/home/user01/Minko/models/Qwen3.5-4B/`
- Smoke ckpt: `checkpoints/stage_a_1m/latest/`

## Phase 2 done

All 13 tasks of `docs/superpowers/plans/2026-05-18-phase2-data-stage-a-smoke.md` complete. Tagged `phase2-smoke`. Phase 3 plan to be written by next `writing-plans` pass — primary scope is Stage A 1M main run (250M tokens), full data pipeline (~1B tokens across all sources), and the eval delta against this report.
```

- [ ] **Step 2: Fill in the TODOs from real metric files**

Read `docs/reports/phase1_artifacts/niah_1m_yarn.json`, `eval/_outputs/vllm/*_ruler_131072/metrics.json` (the post-baseline-rerun on vLLM), and `docs/reports/phase2_artifacts/ruler_128k_after_smoke.json`. Replace each TODO with the corresponding value. Use the same helpers from Phase 1's report-fill step:

```bash
uv run python - <<'PY'
import json, glob
def load(p):
    paths = glob.glob(p) if "*" in p else [p]
    return json.load(open(paths[0])) if paths else None
baseline_ruler = load("docs/reports/phase1_artifacts/ruler_128k_native.json")
vllm_baseline_ruler = load("docs/reports/phase1_artifacts/ruler_128k_vllm.json")
niah_1m = load("docs/reports/phase1_artifacts/niah_1m_yarn.json")
smoke_ruler = load("docs/reports/phase2_artifacts/ruler_128k_after_smoke.json")
print("baseline (transformers) RULER overall:", baseline_ruler["accuracy"])
print("baseline (transformers) RULER vt     :", baseline_ruler.get("by_task", {}).get("vt"))
print("baseline (vllm)         RULER overall:", vllm_baseline_ruler["accuracy"] if vllm_baseline_ruler else "n/a")
print("NIAH @ 1M               accuracy     :", niah_1m["accuracy"] if niah_1m else "n/a")
print("smoke ckpt              RULER overall:", smoke_ruler["accuracy"] if smoke_ruler else "n/a")
print("smoke ckpt              RULER vt     :", smoke_ruler.get("by_task", {}).get("vt") if smoke_ruler else "n/a")
PY
```

Paste the printed numbers into the markdown table.

- [ ] **Step 3: Add link from README**

Open `README.md`. In the "评测报告" section that was added in Phase 1, append:

```markdown
- [`docs/reports/PHASE2_REPORT.md`](docs/reports/PHASE2_REPORT.md) —— Phase 2
  报告：vLLM 解 1M 评测、数据 pipeline v0、Stage A 256K smoke ckpt。
  Tag `phase2-smoke`.
```

- [ ] **Step 4: Commit + tag + push**

```bash
git add docs/reports/PHASE2_REPORT.md README.md
git commit -m "docs(phase2): Phase 2 report — 1M eval unblocked, smoke checkpoint" \
    -m "" \
    -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git tag phase2-smoke -m "Phase 2 smoke complete (vLLM 1M eval, data v0, flame 256K ckpt)"
git push origin main
git push origin phase2-smoke
```

---

## Phase 2 Done Definition

- [ ] All tests pass: `uv run pytest tests/ -q` → ≥ 35 passed
- [ ] All commits land on `main` and pushed to `origin/main`
- [ ] Tag `phase2-smoke` exists locally and on origin
- [ ] `docs/reports/PHASE2_REPORT.md` filled (no TODOs)
- [ ] `eval/_outputs/vllm/*_niah_1048576/metrics.json` exists (1M baseline now real)
- [ ] `data/processed/packed_256k.jsonl` exists
- [ ] `checkpoints/stage_a_1m/latest/` round-trips via HF transformers + vLLM
- [ ] RULER 128K after smoke does not regress more than 5pp overall vs baseline
- [ ] flame, flash-linear-attention installed editable in venv

Phase 3 (`docs/superpowers/plans/<date>-phase3-stage-a-main.md`, written by next `writing-plans` pass) starts from this tag and runs Stage A main (1M CPT, 250M tokens) plus the full data pipeline.
