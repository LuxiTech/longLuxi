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
sys.path.insert(0, str(REPO_ROOT))

from eval.runners._vllm_client import VllmClient  # noqa: E402
from longluxi.eval.haystack import load_haystack_corpus  # noqa: E402
from longluxi.eval.niah import build_cell as build_niah  # noqa: E402
from longluxi.eval.niah import grade as grade_niah  # noqa: E402
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
