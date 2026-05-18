"""W1 baseline NIAH eval for Qwen3.5-4B (or any HF causal LM) with optional YaRN scaling.

设计：
- 不强依赖 torch/transformers/vllm；缺时退到 --dry-run
- 真实跑：用 transformers + flash-attention 直接加载模型；vLLM 路径作为可选
- 输出：eval/_outputs/baseline/<model_id>_<task>_<len>/{metrics.json, predictions.jsonl, summary.md}

用法：
    # dry run（看会跑啥，不加载模型）
    uv run python scripts/baseline_eval.py --task niah --max-len 131072 --limit 4 --dry-run

    # 真跑：默认 transformers backend, YaRN config 1M
    uv run python scripts/baseline_eval.py \
        --model-id Qwen/Qwen3.5-4B-Instruct \
        --task niah \
        --max-len 1010000 \
        --yarn-config configs/yarn/yarn_1m.json \
        --limit 8 \
        --backend transformers
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from longluxi.paths import EVAL_OUTPUTS_DIR, REPO_ROOT  # noqa: E402

# 一段稳定的英文 haystack 来源；首次运行会下载（小，~1MB）
PAUL_GRAHAM_FALLBACK = """\
The most surprising thing I've learned from being a writer is how rarely the way I think
something is going to come out matches the way it actually does. Whenever you write
something, you imagine how it'll read. But the imagined version and the actual version
diverge in unpredictable ways.
"""


@dataclass
class NiahCell:
    length_tokens: int
    depth_pct: int
    needle: str
    expected: str
    haystack: str
    prompt: str


def _hash(s: str, n: int = 8) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def _load_haystack_corpus(target_tokens: int, approx_token_per_word: float = 1.3) -> str:
    """Load enough English text to fill the haystack. Tries datasets; falls back to PG."""
    try:
        from datasets import load_dataset
        ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        chunks = []
        words = 0
        for ex in ds:
            txt = ex.get("text", "")
            if not txt:
                continue
            chunks.append(txt)
            words += len(txt.split())
            if words * approx_token_per_word > target_tokens * 1.2:
                break
        return "\n\n".join(chunks)
    except Exception as e:  # noqa: BLE001
        # 离线 / 无 datasets — 用 PG 重复
        print(f"[baseline] datasets fallback ({e}); using small PG snippet, will repeat to fill", flush=True)
        return (PAUL_GRAHAM_FALLBACK + "\n") * max(1, int(target_tokens * 1.3 / 60))


def _build_niah_cell(haystack_text: str, length_tokens: int, depth_pct: int,
                     tokenizer, rng: random.Random) -> NiahCell:
    """Embed a 'magic number' needle at given depth percentage."""
    secret = f"{rng.randint(10**9, 10**10 - 1)}"
    needle = f"The magic number is {secret}."
    # Tokenize haystack, take first `length_tokens - margin`, insert needle at depth%.
    margin = 200
    target = max(length_tokens - margin, 1024)
    hay_ids = tokenizer(haystack_text, return_tensors=None, add_special_tokens=False)["input_ids"]
    if len(hay_ids) < target:
        # repeat
        rep = (target // max(len(hay_ids), 1)) + 1
        hay_ids = hay_ids * rep
    hay_ids = hay_ids[:target]
    needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]
    insert_pos = int(len(hay_ids) * depth_pct / 100)
    full_ids = hay_ids[:insert_pos] + needle_ids + hay_ids[insert_pos:]
    haystack = tokenizer.decode(full_ids)
    prompt = (
        "You are a helpful assistant. Read the text and answer the question precisely.\n\n"
        f"<text>\n{haystack}\n</text>\n\n"
        "Question: What is the magic number mentioned in the text?\n"
        "Answer with just the number, nothing else."
    )
    return NiahCell(
        length_tokens=length_tokens,
        depth_pct=depth_pct,
        needle=needle,
        expected=secret,
        haystack=haystack,
        prompt=prompt,
    )


def _apply_yarn_to_config(model_config, yarn_config_path: Path) -> None:
    yarn = json.loads(yarn_config_path.read_text())
    rope_params = yarn["rope_parameters"]
    # Qwen3.5 uses `rope_parameters` (multi-RoPE); older HF uses `rope_scaling`.
    if hasattr(model_config, "rope_parameters"):
        model_config.rope_parameters = rope_params
    else:
        model_config.rope_scaling = {
            "rope_type": rope_params["rope_type"],
            "factor": rope_params["factor"],
            "original_max_position_embeddings": rope_params["original_max_position_embeddings"],
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = EVAL_OUTPUTS_DIR / "baseline" / f"{_hash(args.model_id)}_{args.task}_{args.max_len}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    lengths = args.lengths or [args.max_len]
    depths = args.depths

    plan = {
        "model_id": args.model_id,
        "task": args.task,
        "max_len": args.max_len,
        "yarn_config": args.yarn_config,
        "lengths": lengths,
        "depths": depths,
        "n_per_cell": args.n_per_cell,
        "limit": args.limit,
        "backend": args.backend,
        "out_dir": str(out_dir),
    }
    print(json.dumps({"plan": plan}, indent=2, ensure_ascii=False))

    if args.dry_run:
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        return {"ok": True, "dry_run": True, **plan}

    # Heavy imports gated to keep --dry-run lightweight
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[baseline] heavy deps missing ({e}); install with: uv sync --extra eval", file=sys.stderr)
        return {"ok": False, "reason": "missing_deps"}

    print(f"[baseline] loading tokenizer {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # YaRN
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    if args.yarn_config:
        yarn_path = REPO_ROOT / args.yarn_config if not Path(args.yarn_config).is_absolute() else Path(args.yarn_config)
        _apply_yarn_to_config(config, yarn_path)
        print(f"[baseline] applied YaRN from {yarn_path}", flush=True)

    print(f"[baseline] loading model {args.model_id} (bf16)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.eval()

    # Build cells
    haystack_text = _load_haystack_corpus(max(lengths))

    cells = []
    for L in lengths:
        for d in depths:
            for _ in range(args.n_per_cell):
                cells.append(_build_niah_cell(haystack_text, L, d, tokenizer, rng))
            if args.limit and len(cells) >= args.limit:
                break
        if args.limit and len(cells) >= args.limit:
            break
    if args.limit:
        cells = cells[: args.limit]

    print(f"[baseline] running {len(cells)} cells", flush=True)
    predictions = []
    correct = 0
    for i, cell in enumerate(cells):
        inputs = tokenizer(cell.prompt, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        pred = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        ok = cell.expected in pred
        correct += int(ok)
        predictions.append({
            "length_tokens": cell.length_tokens,
            "depth_pct": cell.depth_pct,
            "expected": cell.expected,
            "pred": pred,
            "ok": ok,
            "input_tokens": n_in,
        })
        print(f"[baseline]  cell {i+1}/{len(cells)} L={cell.length_tokens} d={cell.depth_pct}% expected={cell.expected} pred={pred!r} ok={ok}", flush=True)

    acc = correct / max(len(cells), 1)
    metrics = {
        "model_id": args.model_id,
        "yarn_config": args.yarn_config,
        "n_cells": len(cells),
        "accuracy": acc,
        "by_length": {},
    }
    # by-length breakdown
    by_len: dict[int, list[bool]] = {}
    for p in predictions:
        by_len.setdefault(p["length_tokens"], []).append(p["ok"])
    for L, oks in by_len.items():
        metrics["by_length"][str(L)] = sum(oks) / len(oks)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / "predictions.jsonl").open("w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    summary_lines = [f"# Baseline NIAH — {args.model_id}", ""]
    summary_lines.append(f"- Accuracy: **{acc:.2%}** ({correct}/{len(cells)})")
    summary_lines.append(f"- YaRN: `{args.yarn_config}`" if args.yarn_config else "- YaRN: none")
    summary_lines.append("")
    summary_lines.append("## By length")
    for L, oks in sorted(by_len.items()):
        summary_lines.append(f"- {L:>10} tokens — {sum(oks)/len(oks):.2%} ({sum(oks)}/{len(oks)})")
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")
    print(f"[baseline] done. accuracy={acc:.2%}. results -> {out_dir}", flush=True)
    return {"ok": True, **metrics}


def parse_len(s: str) -> int:
    s = s.strip().lower()
    mult = 1
    if s.endswith("k"):
        mult = 1024; s = s[:-1]
    elif s.endswith("m"):
        mult = 1024 * 1024; s = s[:-1]
    return int(float(s) * mult)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B-Instruct")
    p.add_argument("--task", choices=["niah"], default="niah", help="MVP only supports niah; ruler etc. via eval/runners/")
    p.add_argument("--max-len", type=parse_len, default=131072)
    p.add_argument("--lengths", nargs="+", type=parse_len, default=None,
                   help="override grid lengths; default uses [max_len]")
    p.add_argument("--depths", nargs="+", type=int, default=[10, 50, 90])
    p.add_argument("--n-per-cell", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="cap total cells (for quick smoke)")
    p.add_argument("--yarn-config", default=None,
                   help="path to YaRN config json; e.g. configs/yarn/yarn_1m.json")
    p.add_argument("--backend", choices=["transformers"], default="transformers",
                   help="vllm backend coming W3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    result = run(args)
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
