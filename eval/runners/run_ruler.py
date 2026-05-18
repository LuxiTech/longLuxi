"""RULER benchmark runner (MVP subset: 5 tasks)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

# GPU policy: only cards 0-3 (4-7 reserved). Override by exporting CUDA_VISIBLE_DEVICES.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

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
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--lengths", nargs="+", type=parse_len, default=[131072])
    p.add_argument("--tasks", nargs="+", default=list(TASK_REGISTRY.keys()))
    p.add_argument("--n-per-task", type=int, default=10)
    p.add_argument("--yarn-config", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="eval/_outputs/ruler")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=128)
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
    from longluxi.eval.runner import _format_for_model
    yarn_path = (REPO_ROOT / args.yarn_config) if (args.yarn_config and not Path(args.yarn_config).is_absolute()) else args.yarn_config
    model, tokenizer, _cfg = load_model_and_tokenizer(args.model_id, yarn_path)

    import torch

    rng = random.Random(args.seed)
    predictions: list[dict] = []
    for L in args.lengths:
        for task_name in args.tasks:
            for _ in range(args.n_per_task):
                cell = build_ruler_cell(task_name, L, tokenizer, rng)
                text = _format_for_model(cell["prompt"], tokenizer)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                n_in = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
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
    by_task: dict[str, list[bool]] = {}
    for p_ in predictions:
        by_task.setdefault(p_["task"], []).append(p_["ok"])
    metrics["by_task"] = {t: (sum(o) / len(o)) for t, o in by_task.items()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
