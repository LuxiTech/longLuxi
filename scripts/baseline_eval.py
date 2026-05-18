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

    if args.yarn_config:
        yp = Path(args.yarn_config)
        yarn_path: Path | None = yp if yp.is_absolute() else (REPO_ROOT / yp)
    else:
        yarn_path = None

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
