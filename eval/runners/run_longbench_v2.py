"""LongBench-v2 evaluation driven by a running vLLM server.

THUDM/LongBench-v2: 503 multiple-choice questions on long documents.
Length buckets: short (<32K tok), medium (32K-128K), long (>128K).
We default to the `long` subset for the 1M capability characterization.

Each question gives 4 choices A/B/C/D; we score exact-match on the letter.

Note: ~23% of `long` bucket has contexts > 1M tokens; those are skipped via
--max-context-chars to avoid 400 from vLLM. Effective long-bucket N ≈ 83 at
the default 800K-tok cap.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.runners._vllm_client import VllmClient  # noqa: E402

PROMPT_TMPL = (
    "Please read the following long document carefully and then answer the multiple-choice "
    "question. Output only a single letter (A, B, C, or D) — nothing else.\n\n"
    "<document>\n{context}\n</document>\n\n"
    "Question: {question}\n\n"
    "(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\n"
    "Answer:"
)

_LETTER_RE = re.compile(r"\b([ABCD])\b")


def _extract_letter(pred: str) -> str:
    pred = (pred or "").strip()
    m = _LETTER_RE.search(pred.upper())
    return m.group(1) if m else ""


def _resolve_served_id(base_url: str) -> str:
    import httpx
    data = httpx.get(f"{base_url}/v1/models", timeout=30).json()
    return data["data"][0]["id"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8001")
    p.add_argument("--length-bucket", choices=["short", "medium", "long", "all"],
                   default="long")
    p.add_argument("--max-context-chars", type=int, default=2_800_000,
                   help="skip samples whose context exceeds this many chars (~3.5 chars/tok). "
                        "Default 2.8M ~ 800K tokens, fits 1M model with chat-template headroom.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=8)
    args = p.parse_args()

    served = _resolve_served_id(args.base_url)
    client = VllmClient(base_url=args.base_url, model=served, timeout=1800.0)

    from datasets import load_dataset
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    if args.length_bucket != "all":
        ds = [r for r in ds if r["length"] == args.length_bucket]
    else:
        ds = list(ds)

    if args.max_samples:
        ds = ds[: args.max_samples]

    skipped = 0
    preds: list[dict] = []
    by_domain: dict[str, list[bool]] = collections.defaultdict(list)
    by_difficulty: dict[str, list[bool]] = collections.defaultdict(list)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"

    print(f"[longbench-v2] bucket={args.length_bucket} n={len(ds)} cap={args.max_context_chars}c "
          f"served={served}", flush=True)

    with pred_file.open("w") as fp_pred:
        for i, r in enumerate(ds):
            if len(r["context"]) > args.max_context_chars:
                skipped += 1
                continue
            prompt = PROMPT_TMPL.format(
                context=r["context"],
                question=r["question"],
                A=r["choice_A"], B=r["choice_B"], C=r["choice_C"], D=r["choice_D"],
            )
            try:
                resp = client.complete_chat(
                    [{"role": "user", "content": prompt}],
                    max_new_tokens=args.max_new_tokens,
                    enable_thinking=False,
                )
            except Exception as exc:
                print(f"[longbench-v2] err idx={i} id={r['_id']}: {exc}", flush=True)
                continue
            letter = _extract_letter(resp)
            ok = letter == r["answer"]
            preds.append({
                "_id": r["_id"],
                "domain": r["domain"],
                "sub_domain": r["sub_domain"],
                "difficulty": r["difficulty"],
                "length": r["length"],
                "ctx_chars": len(r["context"]),
                "answer": r["answer"],
                "pred_raw": resp[:120],
                "pred_letter": letter,
                "ok": ok,
            })
            by_domain[r["domain"]].append(ok)
            by_difficulty[r["difficulty"]].append(ok)
            fp_pred.write(json.dumps(preds[-1], ensure_ascii=False) + "\n")
            fp_pred.flush()
            print(f"[longbench-v2] {i+1}/{len(ds)} id={r['_id']} ctx={len(r['context'])//1000}Kchars "
                  f"ans={r['answer']} pred={letter or '?'} ok={ok}", flush=True)

    metrics = {
        "model_id": served,
        "bucket": args.length_bucket,
        "n_attempted": len(preds),
        "n_skipped_too_long": skipped,
        "accuracy": (sum(p["ok"] for p in preds) / max(len(preds), 1)),
        "by_domain": {k: sum(v) / len(v) for k, v in by_domain.items()},
        "by_difficulty": {k: sum(v) / len(v) for k, v in by_difficulty.items()},
        "n_by_domain": {k: len(v) for k, v in by_domain.items()},
        "n_by_difficulty": {k: len(v) for k, v in by_difficulty.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
