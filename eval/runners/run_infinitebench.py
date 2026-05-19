"""InfiniteBench runner driven by a running vLLM server.

xinrongzhang2022/InfiniteBench task subsets we wire here:

  longbook_choice_eng   - real-book multiple choice (4 options), 229 samples,
                          context median 183K tok, max 900K tok. THE
                          headline signal for real long-doc reasoning.
  kv_retrieval          - precise key→value lookup, 500 samples @ 57K tok.
                          Tests multi-key memory.
  passkey               - single needle, 590 samples @ 134K tok.

Each task has its own answer format; we pick the matching grader.
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


# ---------------- prompt templates per task ----------------

_LONGBOOK_CHOICE_TMPL = (
    "Read the following text from a book carefully and then answer the multiple-choice "
    "question. Output only a single letter (A, B, C, or D) — nothing else.\n\n"
    "<book>\n{context}\n</book>\n\n"
    "Question: {question}\n\n"
    "(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\n"
    "Answer:"
)

_KV_TMPL = "Below is a JSON object with key-value pairs.\n\n{context}\n{question}"

_PASSKEY_TMPL = "{context}\n{question}"

_LETTER_RE = re.compile(r"\b([ABCD])\b")
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_DIGITS_RE = re.compile(r"\d+")


# ---------------- graders ----------------

def _grade_longbook_choice(record, pred: str) -> tuple[bool, str]:
    m = _LETTER_RE.search((pred or "").upper())
    letter = m.group(1) if m else ""
    # answer field in this dataset is sometimes the letter, sometimes the option text.
    gold_raw = record["answer"]
    if isinstance(gold_raw, list):
        gold_raw = gold_raw[0]
    gold = str(gold_raw).strip().upper()
    # If gold is one of A/B/C/D, exact-match the letter; otherwise resolve gold-text against options.
    if gold in ("A", "B", "C", "D"):
        return letter == gold, letter
    # Resolve text → letter by matching against options
    options = record.get("options") or []
    for idx, opt in enumerate(options[:4]):
        if str(opt).strip() == str(gold_raw).strip():
            return letter == "ABCD"[idx], letter
    # Fallback: substring of gold in pred
    return str(gold_raw).strip().lower() in (pred or "").lower(), letter


def _grade_kv(record, pred: str) -> tuple[bool, str]:
    gold = record["answer"]
    if isinstance(gold, list):
        gold = gold[0]
    m = _UUID_RE.search(pred or "")
    extracted = m.group(0) if m else ""
    return extracted == gold.strip(), extracted


def _grade_passkey(record, pred: str) -> tuple[bool, str]:
    gold = record["answer"]
    if isinstance(gold, list):
        gold = gold[0]
    m = _DIGITS_RE.search(pred or "")
    extracted = m.group(0) if m else ""
    return extracted == str(gold).strip(), extracted


TASKS = {
    "longbook_choice_eng": {
        "tmpl": _LONGBOOK_CHOICE_TMPL,
        "grader": _grade_longbook_choice,
        "max_new_tokens": 8,
        "build_prompt": lambda r: _LONGBOOK_CHOICE_TMPL.format(
            context=r["context"], question=r["input"],
            A=r["options"][0], B=r["options"][1], C=r["options"][2], D=r["options"][3],
        ),
    },
    "kv_retrieval": {
        "tmpl": _KV_TMPL,
        "grader": _grade_kv,
        "max_new_tokens": 64,
        "build_prompt": lambda r: _KV_TMPL.format(context=r["context"], question=r["input"]),
    },
    "passkey": {
        "tmpl": _PASSKEY_TMPL,
        "grader": _grade_passkey,
        "max_new_tokens": 16,
        "build_prompt": lambda r: _PASSKEY_TMPL.format(context=r["context"], question=r["input"]),
    },
}


def _resolve_served_id(base_url: str) -> str:
    import httpx
    return httpx.get(f"{base_url}/v1/models", timeout=30).json()["data"][0]["id"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS.keys()), required=True)
    p.add_argument("--base-url", default="http://localhost:8001")
    p.add_argument("--max-context-chars", type=int, default=2_800_000,
                   help="skip samples whose context exceeds this many chars (~3.5c/tok).")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    served = _resolve_served_id(args.base_url)
    client = VllmClient(base_url=args.base_url, model=served, timeout=1800.0)
    cfg = TASKS[args.task]

    from huggingface_hub import hf_hub_download
    path = hf_hub_download("xinrongzhang2022/InfiniteBench", f"{args.task}.jsonl",
                           repo_type="dataset")
    records = [json.loads(l) for l in open(path)]
    if args.max_samples:
        records = records[: args.max_samples]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"

    print(f"[infbench:{args.task}] n={len(records)} cap={args.max_context_chars}c served={served}",
          flush=True)

    skipped = 0
    preds: list[dict] = []
    by_bucket: dict[str, list[bool]] = collections.defaultdict(list)

    def ctx_bucket(n: int) -> str:
        # categorize by approximate token count
        if n < 200_000:    return "lt_50k"
        if n < 700_000:    return "50k_180k"
        if n < 1_750_000:  return "180k_500k"
        return "gt_500k"

    with pred_file.open("w") as fp_pred:
        for i, r in enumerate(records):
            if len(r["context"]) > args.max_context_chars:
                skipped += 1
                continue
            prompt = cfg["build_prompt"](r)
            try:
                resp = client.complete_chat(
                    [{"role": "user", "content": prompt}],
                    max_new_tokens=cfg["max_new_tokens"],
                    enable_thinking=False,
                )
            except Exception as exc:
                print(f"[infbench:{args.task}] err idx={i}: {exc}", flush=True)
                continue
            ok, extracted = cfg["grader"](r, resp)
            preds.append({
                "id": r.get("id", i),
                "ctx_chars": len(r["context"]),
                "answer": r["answer"],
                "pred_raw": resp[:120],
                "pred_extracted": extracted,
                "ok": ok,
            })
            by_bucket[ctx_bucket(len(r["context"]))].append(ok)
            fp_pred.write(json.dumps(preds[-1], ensure_ascii=False) + "\n")
            fp_pred.flush()
            print(f"[infbench:{args.task}] {i+1}/{len(records)} "
                  f"ctx={len(r['context'])//1000}Kc ok={ok} "
                  f"pred={extracted[:30]!r}", flush=True)

    metrics = {
        "model_id": served,
        "task": args.task,
        "n_attempted": len(preds),
        "n_skipped_too_long": skipped,
        "accuracy": (sum(p["ok"] for p in preds) / max(len(preds), 1)),
        "by_bucket": {k: sum(v) / len(v) for k, v in by_bucket.items()},
        "n_by_bucket": {k: len(v) for k, v in by_bucket.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
