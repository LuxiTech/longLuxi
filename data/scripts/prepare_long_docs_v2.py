"""Scaled-up long-doc ingestion. Target 100M+ tokens combined.

Output writes to data/processed/v2/ so v1 (6.6M-token smoke corpus) is preserved.

Sources:
- arxiv (ccdv/arxiv-summarization, train split) — scaled to 5000 docs
- pg19 (emozilla/pg19, train split, NOT the test split mirror used in v1)
- (optional) c4 long subset — only if we still need more after arxiv+pg19

Run all three in parallel:
    bash -c "
      uv run python data/scripts/prepare_long_docs_v2.py --source arxiv --max-docs 5000 &
      uv run python data/scripts/prepare_long_docs_v2.py --source pg19  --max-docs 800 &
      wait
    "
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
    ds = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
    print(f"[v2 prepare] arxiv source: ccdv/arxiv-summarization (train, streaming)", flush=True)
    yielded = 0
    skipped_short = 0
    for ex in ds:
        text = (ex.get("article") or ex.get("text") or "").strip()
        if not text:
            continue
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n < min_tokens:
            skipped_short += 1
            continue
        yield {
            "doc_id": f"arxiv/{yielded}",
            "source": "arxiv",
            "title": (ex.get("abstract") or "").strip()[:120] or None,
            "text": text,
            "n_tokens": n,
            "metadata": {"dataset_id": "ccdv/arxiv-summarization"},
        }
        yielded += 1
        if yielded >= max_docs:
            print(f"[v2 prepare] arxiv: stopped at {yielded} (skipped {skipped_short} short docs)", flush=True)
            return


def _iter_pg19(max_docs: int, min_tokens: int, tokenizer):
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    print(f"[v2 prepare] pg19 source: emozilla/pg19 (train, streaming)", flush=True)
    yielded = 0
    skipped_short = 0
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n < min_tokens:
            skipped_short += 1
            continue
        yield {
            "doc_id": f"pg19/{yielded}",
            "source": "pg19",
            "title": ex.get("short_book_title"),
            "text": text,
            "n_tokens": n,
            "metadata": {"publication_date": ex.get("publication_date"), "url": ex.get("url")},
        }
        yielded += 1
        if yielded >= max_docs:
            print(f"[v2 prepare] pg19: stopped at {yielded} (skipped {skipped_short} short docs)", flush=True)
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["arxiv", "pg19"], required=True)
    p.add_argument("--max-docs", type=int, default=2000)
    p.add_argument("--min-tokens", type=int, default=4096)
    p.add_argument("--tokenizer", default="/home/user01/Minko/models/Qwen3.5-4B")
    p.add_argument("--out-dir", type=Path,
                   default=PROCESSED_DATA_DIR / "v2")
    args = p.parse_args()

    ensure_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"long_docs_{args.source}.jsonl"
    print(f"[v2 prepare] writing -> {out_path}", flush=True)

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
            if n_docs % 50 == 0:
                print(f"[v2 prepare] {n_docs}/{args.max_docs} docs, {n_tokens/1e6:.2f}M tokens",
                      flush=True)
    print(f"[v2 prepare] done: {n_docs} docs, {n_tokens/1e6:.2f}M tokens -> {out_path}")


if __name__ == "__main__":
    main()
