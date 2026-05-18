"""Download + clean + tokenize long documents into training-ready jsonl.

MVP scope (Phase 2 smoke): ~5-15M tokens total, two sources:
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

    # Fallback order:
    # 1. ccdv/arxiv-summarization (Parquet-based, no script)
    # 2. allenai/peS2o (peer-reviewed papers, field: text)
    dataset_id = "ccdv/arxiv-summarization"
    try:
        ds = load_dataset(dataset_id, split="train", streaming=True)
        # verify fields exist by peeking
        def _get_text(ex):
            # try article first, then text
            t = ex.get("article") or ex.get("text") or ""
            return t.strip()
        def _get_title(ex):
            ab = ex.get("abstract") or ""
            return ab.strip()[:120] or None
    except Exception as e1:
        print(f"[prepare] fallback1 failed ({e1}), trying allenai/peS2o", flush=True)
        dataset_id = "allenai/peS2o"
        ds = load_dataset(dataset_id, split="train", streaming=True)
        def _get_text(ex):
            return (ex.get("text") or "").strip()
        def _get_title(ex):
            t = ex.get("title") or ex.get("abstract") or ""
            return t.strip()[:120] or None

    print(f"[prepare] arxiv source: {dataset_id}", flush=True)
    yielded = 0
    for ex in ds:
        text = _get_text(ex)
        if not text:
            continue
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n < min_tokens:
            continue
        yield {
            "doc_id": f"arxiv/{yielded}",
            "source": "arxiv",
            "title": _get_title(ex),
            "text": text,
            "n_tokens": n,
            "metadata": {"dataset_id": dataset_id},
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
