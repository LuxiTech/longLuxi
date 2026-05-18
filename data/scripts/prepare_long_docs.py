"""Download + clean + chunk long documents into training-ready jsonl.

Sources covered in MVP:
- arXiv (via `datasets:arxiv_dataset` or HuggingFace `arxiv_full` mirror)
- PG-19 (`datasets:pg19`)
- Pile-of-Law (`datasets:pile-of-law/pile-of-law`)
- Wikipedia long articles (top-1% by length)

Output schema (one jsonl per source under data/processed/):
  {
    "doc_id": str,
    "source": str,             # "arxiv" / "pg19" / "pile_of_law" / "wiki"
    "title": str | None,
    "text": str,
    "n_tokens": int,           # by Qwen3.5 tokenizer
    "metadata": {...}
  }

Usage:
    uv run python data/scripts/prepare_long_docs.py --source arxiv --max-docs 10000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from longluxi.paths import PROCESSED_DATA_DIR, ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["arxiv", "pg19", "pile_of_law", "wiki"], required=True)
    parser.add_argument("--max-docs", type=int, default=10000)
    parser.add_argument("--min-tokens", type=int, default=4096)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    ensure_dirs()
    # TODO[W1]:
    # 1. dispatch on args.source: load HF dataset / dump
    # 2. clean (strip headers/footers/refs depending on source)
    # 3. tokenize with Qwen3.5 tokenizer
    # 4. filter by min_tokens
    # 5. write jsonl to {out_dir}/long_docs_{source}.jsonl
    raise NotImplementedError("W1 task: implement long-doc ingestion per source")


if __name__ == "__main__":
    main()
