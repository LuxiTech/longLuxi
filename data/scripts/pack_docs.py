"""Multi-document packing for long-context CPT.

Goal: produce 64K / 256K / 1M / 2M / 4M sequences by:
  - concatenating same-topic docs with explicit <doc> separators
  - optionally adding `cross_doc_attention_mask` so different docs don't attend across boundary
  - length-upsampling so most sequences exceed the target ctx threshold

Output schema:
  {
    "seq_id": str,
    "target_ctx": int,
    "docs": [{"doc_id": ..., "source": ..., "n_tokens": ...}, ...],
    "text": str,               # full packed sequence (already separator-inserted)
    "n_tokens": int,
    "mask_doc_boundaries": list[int]   # token positions where new doc starts
  }
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ctx", type=int, required=True, help="1048576 / 2097152 / 4194304")
    parser.add_argument("--input-jsonls", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topic-clustering", default="random", choices=["random", "kmeans_bgesm"])
    parser.add_argument("--separator", default='<doc id="{doc_id}" source="{source}">')
    parser.add_argument("--separator-end", default="</doc>")
    args = parser.parse_args()
    # TODO[W2]: implement multi-doc topic clustering + packing
    raise NotImplementedError("W2 task: implement pack_docs")


if __name__ == "__main__":
    main()
