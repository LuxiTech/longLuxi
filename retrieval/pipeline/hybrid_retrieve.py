"""Hybrid retrieval: BM25 + Dense + Summary tree → dedupe + union → candidate set.

Pipeline config: configs/retrieval/pipeline.yaml#recall.
"""
from __future__ import annotations

import argparse


def hybrid_retrieve(query: str, top_k: int = 400):
    # TODO[W3-5]:
    # 1. issue BM25 top-200
    # 2. issue dense top-200
    # 3. summary_tree top-100
    # 4. dedupe + union -> top_k
    raise NotImplementedError("W3-5 task")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=400)
    args = parser.parse_args()
    res = hybrid_retrieve(args.query, args.top_k)
    print(res)


if __name__ == "__main__":
    main()
