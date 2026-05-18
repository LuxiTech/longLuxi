"""Build FAISS HNSW dense index using bge-m3 embeddings over raw_chunks.

Usage:
    uv run python retrieval/index/build_dense.py \
        --input data/processed/long_docs_chunked.jsonl \
        --out retrieval/indices/dense_bge_m3 \
        --model BAAI/bge-m3 \
        --batch-size 64
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    # TODO[W3]:
    # 1. load FlagEmbedding / sentence-transformers model
    # 2. iterate jsonl, encode in batch, build FAISS HNSW (M=32, efConstruction=200)
    # 3. save index + id2chunk mapping
    raise NotImplementedError("W3 task: implement build_dense")


if __name__ == "__main__":
    main()
