"""Build BM25 index (OpenSearch) over raw_chunk units (8K tokens each).

Schema follows configs/retrieval/index.yaml#bm25.

Usage:
    uv run python retrieval/index/build_bm25.py \
        --input data/processed/long_docs_chunked.jsonl \
        --index longluxi_raw_chunks \
        --host http://localhost:9200
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--index", default="longluxi_raw_chunks")
    parser.add_argument("--host", default="http://localhost:9200")
    parser.add_argument("--bulk-size", type=int, default=500)
    args = parser.parse_args()
    # TODO[W3]:
    # 1. open OpenSearch client (opensearch-py)
    # 2. (re)create index with mapping from configs/retrieval/index.yaml#bm25
    # 3. bulk index chunks; show progress with tqdm
    # 4. refresh + smoke test query
    raise NotImplementedError("W3 task: implement build_bm25")


if __name__ == "__main__":
    main()
