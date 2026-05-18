"""Rerank candidate chunks with bge-reranker-v2-m3 (default) → top-40.

Pipeline config: configs/retrieval/pipeline.yaml#rerank.
"""
from __future__ import annotations


def rerank(query: str, candidates: list[dict], top_k_out: int = 40,
           model_name: str = "BAAI/bge-reranker-v2-m3"):
    # TODO[W5]: load reranker (FlagEmbedding), score (query, chunk) pairs in batch
    raise NotImplementedError("W5 task: implement rerank")
