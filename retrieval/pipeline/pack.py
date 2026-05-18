"""Order-preserving packing.

After rerank + compression, pack evidence into reader-ready prompt
by original document order (not relevance order).

See spec §9.5.
"""
from __future__ import annotations

SOURCE_HEADER = '<source id="{id}" doc="{doc}" path="{path}" start="{start}" end="{end}" score="{score:.3f}">'


def pack_evidence(evidence_list: list[dict], reader_budget_tokens: int = 1_048_576,
                  order: str = "original") -> str:
    # TODO[W6]:
    # 1. sort evidence by (doc_id, start_token) for original order
    # 2. merge contiguous spans within same doc
    # 3. wrap with SOURCE_HEADER
    # 4. truncate to reader_budget_tokens (LRU by relevance if overflow)
    raise NotImplementedError("W6 task: implement order-preserving packing")
