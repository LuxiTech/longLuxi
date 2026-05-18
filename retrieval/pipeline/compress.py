"""Query-aware compression.

Input: query + (chunk_text, chunk_summary, metadata)
Output: structured <evidence> XML block (see spec §5.4 / §10.2)

Compressor model is configurable (configs/retrieval/pipeline.yaml#compression):
- W7 起点：Qwen3.5-4B-Instruct (baseline)
- W8+：换成 self-trained reader (Stage E ckpt)
"""
from __future__ import annotations

EVIDENCE_TEMPLATE = """<evidence id="{eid}" source="{src}" relevance="{rel}">
  <claim>{claim}</claim>
  <support_span>{span}</support_span>
  <why_relevant>{why}</why_relevant>
  <entities>{entities}</entities>
  <time_or_version>{tv}</time_or_version>
  <conflicts>{conflicts}</conflicts>
</evidence>"""


def compress_one(query: str, chunk: dict, model_name: str) -> str:
    # TODO[W7]: call compressor model, prompt template = spec §10.2
    raise NotImplementedError("W7 task: implement query-aware compression")
