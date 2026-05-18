"""Generate memory-format SFT data via teacher distillation.

Teacher: Claude-Opus or Qwen3-72B-Instruct (configurable).

For each input (query, retrieved_evidence_pack), the teacher produces:
- selected_evidence_ids
- answer with citations
- confidence
- missing_information

Output is the canonical prompt format from spec §6.1.
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="seed queries jsonl")
    parser.add_argument("--retrieval-corpus", required=True, help="corpus to retrieve evidence from")
    parser.add_argument("--teacher", choices=["claude", "qwen3-72b", "gpt-5"], default="claude")
    parser.add_argument("--n-examples", type=int, default=20000)
    parser.add_argument("--task-mix", default="configs/training/stage_e_memory_sft.toml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    # TODO[W9]:
    # 1. for each task type (evidence_selection / answer_w_citation / multi_hop / contradiction / timeline / qfs),
    #    generate query templates + retrieve evidence from corpus
    # 2. call teacher API to produce labeled output
    # 3. format as final SFT example (input + target)
    raise NotImplementedError("W9 task: implement memory-format SFT data generation")


if __name__ == "__main__":
    main()
