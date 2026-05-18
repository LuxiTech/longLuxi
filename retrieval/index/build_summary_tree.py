"""Build RAPTOR-style recursive summary tree.

Leaves = 512K sections. Cluster (kmeans on bge-m3 embeddings) -> summarize cluster ->
recurse until single root.

Output: retrieval/indices/summary_tree.json
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", required=True, help="jsonl of 512K sections")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--height", type=int, default=3)
    parser.add_argument("--branch", type=int, default=8)
    parser.add_argument("--summary-max-tokens", type=int, default=1024)
    args = parser.parse_args()
    # TODO[W4]: RAPTOR-style summary tree.
    raise NotImplementedError("W4 task: implement build_summary_tree")


if __name__ == "__main__":
    main()
