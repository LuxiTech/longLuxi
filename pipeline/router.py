"""Router / orchestrator.

Receives user query → query processing → retrieval → compression → packing → reader call → answer.

FastAPI service; reader is upstream via pipeline/serve.py.
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reader-url", default="http://localhost:8001")
    parser.add_argument("--retrieval-config", default="configs/retrieval/pipeline.yaml")
    parser.add_argument("--policy", choices=["rule", "self_route"], default="rule")
    args = parser.parse_args()
    # TODO[W11]: FastAPI app + dispatch logic per policy.
    raise NotImplementedError("W11 task: implement router")


if __name__ == "__main__":
    main()
