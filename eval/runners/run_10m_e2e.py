"""10M 端到端 benchmark runner.

跑全部 ablation 系统（A-E in configs/eval/eval_matrix.yaml#ablation_systems）
× 全部 task type (passkey / multi-needle / contradiction / timeline / qfs / multi-hop)。

W11 主要交付。
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/eval_matrix.yaml")
    parser.add_argument("--reader-ckpt", required=True)
    parser.add_argument("--retrieval-config", default="configs/retrieval/pipeline.yaml")
    parser.add_argument("--out-dir", default="eval/_outputs/10m_e2e")
    parser.add_argument("--systems", nargs="+", help="subset of ablation_systems names")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise NotImplementedError("W11 task: 10M e2e runner")


if __name__ == "__main__":
    main()
