"""NIAH (Needle in a Haystack) runner.

Lengths: configurable, default 128K / 512K / 1M / 2M / 4M
Depths: 10/30/50/70/90 percentile positions of haystack
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lengths", nargs="+", default=["128k", "512k", "1m"])
    parser.add_argument("--depths", nargs="+", type=int, default=[10, 30, 50, 70, 90])
    parser.add_argument("--n-per-cell", type=int, default=4)
    parser.add_argument("--yarn-config", default=None)
    parser.add_argument("--out-dir", default="eval/_outputs/niah")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="vllm")
    args = parser.parse_args()
    # TODO[W1]: shared NIAH logic. See scripts/baseline_eval.py which 已经写好了一个迷你版本.
    # 这里复用其逻辑，扩到完整 lengths × depths 网格。
    raise NotImplementedError("W1 task: factor out from baseline_eval.py")


if __name__ == "__main__":
    main()
