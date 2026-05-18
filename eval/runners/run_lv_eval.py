"""LV-Eval 中文长文本 benchmark sanity. https://github.com/infinigence/LVEval"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--subsets", nargs="+", default=["all"])
    parser.add_argument("--out-dir", default="eval/_outputs/lv_eval")
    args = parser.parse_args()
    raise NotImplementedError("W10 task")


if __name__ == "__main__":
    main()
