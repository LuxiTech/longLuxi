"""RULER benchmark runner.

13 tasks × multiple lengths. Use the upstream RULER repo or HF mirror.
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lengths", nargs="+", default=["128k", "1m"])
    parser.add_argument("--tasks", nargs="+", default=["niah_single_1", "niah_multikey_1", "niah_multiquery", "vt", "qa_1"])
    parser.add_argument("--n-per-task", type=int, default=50)
    parser.add_argument("--out-dir", default="eval/_outputs/ruler")
    args = parser.parse_args()
    # TODO[W2]: pull RULER tasks from https://github.com/NVIDIA/RULER, run.
    raise NotImplementedError("W2 task: implement RULER runner")


if __name__ == "__main__":
    main()
