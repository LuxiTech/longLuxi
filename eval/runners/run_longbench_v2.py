"""LongBench v2 runner. See https://github.com/THUDM/LongBench."""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--subsets", nargs="+", default=["all"])
    parser.add_argument("--out-dir", default="eval/_outputs/longbench_v2")
    args = parser.parse_args()
    # TODO[W5]: use `datasets:THUDM/LongBench-v2`; per-task metric impl follows official.
    raise NotImplementedError("W5 task")


if __name__ == "__main__":
    main()
