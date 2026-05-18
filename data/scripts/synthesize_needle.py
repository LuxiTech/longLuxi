"""Synthesize NIAH / multi-needle training data.

Used both:
- as training data (small fraction of CPT mix, see spec §4.4)
- as eval data (handled separately by eval/runners/run_niah.py)
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--lengths", nargs="+", type=int, default=[131072, 524288, 1048576, 2097152])
    parser.add_argument("--depth-percentages", nargs="+", type=int, default=[10, 30, 50, 70, 90])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    # TODO[W2]:
    # 1. random English text (Paul Graham essays or random Wikipedia) as haystack
    # 2. insert N needles at specified depth percentages
    # 3. question = "What is the special number/secret embedded in the text?"
    raise NotImplementedError("W2 task: implement synthesize_needle")


if __name__ == "__main__":
    main()
