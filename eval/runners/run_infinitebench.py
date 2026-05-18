"""InfiniteBench runner. https://github.com/OpenBMB/InfiniteBench"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--subsets", nargs="+",
                        default=["passkey", "kv_retrieval", "longbook_qa_eng",
                                 "longdialogue_qa_eng", "code_debug", "math_calc"])
    parser.add_argument("--out-dir", default="eval/_outputs/infinitebench")
    args = parser.parse_args()
    raise NotImplementedError("W5 task")


if __name__ == "__main__":
    main()
