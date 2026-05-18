"""NoCha (Novel Challenge) — book-length QA. https://novelchallenge.github.io/"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-books", type=int, default=30)
    parser.add_argument("--out-dir", default="eval/_outputs/nocha")
    args = parser.parse_args()
    raise NotImplementedError("W10 task")


if __name__ == "__main__":
    main()
