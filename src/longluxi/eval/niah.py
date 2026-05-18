"""NIAH cell construction + grading.

A "cell" = one (length, depth) evaluation instance: haystack with a needle
inserted at a specific depth percentage, plus prompt + expected answer.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class NiahCell:
    length_tokens: int
    depth_pct: int
    needle: str
    expected: str       # the secret number, used for grading
    haystack: str       # haystack including needle
    prompt: str         # full prompt sent to model


_PROMPT_TMPL = (
    "You are a helpful assistant. Read the text and answer the question precisely.\n\n"
    "<text>\n{haystack}\n</text>\n\n"
    "Question: What is the magic number mentioned in the text?\n"
    "Answer with just the number, nothing else."
)


def build_cell(haystack_text: str, length_tokens: int, depth_pct: int,
               tokenizer, rng: random.Random, margin_tokens: int = 200) -> NiahCell:
    """Embed a random 10-digit number at depth_pct of the haystack and build the prompt."""
    secret = f"{rng.randint(10**9, 10**10 - 1)}"
    needle = f"The magic number is {secret}."

    target = max(length_tokens - margin_tokens, 64)
    hay_ids = tokenizer(haystack_text, return_tensors=None, add_special_tokens=False)["input_ids"]
    if len(hay_ids) < target:
        rep = (target // max(len(hay_ids), 1)) + 1
        hay_ids = hay_ids * rep
    hay_ids = hay_ids[:target]

    needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]
    insert_pos = int(len(hay_ids) * depth_pct / 100)
    full_ids = list(hay_ids[:insert_pos]) + list(needle_ids) + list(hay_ids[insert_pos:])
    haystack = tokenizer.decode(full_ids)

    prompt = _PROMPT_TMPL.format(haystack=haystack)
    return NiahCell(
        length_tokens=length_tokens,
        depth_pct=depth_pct,
        needle=needle,
        expected=secret,
        haystack=haystack,
        prompt=prompt,
    )


def grade(cell: NiahCell, prediction: str) -> bool:
    """Pass iff the secret string appears anywhere in the prediction."""
    if not prediction:
        return False
    return cell.expected in prediction
