"""RULER task registry and cell builder.

MVP scope (W1): 5 tasks at 128K to validate harness.
- niah_single_1 (1 needle)
- niah_multikey_1 (1 key, 3 distractors)
- niah_multiquery (4 needles, 1 query at a time)
- vt (variable tracking: foo=bar, bar=42, what is foo?)
- qa_1 (HotpotQA-style on synthetic doc)

Full RULER (13 tasks) handled in Phase 2 once we have ckpts to compare.

Upstream definitions: external/RULER/scripts/data/synthetic/
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from longluxi.eval.haystack import load_haystack_corpus
from longluxi.eval.niah import build_cell as _build_niah


@dataclass
class RulerTask:
    name: str
    build_fn: Callable[..., dict[str, Any]]
    grade_fn: Callable[[dict[str, Any], str], bool]


# ------------- task implementations -------------

def _build_niah_single(length_tokens, tokenizer, rng):
    haystack = load_haystack_corpus(length_tokens, allow_network=True)
    c = _build_niah(haystack, length_tokens, depth_pct=rng.choice([20, 50, 80]),
                    tokenizer=tokenizer, rng=rng)
    return {"task": "niah_single_1", "prompt": c.prompt, "expected": c.expected,
            "length_tokens": c.length_tokens, "depth_pct": c.depth_pct}


def _grade_niah(cell, pred):
    return cell["expected"] in (pred or "")


def _build_niah_multikey(length_tokens, tokenizer, rng):
    """1 real magic number + 3 distractor magic numbers; ask for THE magic number."""
    haystack = load_haystack_corpus(length_tokens, allow_network=True)
    real = f"{rng.randint(10**9, 10**10 - 1)}"
    distractors = [f"{rng.randint(10**9, 10**10 - 1)}" for _ in range(3)]
    hay_ids = tokenizer(haystack, add_special_tokens=False)["input_ids"]
    real_line = f" The magic number is {real}. "
    real_ids = tokenizer(real_line, add_special_tokens=False)["input_ids"]
    distract_ids = [tokenizer(f" Note: {d} is a random tracking id. ",
                              add_special_tokens=False)["input_ids"] for d in distractors]
    target = max(length_tokens - 300, 64)
    if len(hay_ids) < target:
        hay_ids = (hay_ids * ((target // len(hay_ids)) + 1))[:target]
    hay_ids = hay_ids[:target]
    positions = sorted([int(target * p) for p in (0.2, 0.4, 0.6, 0.8)])
    inserts = list(zip(positions, distract_ids + [real_ids], strict=False))
    out_ids: list = []
    last = 0
    for pos, ids in inserts:
        out_ids.extend(hay_ids[last:pos])
        out_ids.extend(ids)
        last = pos
    out_ids.extend(hay_ids[last:])
    haystack_full = tokenizer.decode(out_ids)
    prompt = (
        "Read the text and answer.\n\n<text>\n" + haystack_full + "\n</text>\n\n"
        "Question: What is THE magic number mentioned in the text? "
        "(Ignore any tracking ids.)\nAnswer with just the number."
    )
    return {"task": "niah_multikey_1", "prompt": prompt, "expected": real,
            "length_tokens": length_tokens}


def _build_niah_multiquery(length_tokens, tokenizer, rng):
    """4 needles, one question asks for one specific needle by label."""
    haystack = load_haystack_corpus(length_tokens, allow_network=True)
    labels = ["alpha", "beta", "gamma", "delta"]
    values = [f"{rng.randint(10**9, 10**10 - 1)}" for _ in labels]
    target_idx = rng.randint(0, 3)
    needles = [tokenizer(f" The {lab} number is {val}. ", add_special_tokens=False)["input_ids"]
               for lab, val in zip(labels, values, strict=False)]
    hay_ids = tokenizer(haystack, add_special_tokens=False)["input_ids"]
    target = max(length_tokens - 400, 64)
    if len(hay_ids) < target:
        hay_ids = (hay_ids * ((target // len(hay_ids)) + 1))[:target]
    hay_ids = hay_ids[:target]
    positions = sorted([int(target * p) for p in (0.15, 0.35, 0.55, 0.85)])
    out_ids: list = []
    last = 0
    for pos, ids in zip(positions, needles, strict=False):
        out_ids.extend(hay_ids[last:pos])
        out_ids.extend(ids)
        last = pos
    out_ids.extend(hay_ids[last:])
    haystack_full = tokenizer.decode(out_ids)
    prompt = (
        "Read the text and answer.\n\n<text>\n" + haystack_full + "\n</text>\n\n"
        f"Question: What is the {labels[target_idx]} number? "
        "Answer with just the number."
    )
    return {"task": "niah_multiquery", "prompt": prompt, "expected": values[target_idx],
            "length_tokens": length_tokens}


def _build_vt(length_tokens, tokenizer, rng):
    """Variable tracking. v0=v1; v1=v2; ...; vN=42; what is v0? Expected 42."""
    chain_len = rng.randint(3, 5)
    chain = [f"v{i}" for i in range(chain_len)]
    final = f"{rng.randint(10000, 99999)}"
    assignments = [f" {chain[i]} = {chain[i+1]}. " for i in range(chain_len - 1)]
    assignments.append(f" {chain[-1]} = {final}. ")
    rng.shuffle(assignments)

    # Token-level: build haystack of length_tokens, then splice assignments at
    # evenly-spaced positions. Matches the pattern of _build_niah_multikey /
    # _build_niah_multiquery so input length stays close to length_tokens.
    haystack = load_haystack_corpus(length_tokens, allow_network=True)
    hay_ids = tokenizer(haystack, add_special_tokens=False)["input_ids"]
    assign_ids = [tokenizer(a, add_special_tokens=False)["input_ids"] for a in assignments]
    assign_budget = sum(len(a) for a in assign_ids)
    target = max(length_tokens - assign_budget - 100, 64)
    if len(hay_ids) < target:
        hay_ids = (hay_ids * ((target // max(len(hay_ids), 1)) + 1))[:target]
    hay_ids = hay_ids[:target]
    n = len(assignments)
    # spread n assignment positions across the haystack
    positions = sorted({int(target * (i + 1) / (n + 1)) for i in range(n)})
    # if dedup collapses positions (very short haystack), pad to n
    while len(positions) < n:
        positions.append(min(target, positions[-1] + 1))
    out_ids: list = []
    last = 0
    for pos, ids in zip(positions[:n], assign_ids, strict=True):
        out_ids.extend(hay_ids[last:pos])
        out_ids.extend(ids)
        last = pos
    out_ids.extend(hay_ids[last:])
    text = tokenizer.decode(out_ids)
    prompt = (
        "Read the text and trace the variable chain.\n\n<text>\n" + text + "\n</text>\n\n"
        f"Question: What is the final numeric value of {chain[0]} after following the chain?\n"
        "Answer with just the number."
    )
    return {"task": "vt", "prompt": prompt, "expected": final,
            "length_tokens": length_tokens}


def _build_qa_1(length_tokens, tokenizer, rng):
    """Single-hop synthetic QA: insert a unique fact, ask about it."""
    subject = f"Subject_{rng.randint(1000, 9999)}"
    fact_val = f"{rng.randint(100, 999)}"
    fact = f" {subject} won the {fact_val}-meter race. "
    haystack = load_haystack_corpus(length_tokens, allow_network=True)
    hay_ids = tokenizer(haystack, add_special_tokens=False)["input_ids"]
    fact_ids = tokenizer(fact, add_special_tokens=False)["input_ids"]
    target = max(length_tokens - len(fact_ids) - 50, 64)
    if len(hay_ids) < target:
        hay_ids = (hay_ids * ((target // len(hay_ids)) + 1))[:target]
    hay_ids = hay_ids[:target]
    pos = int(len(hay_ids) * rng.choice([0.3, 0.6, 0.85]))
    out_ids = hay_ids[:pos] + fact_ids + hay_ids[pos:]
    text = tokenizer.decode(out_ids)
    prompt = (
        "Read the text and answer.\n\n<text>\n" + text + "\n</text>\n\n"
        f"Question: How many meters did {subject} race? Answer with just the number."
    )
    return {"task": "qa_1", "prompt": prompt, "expected": fact_val,
            "length_tokens": length_tokens}


TASK_REGISTRY: dict[str, RulerTask] = {
    "niah_single_1":    RulerTask("niah_single_1",    _build_niah_single,    _grade_niah),
    "niah_multikey_1":  RulerTask("niah_multikey_1",  _build_niah_multikey,  _grade_niah),
    "niah_multiquery":  RulerTask("niah_multiquery",  _build_niah_multiquery, _grade_niah),
    "vt":               RulerTask("vt",               _build_vt,             _grade_niah),
    "qa_1":             RulerTask("qa_1",             _build_qa_1,           _grade_niah),
}


def build_ruler_cell(task_name: str, length_tokens: int, tokenizer,
                     rng: random.Random) -> dict[str, Any]:
    task = TASK_REGISTRY[task_name]
    return task.build_fn(length_tokens, tokenizer, rng)
