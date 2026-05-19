"""Synthetic reasoning data for CPT — targeted at RULER vt + multi-hop QA gaps.

Three generators, all producing one jsonl record per document with schema
identical to long_docs_*.jsonl so the existing packer works unchanged:

  vt_chain   - Variable assignment chains embedded in narrative haystack.
               (X knows Y. Y is friends with Z. Z owns the secret code 42.
                ... later ... Question: What does X's friend's friend own?)

  entity_kv  - Entity → attribute chains: A is the manager of B. B is in city C.
               C is the capital of D. D's currency is the EUR. → 4-hop chain.

  code_trace - Pseudocode + step-by-step execution traces. The "haystack" is
               filler text between code lines; the chain is the variable state.

Each generator emits documents of configurable target length. We aim for
~5K-15K tokens per doc so packing into 256K seqs yields 15-50 chains/seq —
similar density to RULER vt at 128K.

Output: data/processed/v2/synthetic_<gen>.jsonl with the SAME schema as
prepare_long_docs_v2 so pack_docs.py can ingest it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from longluxi.paths import PROCESSED_DATA_DIR, ensure_dirs  # noqa: E402

# ---------------- haystack helper (lightweight filler text) ----------------

_NAMES = [
    "Mira", "Esme", "Cordelia", "Hugo", "Aldric", "Liora", "Tarek", "Yseult",
    "Quentin", "Beatrice", "Soren", "Idris", "Caleb", "Daphne", "Roderick",
    "Imelda", "Tristan", "Aurelia", "Casimir", "Octavia", "Theodore",
    "Penelope", "Bartholomew", "Cassandra", "Reginald", "Cornelius", "Eleanor",
    "Phineas", "Ophelia", "Maximilian", "Genevieve", "Sebastian",
]
_VERBS = [
    "walked through", "pondered over", "visited", "examined", "acquired",
    "observed", "studied", "considered", "remembered", "discovered",
    "cataloged", "inscribed", "transcribed", "annotated", "revisited",
    "mentioned", "glanced at", "translated", "questioned",
]
_NOUNS = [
    "manuscript", "chronicle", "ledger", "compass", "relic", "codex",
    "parchment", "portrait", "tapestry", "monolith", "almanac", "journal",
    "cipher", "rune", "cartouche", "etching", "amulet", "decree", "tome",
    "obsidian shard",
]
_PLACES = [
    "the silent archive", "the moonlit cloister", "the granite causeway",
    "the cinder gallery", "the violet observatory", "the iron parlor",
    "the salt scriptorium", "the gilded antechamber", "the basalt rotunda",
    "the cedar mezzanine", "the lapis annex", "the alabaster colonnade",
]


def _filler_sentence(rng: random.Random) -> str:
    n = rng.choice(_NAMES)
    v = rng.choice(_VERBS)
    o = rng.choice(_NOUNS)
    where = rng.choice(_PLACES)
    return f"{n} {v} the {o} at {where}."


def _filler_paragraph(rng: random.Random, n_sent: int = 6) -> str:
    return " ".join(_filler_sentence(rng) for _ in range(n_sent))


def _filler_block(rng: random.Random, n_paragraphs: int) -> str:
    return "\n\n".join(_filler_paragraph(rng) for _ in range(n_paragraphs))


# ---------------- vt_chain generator ----------------

def _vt_chain(rng: random.Random, chain_len: int) -> tuple[str, str, str]:
    """Generate (chain_facts: str, question: str, answer: str)."""
    chain = [f"v{i}_{rng.randint(0, 999)}" for i in range(chain_len)]
    final = f"{rng.randint(10000, 99999)}"
    parts = []
    # Chain: v0 → v1 → ... → final
    for i in range(chain_len - 1):
        parts.append(f"The token labeled {chain[i]} maps to {chain[i+1]}.")
    parts.append(f"The token labeled {chain[-1]} maps to the number {final}.")
    rng.shuffle(parts)
    facts_block = " ".join(parts)
    question = (
        f"Trace the mapping chain starting from {chain[0]}. "
        f"Follow each step until you reach a numeric value. "
        f"What is the final numeric value of {chain[0]}?"
    )
    return facts_block, question, final


def _gen_vt_chain_doc(rng: random.Random, target_tokens: int, tokenizer,
                      chain_len: int) -> dict:
    """One doc: long narrative haystack with a vt chain embedded at random
    positions. Question + final answer printed at the end.
    """
    facts, q, ans = _vt_chain(rng, chain_len)
    # Roughly 1.4 tokens per word; haystack at target_tokens means roughly
    # target_tokens / 1.4 words. Build by paragraphs and stop when we exceed.
    paragraphs = []
    cur_words = 0
    target_words = int(target_tokens / 1.4)
    while cur_words < target_words:
        para = _filler_paragraph(rng, n_sent=rng.randint(5, 10))
        paragraphs.append(para)
        cur_words += len(para.split())

    # Insert chain facts at distinct random positions in the paragraph list.
    fact_sentences = facts.split(". ")
    fact_sentences = [s.strip() + "." for s in fact_sentences if s.strip()]
    n_paragraphs = len(paragraphs)
    insert_at = sorted(rng.sample(range(n_paragraphs),
                                  k=min(len(fact_sentences), n_paragraphs)))
    for idx, sent in zip(insert_at, fact_sentences, strict=False):
        paragraphs[idx] = paragraphs[idx] + " " + sent

    body = "\n\n".join(paragraphs)
    full = (
        body
        + f"\n\nQuestion: {q}\n"
        + f"Answer: After following the chain, the final numeric value is {ans}.\n"
    )
    return full, ans, q


# ---------------- entity_kv generator ----------------

def _entity_chain_doc(rng: random.Random, target_tokens: int, tokenizer,
                       chain_len: int) -> tuple[str, str, str]:
    """Entity → attribute chain. 'Alice manages Bob. Bob lives in Trondheim.'"""
    entities = rng.sample(_NAMES, k=chain_len)
    cities = ["Trondheim", "Bergen", "Ravenna", "Lucca", "Toulouse",
              "Skopje", "Bilbao", "Reykjavik", "Riga", "Tallinn",
              "Maastricht", "Aachen", "Helsingør", "Salzburg", "Quebec",
              "Salamanca", "Coimbra", "Antwerp", "Ghent", "Bruges"]
    locations = rng.sample(cities, k=chain_len)
    final = f"{rng.randint(1000, 9999)}"
    chain_facts: list[str] = []
    # entity[0] is the manager of entity[1], entity[1] lives in city[0],
    # city[0]'s landmark is X, etc. Keep it simple: a single dimension chain.
    for i in range(chain_len - 1):
        chain_facts.append(f"{entities[i]} manages {entities[i+1]}.")
    chain_facts.append(f"{entities[-1]} owns building #{final}.")
    rng.shuffle(chain_facts)
    facts_block = " ".join(chain_facts)
    target_words = int(target_tokens / 1.4)
    paragraphs = []
    cur_words = 0
    while cur_words < target_words:
        para = _filler_paragraph(rng, n_sent=rng.randint(5, 10))
        paragraphs.append(para)
        cur_words += len(para.split())
    fact_sentences = [s.strip() for s in facts_block.split(".") if s.strip()]
    fact_sentences = [s + "." for s in fact_sentences]
    n_paragraphs = len(paragraphs)
    insert_at = sorted(rng.sample(range(n_paragraphs),
                                  k=min(len(fact_sentences), n_paragraphs)))
    for idx, sent in zip(insert_at, fact_sentences, strict=False):
        paragraphs[idx] = paragraphs[idx] + " " + sent
    body = "\n\n".join(paragraphs)
    q = (f"Starting from {entities[0]}, follow the management chain to its "
         f"leaf. What is the building number owned by the final person?")
    full = (
        body
        + f"\n\nQuestion: {q}\n"
        + f"Answer: Following the management chain from {entities[0]} → "
        + " → ".join(entities[1:])
        + f", the leaf person owns building #{final}.\n"
    )
    return full, final, q


# ---------------- code_trace generator ----------------

def _code_trace_doc(rng: random.Random, target_tokens: int, tokenizer,
                     chain_len: int) -> tuple[str, str, str]:
    """Pseudocode block interleaved with filler. Trace ends with print(x)."""
    var_names = [f"x{i}" for i in range(chain_len)]
    final = rng.randint(10, 99) * rng.choice([1, 7, 11, 13])
    lines = []
    lines.append(f"{var_names[0]} = {rng.randint(1, 9)}")
    for i in range(1, chain_len):
        op = rng.choice(["+", "*"])
        rhs = rng.randint(2, 9)
        lines.append(f"{var_names[i]} = {var_names[i-1]} {op} {rhs}")
    # compute final value
    val = int(lines[0].split("=")[1].strip())
    for L in lines[1:]:
        rhs = L.split("=")[1].strip()
        a, op, b = rhs.split()
        b = int(b)
        if op == "+":
            val = val + b
        else:
            val = val * b
    final = val
    lines.append(f"print({var_names[-1]})")

    target_words = int(target_tokens / 1.4)
    blocks = []
    cur_words = 0
    while cur_words < target_words:
        para = _filler_paragraph(rng, n_sent=rng.randint(4, 9))
        blocks.append(para)
        cur_words += len(para.split())
    # interleave: every K paragraphs, insert one code line
    interleave_k = max(1, len(blocks) // len(lines))
    out_parts: list[str] = []
    code_idx = 0
    for i, b in enumerate(blocks):
        out_parts.append(b)
        if code_idx < len(lines) and (i + 1) % interleave_k == 0:
            out_parts.append(f"\n```\n{lines[code_idx]}\n```\n")
            code_idx += 1
    # ensure any remaining lines are appended
    while code_idx < len(lines):
        out_parts.append(f"\n```\n{lines[code_idx]}\n```\n")
        code_idx += 1
    body = "\n\n".join(out_parts)
    q = (f"Trace the variable updates above in source order. "
         f"What integer value does the program print at the end?")
    full = (
        body
        + f"\n\nQuestion: {q}\n"
        + f"Answer: Following the code in source order, the printed value is {final}.\n"
    )
    return full, str(final), q


# ---------------- runner ----------------

GENERATORS = {
    "vt_chain":   ("vt_chain",   _gen_vt_chain_doc),
    "entity_kv":  ("entity_kv",  _entity_chain_doc),
    "code_trace": ("code_trace", _code_trace_doc),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", choices=list(GENERATORS), required=True)
    p.add_argument("--n-docs", type=int, default=2000)
    p.add_argument("--target-tokens", type=int, default=10_000,
                   help="approx tokens per generated doc")
    p.add_argument("--chain-len-min", type=int, default=4)
    p.add_argument("--chain-len-max", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", default="/home/user01/Minko/models/Qwen3.5-4B")
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR / "v2")
    args = p.parse_args()

    ensure_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name, fn = GENERATORS[args.gen]
    out_path = args.out_dir / f"synthetic_{name}.jsonl"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rng = random.Random(args.seed)
    print(f"[synthesize] {name}: writing {args.n_docs} docs -> {out_path}", flush=True)

    n_tokens_total = 0
    with out_path.open("w") as f:
        for i in range(args.n_docs):
            chain_len = rng.randint(args.chain_len_min, args.chain_len_max)
            text, ans, q = fn(rng, args.target_tokens, tokenizer, chain_len)
            n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            doc_id = f"{name}/{i}"
            rec = {
                "doc_id": doc_id,
                "source": f"synthetic_{name}",
                "title": f"chain_len={chain_len}",
                "text": text,
                "n_tokens": n_tok,
                "metadata": {"chain_len": chain_len, "answer": ans, "question": q[:200]},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_tokens_total += n_tok
            if (i + 1) % 200 == 0:
                print(f"[synthesize] {name} {i+1}/{args.n_docs} cum_tokens={n_tokens_total/1e6:.2f}M", flush=True)

    print(f"[synthesize] {name} done: {args.n_docs} docs, {n_tokens_total/1e6:.2f}M tokens -> {out_path}")


if __name__ == "__main__":
    main()
