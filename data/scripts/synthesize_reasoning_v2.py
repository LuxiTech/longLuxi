"""Synthetic reasoning data v2 — fixes Stage A failures.

Stage A v1 problems diagnosed in STAGE_A_RETROSPECTIVE.md:
  P1: model memorized "answer = X" right after "Question:" → never learned chain.
  P2: single-answer per doc → niah_multiquery (4 needles) confused.
  P3: chain only in one direction → trivial copy from doc end works.

V2 fixes:
  F1: each doc has N=3-6 INDEPENDENT chains; N questions; model can't pick "the"
      one answer. Different chain IDs in each so model must use chain-ID to
      pick which trace to do.
  F2: each Q&A includes an explicit <trace>step1; step2; ...</trace> *before*
      the final answer. Trains the chain-walking behavior, not the
      copy-pattern.
  F3: chain links scattered in haystack at random positions, but ALSO some
      links given in reversed order, forcing the model to handle both
      directions (Stage A v1 always had links in shuffled-but-readable order).

  vt_chain: token X maps to token Y; final token maps to integer.
  entity_kv: entity manages entity; final manages building #N.
  code_trace: assignments to vars; final print.

All generators write {doc_id, source, title, text, n_tokens, metadata} —
same schema as v1 / long_docs_*.jsonl so pack_v2.py just works.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from longluxi.paths import PROCESSED_DATA_DIR, ensure_dirs  # noqa: E402

_NAMES = [
    "Mira", "Esme", "Cordelia", "Hugo", "Aldric", "Liora", "Tarek", "Yseult",
    "Quentin", "Beatrice", "Soren", "Idris", "Caleb", "Daphne", "Roderick",
    "Imelda", "Tristan", "Aurelia", "Casimir", "Octavia", "Theodore",
    "Penelope", "Bartholomew", "Cassandra", "Reginald", "Cornelius", "Eleanor",
    "Phineas", "Ophelia", "Maximilian", "Genevieve", "Sebastian", "Anya",
    "Bram", "Cleo", "Dorin", "Elysia", "Falke", "Greta", "Hanno", "Ines",
    "Jaro", "Kerris", "Lazlo", "Margit", "Nils", "Orin", "Petra",
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


def _filler_sentence(rng):
    return f"{rng.choice(_NAMES)} {rng.choice(_VERBS)} the {rng.choice(_NOUNS)} at {rng.choice(_PLACES)}."


def _filler_para(rng, n=6):
    return " ".join(_filler_sentence(rng) for _ in range(n))


# ---------------- vt_chain v2 ----------------

def _vt_chain_doc_v2(rng, target_tokens, n_chains=4, chain_len_range=(4, 9)):
    """Multi-chain vt-style doc with explicit traces."""
    # Build N independent chains. Each chain has a unique ID prefix (alpha/beta/...).
    chain_ids = rng.sample(
        ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"],
        k=n_chains,
    )
    chains = []
    for cid in chain_ids:
        L = rng.randint(*chain_len_range)
        tokens = [f"{cid}_{rng.randint(100, 999)}_{i}" for i in range(L)]
        final = rng.randint(1000, 99999)
        chains.append({"cid": cid, "tokens": tokens, "final": final, "len": L})

    # Generate haystack paragraphs sized to target_tokens (approx words ≈ tokens/1.4)
    target_words = int(target_tokens / 1.4)
    paragraphs = []
    cur_words = 0
    while cur_words < target_words:
        p = _filler_para(rng, n=rng.randint(5, 10))
        paragraphs.append(p)
        cur_words += len(p.split())

    # Collect all chain link facts (each chain contributes L sentences).
    # F3: half the chains have links in forward order, half in random shuffled order.
    all_facts = []
    for c in chains:
        facts = []
        for i in range(c["len"] - 1):
            facts.append(f"The token {c['tokens'][i]} maps to {c['tokens'][i+1]}.")
        facts.append(f"The token {c['tokens'][-1]} maps to the number {c['final']}.")
        if rng.random() < 0.5:
            rng.shuffle(facts)
        all_facts.extend([(c["cid"], f) for f in facts])

    # Scatter facts at random positions
    rng.shuffle(all_facts)
    n_paragraphs = len(paragraphs)
    if n_paragraphs >= len(all_facts):
        slots = sorted(rng.sample(range(n_paragraphs), k=len(all_facts)))
    else:
        slots = [rng.randint(0, n_paragraphs - 1) for _ in all_facts]
    for idx, (cid, fact) in zip(slots, all_facts, strict=False):
        paragraphs[idx] = paragraphs[idx] + " " + fact

    body = "\n\n".join(paragraphs)

    # Build N questions, each with a <trace> showing the steps.
    qa_blocks = []
    for c in chains:
        cid = c["cid"]
        trace_steps = []
        for i in range(c["len"] - 1):
            trace_steps.append(f"Step {i+1}: {c['tokens'][i]} → {c['tokens'][i+1]}.")
        trace_steps.append(f"Step {c['len']}: {c['tokens'][-1]} → {c['final']}.")
        trace = "\n".join(trace_steps)
        qa = (
            f"\n\nQuestion: Starting from {c['tokens'][0]}, follow the {cid} mapping "
            f"chain. What numeric value does it end at?\n"
            f"<trace>\n{trace}\n</trace>\n"
            f"Answer: {c['final']}\n"
        )
        qa_blocks.append(qa)

    full = body + "".join(qa_blocks)
    # Use the last chain as the "primary" metadata answer (analytics)
    return full, str(chains[-1]["final"]), chains[-1]["tokens"][0], len(chains)


# ---------------- entity_kv v2 ----------------

_CITIES = [
    "Trondheim", "Bergen", "Ravenna", "Lucca", "Toulouse",
    "Skopje", "Bilbao", "Reykjavik", "Riga", "Tallinn",
    "Maastricht", "Aachen", "Helsingør", "Salzburg", "Quebec",
    "Salamanca", "Coimbra", "Antwerp", "Ghent", "Bruges",
    "Padua", "Verona", "Mantua", "Gdansk", "Krakow",
]


def _entity_chain_doc_v2(rng, target_tokens, n_chains=3, chain_len_range=(4, 8)):
    chains = []
    used_names = set()
    for ci in range(n_chains):
        L = rng.randint(*chain_len_range)
        # Sample disjoint names per chain
        avail = [n for n in _NAMES if n not in used_names]
        ents = rng.sample(avail, k=min(L, len(avail)))
        used_names.update(ents)
        building = rng.randint(1000, 9999)
        chains.append({"ents": ents, "building": building})

    target_words = int(target_tokens / 1.4)
    paragraphs = []
    cur_words = 0
    while cur_words < target_words:
        p = _filler_para(rng, n=rng.randint(5, 10))
        paragraphs.append(p)
        cur_words += len(p.split())

    all_facts = []
    for c in chains:
        facts = []
        for i in range(len(c["ents"]) - 1):
            facts.append(f"{c['ents'][i]} manages {c['ents'][i+1]}.")
        facts.append(f"{c['ents'][-1]} owns building #{c['building']}.")
        if rng.random() < 0.5:
            rng.shuffle(facts)
        all_facts.extend(facts)
    rng.shuffle(all_facts)

    n_paragraphs = len(paragraphs)
    if n_paragraphs >= len(all_facts):
        slots = sorted(rng.sample(range(n_paragraphs), k=len(all_facts)))
    else:
        slots = [rng.randint(0, n_paragraphs - 1) for _ in all_facts]
    for idx, fact in zip(slots, all_facts, strict=False):
        paragraphs[idx] = paragraphs[idx] + " " + fact

    body = "\n\n".join(paragraphs)

    qa_blocks = []
    for c in chains:
        trace_lines = []
        for i in range(len(c["ents"]) - 1):
            trace_lines.append(f"  {c['ents'][i]} → {c['ents'][i+1]} (manages)")
        trace_lines.append(f"  {c['ents'][-1]} → building #{c['building']}")
        trace = "\n".join(trace_lines)
        qa = (
            f"\n\nQuestion: Starting from {c['ents'][0]}, follow the management "
            f"chain to its leaf. What building number is owned by the final person?\n"
            f"<trace>\n{trace}\n</trace>\n"
            f"Answer: {c['building']}\n"
        )
        qa_blocks.append(qa)

    full = body + "".join(qa_blocks)
    return full, str(chains[-1]["building"]), chains[-1]["ents"][0], n_chains


# ---------------- code_trace v2 ----------------

def _code_trace_doc_v2(rng, target_tokens, n_programs=3, chain_len_range=(4, 8)):
    """Embedded short programs with traced execution."""
    programs = []
    for pi in range(n_programs):
        L = rng.randint(*chain_len_range)
        vnames = [f"x{pi}_{i}" for i in range(L)]
        v0 = rng.randint(1, 9)
        lines = [f"{vnames[0]} = {v0}"]
        val = v0
        for i in range(1, L):
            op = rng.choice(["+", "*"])
            rhs = rng.randint(2, 9)
            lines.append(f"{vnames[i]} = {vnames[i-1]} {op} {rhs}")
            val = (val + rhs) if op == "+" else (val * rhs)
        lines.append(f"print({vnames[-1]})")
        programs.append({"vnames": vnames, "lines": lines, "result": val})

    target_words = int(target_tokens / 1.4)
    paragraphs = []
    cur_words = 0
    while cur_words < target_words:
        p = _filler_para(rng, n=rng.randint(4, 9))
        paragraphs.append(p)
        cur_words += len(p.split())

    # Scatter program lines (each program's lines interleaved separately)
    all_lines = []
    for pi, prog in enumerate(programs):
        for L in prog["lines"]:
            all_lines.append((pi, L))
    rng.shuffle(all_lines)

    n_paragraphs = len(paragraphs)
    slots = sorted([rng.randint(0, n_paragraphs - 1) for _ in all_lines])
    for idx, (pi, L) in zip(slots, all_lines, strict=False):
        paragraphs[idx] = paragraphs[idx] + f"\n\n```\n{L}\n```\n"

    body = "\n\n".join(paragraphs)

    qa_blocks = []
    for pi, prog in enumerate(programs):
        # Build trace: re-execute step by step
        v0 = int(prog["lines"][0].split("=")[1].strip())
        trace_lines = [f"{prog['vnames'][0]} = {v0}"]
        val = v0
        for i in range(1, len(prog["vnames"])):
            rhs = prog["lines"][i].split("=")[1].strip()
            a, op, b = rhs.split()
            b = int(b)
            val = val + b if op == "+" else val * b
            trace_lines.append(f"{prog['vnames'][i]} = {val}")
        trace = "\n".join(trace_lines)
        qa = (
            f"\n\nQuestion: Trace the program lines whose variables start with "
            f"`x{pi}_`. What integer does the program print?\n"
            f"<trace>\n{trace}\n</trace>\n"
            f"Answer: {prog['result']}\n"
        )
        qa_blocks.append(qa)

    full = body + "".join(qa_blocks)
    return full, str(programs[-1]["result"]), f"program_{n_programs-1}", n_programs


GENERATORS = {
    "vt_chain_v2":   _vt_chain_doc_v2,
    "entity_kv_v2":  _entity_chain_doc_v2,
    "code_trace_v2": _code_trace_doc_v2,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", choices=list(GENERATORS), required=True)
    p.add_argument("--n-docs", type=int, default=2000)
    p.add_argument("--target-tokens", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--tokenizer", default="/home/user01/Minko/models/Qwen3.5-4B")
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR / "v2")
    args = p.parse_args()

    ensure_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fn = GENERATORS[args.gen]
    out_path = args.out_dir / f"synthetic_{args.gen}.jsonl"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rng = random.Random(args.seed)
    print(f"[synthesize_v2] {args.gen}: {args.n_docs} docs target {args.target_tokens}t -> {out_path}", flush=True)

    n_tok_total = 0
    with out_path.open("w") as f:
        for i in range(args.n_docs):
            text, ans, primary_input, n_chains = fn(rng, args.target_tokens)
            n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            rec = {
                "doc_id": f"{args.gen}/{i}",
                "source": f"synthetic_{args.gen}",
                "title": f"n_chains={n_chains}",
                "text": text,
                "n_tokens": n_tok,
                "metadata": {"primary_answer": ans, "primary_input": primary_input,
                             "n_chains": n_chains},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_tok_total += n_tok
            if (i + 1) % 200 == 0:
                print(f"[synthesize_v2] {args.gen} {i+1}/{args.n_docs} cum_tokens={n_tok_total/1e6:.2f}M", flush=True)
    print(f"[synthesize_v2] {args.gen} done: {args.n_docs} docs, {n_tok_total/1e6:.2f}M tokens -> {out_path}")


if __name__ == "__main__":
    main()
