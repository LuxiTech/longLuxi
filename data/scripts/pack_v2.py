"""Pack mixed long-doc + synthetic reasoning into target-ctx training sequences.

Differs from pack_docs.py (v1) in three ways:
1. Shuffles docs ACROSS sources before packing (so each packed seq sees a
   mix of arxiv/pg19/synthetic instead of all-from-one-source seqs).
2. Reports per-source token contribution after packing.
3. Skips docs whose token count is so close to target_ctx that the doc would
   completely fill a sequence — those should be split externally, not packed.

Output schema same as v1 pack_docs.py so existing LF training YAMLs work
unchanged (just point dataset_dir at the new file).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


_SEP_OPEN_TMPL = '<doc id="{doc_id}" source="{source}">\n'
_SEP_CLOSE = '\n</doc>\n'


def _make_tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _token_len(tok, s: str) -> int:
    return len(tok(s, add_special_tokens=False)["input_ids"])


def pack(docs: list[dict], target_ctx: int, tok) -> list[dict]:
    sep_overhead = _token_len(tok, _SEP_OPEN_TMPL.format(doc_id="x", source="y")) \
                   + _token_len(tok, _SEP_CLOSE)
    seqs: list[dict] = []
    cur_text_parts: list[str] = []
    cur_docs: list[dict] = []
    cur_boundaries: list[int] = []
    cur_tokens = 0
    seq_idx = 0

    def flush():
        nonlocal cur_text_parts, cur_docs, cur_boundaries, cur_tokens, seq_idx
        if not cur_docs:
            return
        seqs.append({
            "seq_id": f"v2pack/{target_ctx}/{seq_idx}",
            "target_ctx": target_ctx,
            "docs": cur_docs,
            "text": "".join(cur_text_parts),
            "n_tokens": cur_tokens,
            "mask_doc_boundaries": cur_boundaries,
        })
        seq_idx += 1
        cur_text_parts = []
        cur_docs = []
        cur_boundaries = []
        cur_tokens = 0

    for d in docs:
        body = d["text"]
        body_tokens = d.get("n_tokens") or _token_len(tok, body)
        if body_tokens + sep_overhead > target_ctx:
            # truncate single oversized doc
            ids = tok(body, add_special_tokens=False)["input_ids"]
            ids = ids[: max(target_ctx - sep_overhead, 0)]
            body = tok.decode(ids)
            body_tokens = len(ids)
        delta = body_tokens + sep_overhead
        if cur_tokens + delta > target_ctx and cur_docs:
            flush()
        open_tag = _SEP_OPEN_TMPL.format(doc_id=d["doc_id"], source=d["source"])
        cur_boundaries.append(cur_tokens)
        cur_text_parts.append(open_tag)
        cur_text_parts.append(body)
        cur_text_parts.append(_SEP_CLOSE)
        cur_docs.append({"doc_id": d["doc_id"], "source": d["source"],
                          "n_tokens": body_tokens})
        cur_tokens += delta
    flush()
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-ctx", type=int, required=True)
    p.add_argument("--input-jsonls", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tokenizer-path", default="/home/user01/Minko/models/Qwen3.5-4B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=0,
                   help="Stop adding docs after this many input tokens (0 = unlimited).")
    args = p.parse_args()

    tok = _make_tokenizer(args.tokenizer_path)
    rng = random.Random(args.seed)

    docs: list[dict] = []
    by_source: dict[str, int] = collections.Counter()
    by_source_tokens: dict[str, int] = collections.Counter()
    for path in args.input_jsonls:
        n = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            docs.append(d)
            by_source[d["source"]] += 1
            by_source_tokens[d["source"]] += d.get("n_tokens", 0)
            n += 1
        print(f"[pack_v2] loaded {n} docs from {path}", flush=True)

    rng.shuffle(docs)
    if args.max_tokens > 0:
        cum = 0
        cut = len(docs)
        for i, d in enumerate(docs):
            cum += d.get("n_tokens", 0)
            if cum >= args.max_tokens:
                cut = i + 1
                break
        docs = docs[:cut]
        print(f"[pack_v2] capped at {cum/1e6:.2f}M tokens ({cut} docs)", flush=True)

    print("[pack_v2] per-source breakdown (after shuffle, before packing):")
    for s, n in by_source.most_common():
        print(f"  {s:30s} {n:5d} docs   {by_source_tokens[s]/1e6:7.2f}M tokens")

    seqs = pack(docs, args.target_ctx, tok)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for s in seqs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    total = sum(s["n_tokens"] for s in seqs)
    avg_ctx = total / max(len(seqs), 1)
    print(f"[pack_v2] {len(seqs)} sequences -> {args.out}")
    print(f"[pack_v2] total {total/1e6:.2f}M tokens, avg {avg_ctx/1000:.1f}K tokens/seq")


if __name__ == "__main__":
    main()
