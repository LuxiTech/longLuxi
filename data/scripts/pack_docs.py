"""Pack multi-document jsonl into target-ctx training sequences.

Strategy:
- Concatenate docs in input order (no topic clustering in MVP) until adding the next
  doc would exceed target_ctx. Emit one packed seq, start the next.
- Each doc wrapped with <doc id="..." source="..."> ... </doc> separators.
- Track per-seq token positions where each doc starts (`mask_doc_boundaries`) so a
  later trainer can apply cross-doc attention masks if desired.

Output schema:
  {
    "seq_id": str,
    "target_ctx": int,
    "docs": [{"doc_id": ..., "source": ..., "n_tokens": ...}, ...],
    "text": str,
    "n_tokens": int,
    "mask_doc_boundaries": [int, ...]  # token offsets of each doc's <doc ...> start
  }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_SEP_OPEN_TMPL = '<doc id="{doc_id}" source="{source}">\n'
_SEP_CLOSE = '\n</doc>\n'


def _make_tokenizer(mode: str, model_path: str):
    if mode == "whitespace":
        class _Ws:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": text.split()}

            def decode(self, ids):
                return " ".join(ids)
        return _Ws()
    if mode == "qwen":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    raise ValueError(f"unknown tokenizer mode: {mode}")


def _token_len(tok, s: str) -> int:
    return len(tok(s, add_special_tokens=False)["input_ids"])


def pack(input_paths: list[Path], target_ctx: int, tok) -> list[dict]:
    """Return a list of packed-seq records."""
    sep_overhead = _token_len(tok, _SEP_OPEN_TMPL.format(doc_id="x", source="y")) \
                   + _token_len(tok, _SEP_CLOSE)

    docs_iter = []
    for path in input_paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            docs_iter.append(json.loads(line))

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
            "seq_id": f"pack/{target_ctx}/{seq_idx}",
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

    for d in docs_iter:
        body = d["text"]
        body_tokens = d.get("n_tokens") or _token_len(tok, body)
        # If a single doc exceeds target_ctx, truncate it on token-boundary.
        if body_tokens + sep_overhead > target_ctx:
            ids = tok(body, add_special_tokens=False)["input_ids"]
            ids = ids[:max(target_ctx - sep_overhead, 0)]
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
        cur_docs.append({"doc_id": d["doc_id"], "source": d["source"], "n_tokens": body_tokens})
        cur_tokens += delta
    flush()
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-ctx", type=int, required=True)
    p.add_argument("--input-jsonls", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tokenizer-mode", choices=["qwen", "whitespace"], default="qwen")
    p.add_argument("--tokenizer-path", default="/home/user01/Minko/models/Qwen3.5-4B")
    args = p.parse_args()

    tok = _make_tokenizer(args.tokenizer_mode, args.tokenizer_path)
    seqs = pack(args.input_jsonls, args.target_ctx, tok)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for s in seqs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[pack] {len(seqs)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
