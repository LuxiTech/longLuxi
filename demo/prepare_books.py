#!/usr/bin/env python
"""Clean the four Gutenberg classics for the longluxi live demo.

- slice out Project Gutenberg license boilerplate (keep text between the
  START / END markers)
- traditional -> simplified (opencc t2s)
- normalise whitespace (collapse 3+ blank lines)
- report real token counts with the model's own tokenizer so we know each
  book fits the 2M reader and can show honest numbers in the UI

Output: demo/books/<name>.txt  +  demo/books/manifest.json
"""
from __future__ import annotations
import json, re, pathlib

import opencc
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "books_raw"
OUT = ROOT / "books"
OUT.mkdir(exist_ok=True)
MODEL = "/home/user01/Minko/models/Qwen3.5-4B"

BOOKS = {
    "honglou": {"file": "honglou_24264.txt", "title": "红楼梦", "author": "曹雪芹"},
    "sanguo":  {"file": "sanguo_23950.txt",  "title": "三国演义", "author": "罗贯中"},
    "xiyou":   {"file": "xiyou_23962.txt",    "title": "西游记",   "author": "吴承恩"},
    "shuihu":  {"file": "shuihu_23863.txt",   "title": "水浒传",   "author": "施耐庵"},
}

START = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG.*?\*\*\*", re.S)
END = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG.*?\*\*\*", re.S)

def strip_boilerplate(text: str) -> str:
    m1 = START.search(text)
    if m1:
        text = text[m1.end():]
    m2 = END.search(text)
    if m2:
        text = text[: m2.start()]
    return text.strip()

def normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def main():
    print("loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    cc = opencc.OpenCC("t2s")
    manifest = {}
    for key, meta in BOOKS.items():
        raw = (RAW / meta["file"]).read_text(encoding="utf-8", errors="ignore")
        body = normalise(cc.convert(strip_boilerplate(raw)))
        (OUT / f"{key}.txt").write_text(body, encoding="utf-8")
        n_char = len(body)
        n_tok = len(tok(body, add_special_tokens=False)["input_ids"])
        manifest[key] = {
            "title": meta["title"], "author": meta["author"],
            "chars": n_char, "tokens": n_tok,
        }
        print(f"{key:8s} {meta['title']:6s} chars={n_char:>8,} tokens={n_tok:>9,}")
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(m["tokens"] for m in manifest.values())
    print(f"\nfour-book total tokens = {total:,}  (combo '四库全开')")
    print(f"manifest -> {OUT/'manifest.json'}")

if __name__ == "__main__":
    main()
