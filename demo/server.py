#!/usr/bin/env python
"""longluxi demo backend — live single-book QA over the real 2M reader.

Hybrid demo:
  - Preset questions -> answered client-side from pre-baked results (instant).
  - Free questions   -> POST /ask streams a REAL answer from the vLLM 2M reader
                        with the whole book in context. The citation ("命中位置"
                        scroll + 原文 + 回目) is derived FROM THE BOOK ITSELF: we
                        locate the answer's key term in the real text, compute its
                        position %, pull the surrounding passage, and infer the
                        chapter from the nearest preceding 第N回 header. Robust
                        even when the model paraphrases.
  - 四库全开 (combo)  -> a minimal but REAL version of the 10M retrieval story:
                        a lightweight router scores the 4 books, picks the one
                        that holds the answer, then reads THAT whole book densely.
                        (229 万 tok > 2M reader, so we retrieve-then-read instead
                        of stuffing all four in — exactly the roadmap shape.)
  - Uploaded docs    -> POST /register stores a user's text; it then answers live
                        like any book (/ask with the returned id).

Book sits in the system message (stable prefix) so vLLM prefix caching makes the
2nd+ question on a book fast. /warmup primes that cache.

Run:  .venv/bin/python -m uvicorn demo.server:app --host 0.0.0.0 --port 8800
Env:  VLLM_URL (default http://localhost:8001), VLLM_MODEL (default model path)
"""
from __future__ import annotations
import json, os, re, pathlib
from typing import Iterator

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = pathlib.Path(__file__).resolve().parent
BOOKS_DIR = ROOT / "books"
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8001").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "/home/user01/Minko/models/Qwen3.5-4B")

MANIFEST = json.loads((BOOKS_DIR / "manifest.json").read_text(encoding="utf-8"))
BOOK_TEXT: dict[str, str] = {
    k: (BOOKS_DIR / f"{k}.txt").read_text(encoding="utf-8") for k in MANIFEST
}
CANON = list(MANIFEST.keys())  # honglou / sanguo / xiyou / shuihu

_CHAP = re.compile(r"^第[零〇一二三四五六七八九十百千0-9]+回[：:　\s].*$", re.M)
CHAPTERS = {
    k: [(m.start(), m.group().strip()) for m in _CHAP.finditer(v)]
    for k, v in BOOK_TEXT.items()
}
_CJK = re.compile(r"[一-鿿]")

# user-uploaded documents registered at runtime
REGISTERED: dict[str, dict] = {}
_TOK = None
def _count_tokens(text: str) -> int:
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(VLLM_MODEL, trust_remote_code=True)
    return len(_TOK(text, add_special_tokens=False)["input_ids"])

SYSTEM_TMPL = (
    "你是「长麓溪」长文阅读引擎。下面〈全文〉之间是《{title}》（{author}）的完整原文，"
    "已一次性整本载入你的上下文，没有任何切片或检索拼接。\n"
    "请严格只依据这部原文回答用户的问题：先用不超过 3 句话给出简洁、准确的答案"
    "（不要用 markdown 星号，不要长篇罗列，点到为止）。\n"
    "然后另起一行，只输出一行，格式严格为：\n"
    "关键词：<答案里最关键的一个词或短语，必须是上面原文中确实出现过的字面词，便于在原文中定位>\n"
    "不要输出多行关键词，不要再补充出处解释。\n"
    "\n〈全文〉\n{book}\n〈全文完〉"
)

app = FastAPI(title="longluxi-demo")
app.mount("/assets", StaticFiles(directory=str(ROOT / "assets")), name="assets")

class AskReq(BaseModel):
    book: str
    question: str

class WarmReq(BaseModel):
    book: str

class RegReq(BaseModel):
    name: str
    text: str


# ----- generic accessors over canon books + uploaded docs -----
def _known(b: str) -> bool:
    return b in BOOK_TEXT or b in REGISTERED

def _text(b: str) -> str:
    return BOOK_TEXT[b] if b in BOOK_TEXT else REGISTERED[b]["text"]

def _meta(b: str) -> dict:
    if b in MANIFEST:
        return MANIFEST[b]
    r = REGISTERED[b]
    return {"title": r["title"], "author": r["author"], "tokens": r["tokens"]}

def _chaps(b: str):
    return CHAPTERS[b] if b in CHAPTERS else REGISTERED[b]["chapters"]


def _vllm_up() -> bool:
    try:
        return httpx.get(f"{VLLM_URL}/v1/models", timeout=3).status_code == 200
    except Exception:
        return False


def _messages(book: str, question: str):
    m = _meta(book)
    system = SYSTEM_TMPL.format(title=m["title"], author=m["author"], book=_text(book))
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def _route(question: str) -> str:
    """Pick the canon book most likely to hold the answer (minimal retrieval)."""
    cjk = "".join(_CJK.findall(question))
    cands = set()
    for L in (4, 3, 2):
        for i in range(len(cjk) - L + 1):
            cands.add(cjk[i:i + L])
    scores = {k: 0.0 for k in CANON}
    for s in cands:
        present = [k for k in CANON if s in BOOK_TEXT[k]]
        if not present or len(present) == len(CANON):
            continue  # absent or ubiquitous -> not discriminative
        w = len(s) / len(present)  # longer & rarer term -> stronger signal
        for k in present:
            scores[k] += w
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "honglou"


def _split(full: str):
    kw = None
    m = re.search(r"关键词[：:]\s*(.+)", full)
    if m:
        kw = m.group(1).splitlines()[0].strip().strip("「」『』\"'《》 　")
    answer = re.split(r"\n*关键词[：:]", full)[0].strip()
    answer = re.sub(r"\*\*", "", answer)
    return answer, kw


def _best_anchor(book: str, answer: str, kw: str | None):
    raw = _text(book)
    if kw and kw in raw:
        return kw
    cjk = "".join(_CJK.findall(answer))
    best = ""
    n = len(cjk)
    for i in range(n):
        for j in range(n, i + 2, -1):
            if j - i <= len(best):
                break
            sub = cjk[i:j]
            if sub in raw:
                best = sub
                break
    return best or None


def _cite(book: str, anchor: str | None):
    if not anchor:
        return None
    raw = _text(book)
    i = raw.find(anchor)
    if i < 0:
        return None
    pos = round(i / max(len(raw), 1) * 100)
    chapter = None
    for off, title in _chaps(book):
        if off <= i:
            chapter = title
        else:
            break
    a, b = max(0, i - 14), min(len(raw), i + len(anchor) + 22)
    pre = raw[a:i].replace("\n", "").lstrip("：:　 ")
    hit = raw[i:i + len(anchor)]
    suf = raw[i + len(anchor):b].replace("\n", "")
    return {"position": pos, "chapter": chapter, "pre": pre, "hit": hit, "suf": suf}


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")

@app.get("/status")
def status():
    return JSONResponse({"ok": True, "vllm": _vllm_up(), "model": VLLM_MODEL, "books": MANIFEST})

@app.post("/register")
def register(req: RegReq):
    text = req.text.replace("\r\n", "\n").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
    text = text[:4_000_000]  # safety cap (~ keeps under the 2M reader)
    try:
        n = _count_tokens(text)
    except Exception:
        n = round(len(text) * 0.9)
    rid = f"user_{len(REGISTERED) + 1}"
    REGISTERED[rid] = {
        "title": req.name or "我的文档", "author": "我的上传", "text": text, "tokens": n,
        "chapters": [(m.start(), m.group().strip()) for m in _CHAP.finditer(text)],
    }
    return {"ok": True, "id": rid, "title": REGISTERED[rid]["title"],
            "tokens": n, "chars": len(text), "fits": n < 1_950_000}

@app.post("/warmup")
def warmup(req: WarmReq):
    book = req.book
    if book == "combo":
        return {"ok": True, "skipped": "combo routes per-question"}
    if not _known(book):
        return JSONResponse({"ok": False, "error": "unknown book"}, status_code=400)
    payload = {"model": VLLM_MODEL, "messages": _messages(book, "请回答：就绪。"),
               "max_tokens": 1, "temperature": 0.0,
               "chat_template_kwargs": {"enable_thinking": False}}
    try:
        httpx.post(f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=600).raise_for_status()
        return {"ok": True, "tokens": _meta(book)["tokens"]}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.post("/ask")
def ask(req: AskReq):
    routed_title = None
    if req.book == "combo":
        target = _route(req.question)
        routed_title = MANIFEST[target]["title"]
    else:
        if not _known(req.book):
            return JSONResponse({"ok": False, "error": "unknown book"}, status_code=400)
        target = req.book

    def gen() -> Iterator[bytes]:
        if routed_title:
            yield f"data: {json.dumps({'route': routed_title}, ensure_ascii=False)}\n\n".encode()
        payload = {"model": VLLM_MODEL, "messages": _messages(target, req.question),
                   "max_tokens": 420, "temperature": 0.0, "stream": True,
                   "chat_template_kwargs": {"enable_thinking": False}}
        full = ""
        try:
            with httpx.stream("POST", f"{VLLM_URL}/v1/chat/completions",
                              json=payload, timeout=600) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                    except Exception:
                        delta = ""
                    if delta:
                        full += delta
                        yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n".encode()
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n".encode()
            return
        answer, kw = _split(full)
        anchor = _best_anchor(target, answer, kw)
        cite = _cite(target, anchor)
        done = {"done": True, "answer": answer, "keyword": kw,
                "tokens": _meta(target)["tokens"], "routed": routed_title}
        if cite:
            done.update(cite)
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n".encode()

    return StreamingResponse(gen(), media_type="text/event-stream")
