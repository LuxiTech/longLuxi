"""Haystack corpus loaders for NIAH/RULER.

Production path: stream PG-19 from HF datasets.
Offline fallback: repeat a hard-coded Paul Graham snippet to fill the target.
"""
from __future__ import annotations

PAUL_GRAHAM_FALLBACK = """\
The most surprising thing I've learned from being a writer is how rarely the way I think
something is going to come out matches the way it actually does. Whenever you write
something, you imagine how it'll read. But the imagined version and the actual version
diverge in unpredictable ways. The act of writing forces you to think more rigorously than
you would just imagining; you can't paper over weaknesses with fluent prose when each
sentence sits on the page and has to defend itself.
"""

_APPROX_TOKEN_PER_WORD = 1.3


def load_haystack_corpus(target_tokens: int, allow_network: bool = True) -> str:
    """Return a string with at least ~target_tokens worth of English prose."""
    if allow_network:
        text = _try_pg19(target_tokens)
        if text is not None:
            return text
    return _fallback_fill(target_tokens)


def _try_pg19(target_tokens: int) -> str | None:
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
    except Exception:  # noqa: BLE001
        return None
    chunks: list[str] = []
    words = 0
    for ex in ds:
        txt = ex.get("text", "") or ""
        if not txt:
            continue
        chunks.append(txt)
        words += len(txt.split())
        if words * _APPROX_TOKEN_PER_WORD > target_tokens * 1.2:
            break
    return "\n\n".join(chunks) if chunks else None


def _fallback_fill(target_tokens: int) -> str:
    target_words = max(1, int(target_tokens / _APPROX_TOKEN_PER_WORD))
    snippet = PAUL_GRAHAM_FALLBACK
    snippet_words = len(snippet.split())
    n_repeat = max(1, (target_words // snippet_words) + 2)
    return (snippet + "\n") * n_repeat
