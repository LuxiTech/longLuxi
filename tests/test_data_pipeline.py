"""Data pipeline tests — no GPU, no network. Uses a tiny synthetic input."""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_sample_jsonl(path: Path, n_docs: int, n_tokens_per_doc: int):
    with path.open("w") as f:
        for i in range(n_docs):
            text = "word " * n_tokens_per_doc  # 1 token per word with whitespace tokenizer
            f.write(json.dumps({
                "doc_id": f"sample/{i}",
                "source": "sample",
                "title": f"doc {i}",
                "text": text.strip(),
                "n_tokens": n_tokens_per_doc,
                "metadata": {},
            }) + "\n")


def test_pack_docs_writes_target_ctx_sequences(tmp_path):
    src = tmp_path / "in.jsonl"
    _write_sample_jsonl(src, n_docs=20, n_tokens_per_doc=1000)
    out = tmp_path / "packed.jsonl"
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "data/scripts/pack_docs.py"),
         "--target-ctx", "5000",
         "--input-jsonls", str(src),
         "--out", str(out),
         "--tokenizer-mode", "whitespace"],  # bypass HF tokenizer for tests
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, res.stderr
    assert out.exists()
    lines = out.read_text().splitlines()
    assert len(lines) >= 3  # 20 docs * 1000 tokens / 5000 = 4 sequences-ish
    for line in lines:
        rec = json.loads(line)
        assert "seq_id" in rec
        assert "target_ctx" in rec and rec["target_ctx"] == 5000
        assert "docs" in rec
        assert rec["n_tokens"] <= 5000 + 100  # allow some slack for separators
        assert "<doc id=" in rec["text"]


def test_pack_docs_respects_doc_boundary_mask(tmp_path):
    src = tmp_path / "in.jsonl"
    _write_sample_jsonl(src, n_docs=4, n_tokens_per_doc=600)
    out = tmp_path / "packed.jsonl"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "data/scripts/pack_docs.py"),
         "--target-ctx", "5000",
         "--input-jsonls", str(src),
         "--out", str(out),
         "--tokenizer-mode", "whitespace"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    rec = json.loads(out.read_text().splitlines()[0])
    # mask_doc_boundaries indicates where each new doc starts (in tokens)
    assert "mask_doc_boundaries" in rec
    assert len(rec["mask_doc_boundaries"]) == len(rec["docs"])
