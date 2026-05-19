"""Verify LF dataset registration: longluxi_packed_256k resolves and parses."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LF_DATA = REPO_ROOT / "external" / "LLaMA-Factory" / "data"


def test_longluxi_packed_256k_registered():
    info = json.loads((LF_DATA / "dataset_info.json").read_text())
    assert "longluxi_packed_256k" in info, "dataset not registered in dataset_info.json"
    entry = info["longluxi_packed_256k"]
    assert entry["file_name"] == "longluxi_packed_256k.jsonl"
    assert entry["columns"]["prompt"] == "text"


def test_longluxi_packed_256k_file_resolves():
    f = LF_DATA / "longluxi_packed_256k.jsonl"
    assert f.exists(), f"data file missing: {f}"
    # Should resolve to our packed jsonl with 30 sequences.
    n = sum(1 for _ in f.open())
    assert n == 30, f"expected 30 packed sequences, got {n}"
    rec = json.loads(f.open().readline())
    assert "text" in rec
    assert len(rec["text"]) > 100_000, "first sequence text too short — wrong file?"
