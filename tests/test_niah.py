"""NIAH cell construction + grading tests. Uses a fake whitespace tokenizer (no HF dep)."""
import random

from longluxi.eval.niah import NiahCell, build_cell, grade


class _WhitespaceTokenizer:
    """Minimal tokenizer for unit tests: split on whitespace, no special tokens."""
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


def test_build_cell_inserts_needle_at_depth():
    tok = _WhitespaceTokenizer()
    hay = " ".join(["word"] * 1000)
    cell = build_cell(hay, length_tokens=500, depth_pct=50, tokenizer=tok,
                      rng=random.Random(0), margin_tokens=50)
    assert isinstance(cell, NiahCell)
    assert cell.expected.isdigit()
    assert "magic number" in cell.haystack.lower()
    # The needle should sit near the midpoint of the haystack region.
    words = cell.haystack.split()
    midword_idx = words.index("magic")
    # depth_pct=50 of (length_tokens - margin) = 50% of 450 = 225-ish; allow ±50.
    assert 175 <= midword_idx <= 275


def test_grade_substring_match():
    cell = NiahCell(length_tokens=1, depth_pct=50, needle="The magic number is 12345.",
                    expected="12345", haystack="x", prompt="x")
    assert grade(cell, "12345") is True
    assert grade(cell, "The number is 12345 indeed.") is True
    assert grade(cell, "98765") is False
    assert grade(cell, "") is False


def test_build_cell_deterministic_with_seed():
    tok = _WhitespaceTokenizer()
    hay = " ".join(["word"] * 1000)
    c1 = build_cell(hay, 500, 50, tok, random.Random(42), margin_tokens=50)
    c2 = build_cell(hay, 500, 50, tok, random.Random(42), margin_tokens=50)
    assert c1.expected == c2.expected
