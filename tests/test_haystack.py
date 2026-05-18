"""Haystack corpus loader tests. No network or HF download required."""
from longluxi.eval import haystack


def test_fallback_corpus_long_enough_for_small_target():
    text = haystack.load_haystack_corpus(target_tokens=2_000, allow_network=False)
    # Heuristic: at ~1.3 tokens/word, 2000 tokens ≈ 1540 words. Fallback repeats PG snippet to fill.
    assert len(text.split()) >= 1500


def test_fallback_corpus_scales_to_request():
    small = haystack.load_haystack_corpus(target_tokens=1_000, allow_network=False)
    big = haystack.load_haystack_corpus(target_tokens=50_000, allow_network=False)
    assert len(big) > len(small) * 10


def test_pg_snippet_constant_present():
    assert "writer" in haystack.PAUL_GRAHAM_FALLBACK.lower()
