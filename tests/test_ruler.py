"""RULER runner unit tests (no GPU)."""
from longluxi.eval.ruler import TASK_REGISTRY, RulerTask, build_ruler_cell


def test_task_registry_has_canonical_tasks():
    for name in ("niah_single_1", "niah_multikey_1", "niah_multiquery", "vt", "qa_1"):
        assert name in TASK_REGISTRY, f"missing task {name}"
        t = TASK_REGISTRY[name]
        assert isinstance(t, RulerTask)
        assert t.name == name
        assert callable(t.build_fn)
        assert callable(t.grade_fn)


def test_build_niah_single_cell_has_expected_field():
    class _Tok:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": text.split()}

        def decode(self, ids):
            return " ".join(ids)

    import random
    cell = build_ruler_cell("niah_single_1", length_tokens=200, tokenizer=_Tok(),
                              rng=random.Random(0))
    assert cell["task"] == "niah_single_1"
    assert "prompt" in cell
    assert "expected" in cell
