"""Runner-level unit tests using the fake whitespace tokenizer from test_niah."""
from longluxi.eval.runner import build_grid


class _WhitespaceTokenizer:
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


def test_build_grid_max_cells_zero_returns_empty():
    """Regression: max_cells=0 must mean 'cap at zero', not 'no cap'."""
    grid = build_grid(lengths=[500], depths=[50], n_per_cell=2,
                      tokenizer=_WhitespaceTokenizer(), seed=0, max_cells=0)
    assert grid == []


def test_build_grid_max_cells_none_returns_all():
    grid = build_grid(lengths=[500], depths=[10, 50, 90], n_per_cell=2,
                      tokenizer=_WhitespaceTokenizer(), seed=0, max_cells=None)
    assert len(grid) == 6  # 1 length * 3 depths * 2 per cell


def test_build_grid_max_cells_caps_at_value():
    grid = build_grid(lengths=[500], depths=[10, 50, 90], n_per_cell=4,
                      tokenizer=_WhitespaceTokenizer(), seed=0, max_cells=5)
    assert len(grid) == 5
