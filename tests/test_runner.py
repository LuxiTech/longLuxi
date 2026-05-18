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


def test_format_for_model_with_chat_template():
    """If tokenizer has apply_chat_template + chat_template, use it with enable_thinking=False."""
    from longluxi.eval.runner import _format_for_model

    calls: list[dict] = []

    class _FakeChatTok:
        chat_template = "fake"
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                                 enable_thinking=None):
            calls.append({"messages": messages, "tokenize": tokenize,
                          "add_generation_prompt": add_generation_prompt,
                          "enable_thinking": enable_thinking})
            return f"<chat>{messages[0]['content']}</chat>"

    tok = _FakeChatTok()
    out = _format_for_model("hello", tok)
    assert out == "<chat>hello</chat>"
    assert calls[0]["enable_thinking"] is False
    assert calls[0]["add_generation_prompt"] is True


def test_format_for_model_falls_back_when_no_chat_template():
    """Tokenizers without apply_chat_template or with empty chat_template should pass-through."""
    from longluxi.eval.runner import _format_for_model

    class _NoTemplate:
        chat_template = None  # explicit no template
        def apply_chat_template(self, *a, **kw):
            raise AssertionError("should not be called when chat_template is None")

    assert _format_for_model("hello", _NoTemplate()) == "hello"

    class _NoMethod:
        pass

    assert _format_for_model("hello", _NoMethod()) == "hello"


def test_format_for_model_falls_back_when_enable_thinking_unsupported():
    """Old tokenizers without enable_thinking kwarg should still work."""
    from longluxi.eval.runner import _format_for_model

    class _OldChatTok:
        chat_template = "fake"
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return f"<old>{messages[0]['content']}</old>"

    assert _format_for_model("hi", _OldChatTok()) == "<old>hi</old>"
