"""Phase 1: ensure src/longluxi/eval/ is an importable package."""
import importlib


def test_eval_package_importable():
    mod = importlib.import_module("longluxi.eval")
    assert mod is not None
