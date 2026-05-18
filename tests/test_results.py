"""Results writer tests — verify metrics.json / predictions.jsonl / summary.md emitted."""
import json
from pathlib import Path

from longluxi.eval.results import write_results


def test_write_results_creates_all_three_files(tmp_path):
    preds = [
        {"length_tokens": 1024, "depth_pct": 50, "expected": "12345", "pred": "12345", "ok": True, "input_tokens": 1000},
        {"length_tokens": 1024, "depth_pct": 50, "expected": "67890", "pred": "wrong", "ok": False, "input_tokens": 1000},
        {"length_tokens": 8192, "depth_pct": 10, "expected": "11111", "pred": "11111", "ok": True, "input_tokens": 8000},
    ]
    out = write_results(tmp_path, model_id="m/x", yarn_config="y.json", predictions=preds)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "predictions.jsonl").exists()
    assert (tmp_path / "summary.md").exists()
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["accuracy"] == 2 / 3
    assert m["by_length"]["1024"] == 0.5
    assert m["by_length"]["8192"] == 1.0
    assert out["accuracy"] == 2 / 3


def test_write_results_empty_predictions(tmp_path):
    out = write_results(tmp_path, model_id="m/x", yarn_config=None, predictions=[])
    assert out["accuracy"] == 0.0
    assert (tmp_path / "metrics.json").exists()
