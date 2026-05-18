"""Generic grid runner for NIAH-style evals."""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from longluxi.eval.haystack import load_haystack_corpus
from longluxi.eval.niah import NiahCell, build_cell, grade
from longluxi.eval.results import write_results


def build_grid(lengths: list[int], depths: list[int], n_per_cell: int,
               tokenizer, seed: int = 42, max_cells: int | None = None) -> list[NiahCell]:
    rng = random.Random(seed)
    hay = load_haystack_corpus(max(lengths), allow_network=True)
    cells: list[NiahCell] = []
    for L in lengths:
        for d in depths:
            for _ in range(n_per_cell):
                cells.append(build_cell(hay, L, d, tokenizer, rng))
            if max_cells and len(cells) >= max_cells:
                return cells[:max_cells]
    return cells


def evaluate_cells(cells: list[NiahCell], model, tokenizer, max_new_tokens: int = 16,
                   verbose: bool = True) -> list[dict[str, Any]]:
    """Iterate cells, run greedy decode, return prediction records."""
    import torch  # heavy

    predictions: list[dict[str, Any]] = []
    for i, cell in enumerate(cells):
        inputs = tokenizer(cell.prompt, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        pred = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        ok = grade(cell, pred)
        predictions.append({
            "length_tokens": cell.length_tokens,
            "depth_pct": cell.depth_pct,
            "expected": cell.expected,
            "pred": pred,
            "ok": ok,
            "input_tokens": n_in,
        })
        if verbose:
            print(f"[eval] cell {i+1}/{len(cells)} L={cell.length_tokens} d={cell.depth_pct}% "
                  f"expected={cell.expected} pred={pred!r} ok={ok}", flush=True)
    return predictions


def run_niah(model_id: str, lengths: list[int], depths: list[int], n_per_cell: int,
             out_dir: Path, yarn_path: str | None = None, seed: int = 42,
             max_cells: int | None = None) -> dict[str, Any]:
    from longluxi.eval.model_loader import load_model_and_tokenizer

    model, tokenizer, _cfg = load_model_and_tokenizer(model_id, yarn_path)
    cells = build_grid(lengths, depths, n_per_cell, tokenizer, seed=seed, max_cells=max_cells)
    preds = evaluate_cells(cells, model, tokenizer)
    return write_results(out_dir, model_id=model_id, yarn_config=str(yarn_path) if yarn_path else None,
                         predictions=preds)
