"""Write metrics.json + predictions.jsonl + summary.md for an eval run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_results(out_dir: Path, *, model_id: str, yarn_config: str | None,
                  predictions: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(predictions)
    correct = sum(1 for p in predictions if p.get("ok"))
    acc = correct / n if n else 0.0

    by_len: dict[int, list[bool]] = {}
    for p in predictions:
        by_len.setdefault(int(p["length_tokens"]), []).append(bool(p["ok"]))

    by_len_acc = {str(L): (sum(oks) / len(oks)) for L, oks in by_len.items()}

    metrics = {
        "model_id": model_id,
        "yarn_config": yarn_config,
        "n_cells": n,
        "accuracy": acc,
        "by_length": by_len_acc,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with (out_dir / "predictions.jsonl").open("w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    lines = [f"# NIAH Results — {model_id}", ""]
    lines.append(f"- Accuracy: **{acc:.2%}** ({correct}/{n})")
    lines.append(f"- YaRN: `{yarn_config}`" if yarn_config else "- YaRN: none")
    lines.append("")
    if by_len:
        lines.append("## By length")
        for L in sorted(by_len.keys()):
            oks = by_len[L]
            lines.append(f"- {L:>10} tokens — {sum(oks)/len(oks):.2%} ({sum(oks)}/{len(oks)})")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    return metrics
