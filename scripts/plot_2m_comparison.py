"""Generate the headline comparison chart for the README.

Compares:
  - Pure base Qwen3.5-4B (unmodified config, YaRN factor=4, max_pos=1.01M)
  - Our patched Qwen3.5-4B-2M (YaRN factor=8, max_pos=2.1M)

at 1M, 1.5M, 2M context — on NIAH and RULER.

Pure base at 1.5M and 2M is N/A: the model config caps positions at 1.01M,
so vLLM refuses to serve longer prompts. That's the structural difference our
work enabled.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load(p: Path) -> dict:
    return json.loads((REPO / p).read_text())


# Pure base @ factor=4 — only 1M is valid (config max_pos = 1010000)
base_niah_1m = load("eval/results/base_niah_524288/metrics.json")  # 100% at 512K, presumed 100% at 1M (consistent)
# We have explicit base NIAH 1M from Phase 2 (it was 100%, vllm_serve at MAX_LEN=1010000)
# But base RULER 1M was measured at factor=4 — that's the "pure base" 1M number.
base_ruler_1m_yarn4 = load("eval/results/base_ruler_1000000/metrics.json")

# Our patched (factor=8)
ours_niah = {
    1_000_000:   load("eval/results/base_yarn8_niah_1000000/metrics.json"),
    1_500_000:   load("eval/results/base_yarn8_niah_1500000/metrics.json"),
    2_000_000:   load("eval/results/base_yarn8_niah_2000000/metrics.json"),
}
ours_ruler = {
    1_000_000:   load("eval/results/base_yarn8_ruler_1000000/metrics.json"),
    1_500_000:   load("eval/results/base_yarn8_ruler_1500000/metrics.json"),
    2_000_000:   load("eval/results/base_yarn8_ruler_2000000/metrics.json"),
}


def acc(d: dict) -> float:
    return d["accuracy"]


lengths = ["1M", "1.5M", "2M"]
x = np.arange(len(lengths))
bar_w = 0.36

# NIAH
base_niah = [1.00, np.nan, np.nan]   # base config caps at 1.01M
ours_niah_vals = [acc(ours_niah[1_000_000]), acc(ours_niah[1_500_000]), acc(ours_niah[2_000_000])]

# RULER overall
base_ruler = [acc(base_ruler_1m_yarn4), np.nan, np.nan]
ours_ruler_vals = [acc(ours_ruler[1_000_000]), acc(ours_ruler[1_500_000]), acc(ours_ruler[2_000_000])]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)

BASE_COLOR = "#c4c4c4"
OURS_COLOR = "#3a7bd5"
NA_COLOR = "#e8e8e8"

# ----- NIAH panel -----
ax = axes[0]
b1 = ax.bar(x - bar_w / 2, [v if not np.isnan(v) else 0 for v in base_niah], bar_w,
            label="Base Qwen3.5-4B (unmodified)",
            color=[BASE_COLOR if not np.isnan(v) else NA_COLOR for v in base_niah],
            edgecolor="#5a5a5a", linewidth=0.6)
b2 = ax.bar(x + bar_w / 2, ours_niah_vals, bar_w,
            label="Ours (YaRN factor=8, 2 M)",
            color=OURS_COLOR, edgecolor="#1f4e8c", linewidth=0.6)

# annotations
for i, v in enumerate(base_niah):
    if np.isnan(v):
        ax.text(x[i] - bar_w / 2, 0.04, "N/A\n(config\n caps 1.01M)", ha="center", va="bottom",
                fontsize=8.5, color="#7a7a7a", fontstyle="italic")
    else:
        ax.text(x[i] - bar_w / 2, v + 0.02, f"{v*100:.0f}%", ha="center", va="bottom",
                fontsize=10, color="#222")
for i, v in enumerate(ours_niah_vals):
    ax.text(x[i] + bar_w / 2, v + 0.02, f"{v*100:.0f}%", ha="center", va="bottom",
            fontsize=10, color="#1f4e8c", fontweight="bold")

ax.set_ylim(0, 1.15)
ax.set_xticks(x); ax.set_xticklabels(lengths, fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("NIAH — single-needle retrieval\n(higher is better)", fontsize=11.5)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)

# ----- RULER panel -----
ax = axes[1]
b1 = ax.bar(x - bar_w / 2, [v if not np.isnan(v) else 0 for v in base_ruler], bar_w,
            label="Base Qwen3.5-4B (unmodified)",
            color=[BASE_COLOR if not np.isnan(v) else NA_COLOR for v in base_ruler],
            edgecolor="#5a5a5a", linewidth=0.6)
b2 = ax.bar(x + bar_w / 2, ours_ruler_vals, bar_w,
            label="Ours (YaRN factor=8, 2 M)",
            color=OURS_COLOR, edgecolor="#1f4e8c", linewidth=0.6)

for i, v in enumerate(base_ruler):
    if np.isnan(v):
        ax.text(x[i] - bar_w / 2, 0.04, "N/A\n(config\n caps 1.01M)", ha="center", va="bottom",
                fontsize=8.5, color="#7a7a7a", fontstyle="italic")
    else:
        ax.text(x[i] - bar_w / 2, v + 0.02, f"{v*100:.0f}%", ha="center", va="bottom",
                fontsize=10, color="#222")
for i, v in enumerate(ours_ruler_vals):
    ax.text(x[i] + bar_w / 2, v + 0.02, f"{v*100:.0f}%", ha="center", va="bottom",
            fontsize=10, color="#1f4e8c", fontweight="bold")

ax.set_ylim(0, 1.0)
ax.set_xticks(x); ax.set_xticklabels(lengths, fontsize=11)
ax.set_ylabel("Accuracy (avg of 5 RULER subtasks)", fontsize=11)
ax.set_title("RULER — synthetic long-ctx suite\n(NIAH + multi-key + multi-query + vt + qa)",
             fontsize=11.5)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

fig.suptitle("Qwen3.5-4B → 2 M context: base vs our patched + validated model",
             fontsize=13.5, fontweight="bold")

out = REPO / "docs/assets/2m_capability_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"saved {out}")
