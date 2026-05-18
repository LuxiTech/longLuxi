# Eval

两层评测，见 spec §7。

| Runner | 用途 | 周次 |
|---|---|---|
| `runners/run_niah.py` | NIAH / passkey 位置检索 | W1 起每 stage 跑 |
| `runners/run_ruler.py` | RULER 完整 benchmark | W2 起每 stage 跑 |
| `runners/run_longbench_v2.py` | LongBench v2 | W5+ 每 stage |
| `runners/run_infinitebench.py` | InfiniteBench subset | W5+ 每 stage |
| `runners/run_nocha.py` | NoCha book-level | W10 |
| `runners/run_lv_eval.py` | LV-Eval 中文 sanity | W10 |
| `runners/run_10m_e2e.py` | 自建 10M 端到端 benchmark | W11 |

## 启动

```bash
# 单 benchmark
uv run python eval/runners/run_niah.py --model checkpoints/stage_a_1m/latest --lengths 128k 1m

# 全套 (按 matrix)
uv run python eval/runners/run_all.py --config configs/eval/eval_matrix.yaml --model checkpoints/stage_e_memory_sft/latest

# 10M 端到端
make e2e-eval
```

## 输出

每个 runner 写到 `eval/_outputs/<run_name>/`，包含：
- `metrics.json` — 主指标
- `predictions.jsonl` — per-instance 输出
- `summary.md` — 人读 markdown

## Baseline

W1 第一个交付物是 `scripts/baseline_eval.py`，跑 Qwen3.5-4B 在 NIAH 128K-1M 上的表现，
给后续 CPT 提供对照。
