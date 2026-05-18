# longluxi

Qwen3.5-4B 超长上下文项目 —— 把 4B 模型从 YaRN-static-1M 推到 **dense 2M-4M reader**，配 **hierarchical retrieval** 端到端处理 **10M+ tokens** 输入。

主场景：长文档 / 多文档研究 QA。硬件预算：16×H100 (2×8 节点 + IB)。时间窗口：3 个月 MVP。

完整设计见 [`docs/superpowers/specs/2026-05-18-qwen3.5-4b-10m-context-design.md`](docs/superpowers/specs/2026-05-18-qwen3.5-4b-10m-context-design.md)。

---

## Quick Start

```bash
# 1. 装最小依赖（不含 torch/vllm 等重型）
uv sync

# 2. 按需装重型 extras（其中之一或全部）
uv sync --extra eval         # 跑 baseline / NIAH / RULER 评测
uv sync --extra retrieval    # 检索系统（FAISS + OpenSearch + reranker）
uv sync --extra train        # 训练栈（torch + transformers + flash-attn + deepspeed）
uv sync --extra inference    # vllm / sglang serving
uv sync --extra all          # 一次装齐
uv sync --extra dev          # 开发工具

# 3. flame 框架走 git，单独 clone
make setup-flame             # 把 fla-org/flame 拉到 external/flame

# 4. 跑基线评测（W1 第一个交付物）
make baseline-eval
# 或：
uv run python scripts/baseline_eval.py --model-id Qwen/Qwen3.5-4B --task niah --max-len 131072 --limit 16
```

---

## 项目结构

```
longluxi/
├── docs/superpowers/specs/      # 设计文档（spec）
├── configs/
│   ├── yarn/                    # YaRN config JSON (1M/2M/4M)
│   ├── training/                # 每个 stage 的训练配置 TOML
│   ├── retrieval/               # chunking / index / pipeline 配置
│   └── eval/                    # 评测 matrix
├── src/longluxi/                # 公共 Python 包（路径常量、工具）
├── data/                        # 数据准备脚本与样例
│   └── scripts/
├── training/                    # 训练入口（shell wrapper for flame/megatron）
│   ├── flame_wrapper/
│   └── scripts/
├── retrieval/                   # hierarchical retrieval 系统
│   ├── index/
│   └── pipeline/
├── eval/                        # 评测脚本
│   ├── runners/
│   └── datasets/
├── pipeline/                    # 端到端推理 pipeline 与 server
├── scripts/                     # 一次性运行脚本（baseline_eval 等）
├── tests/                       # 单元/集成测试
├── external/                    # 第三方 clone（flame, FLA, etc.）
├── ttt.md                       # 原始 brainstorming 输入
├── Makefile                     # 常用命令入口
├── pyproject.toml               # uv 项目定义
└── .python-version              # 3.11
```

---

## 12 周里程碑

| 周 | 训练轨 | 检索轨 |
|---|---|---|
| W1 | env + baseline eval | 数据 ingest + index schema |
| W2 | 1M smoke 50M tokens | 8K/64K/512K chunking pipeline |
| W3 | 1M main 250M tokens | BM25 + FAISS index v0 |
| W4 | 2M CPT 150-200M | RAPTOR summary tree v0 |
| W5 | 2M eval + ablation | hybrid retrieve + rerank |
| W6 | 4M boundary (optional) | 邻居展开 + packing |
| W7 | 4M eval / go-no-go | query-aware compression v0 |
| W8 | short SFT recovery | compression v1 |
| W9 | memory-format SFT | e2e pipeline 拼接 |
| W10 | SFT 后半 + final eval | LongBench v2 / InfiniteBench / NoCha |
| W11 | — | 自建 10M benchmark + 全 ablation |
| W12 | — | 内部 demo + tech report |

---

## 评测报告

- [`docs/reports/BASELINE_W1.md`](docs/reports/BASELINE_W1.md) —— Phase 1 基线
  （Qwen3.5-4B 在 NIAH 32K/128K/512K-YaRN 与 RULER 128K 上的表现；
  headline weakness = RULER `vt` 20% @ 128K；1M 受限于单卡内存）。Tag `phase1-baseline`.

## 关键设计取舍

- **不靠纯 retrieval 假装 10M**：reader 自身必须 dense 推到 2M-4M
- **不改 attention 架构**：保留 Qwen3.5-4B GDN/standard 混合配比，算法侧只动 YaRN / packing / mask / data
- **MVP 不做 graph retrieval**：只做 BM25 + dense + summary tree
- **优先 flame，后备 Megatron-Core**：W2 smoke 验证 flame 在 1M 长度稳定性
- **memory-format SFT 用 stronger teacher 蒸馏**：自建数据 ROI 在 3 个月窗口太低

---

## License

Apache-2.0
