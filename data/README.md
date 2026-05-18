# Data

负责长上下文 CPT + SFT 的数据准备。

## 目录约定

```
data/
├── raw/           # 原始下载（gitignored）：arXiv tar, Project Gutenberg, Pile-of-Law, ...
├── processed/     # 加工后 jsonl，可直接喂训练（gitignored）
├── synthetic/     # 合成数据（needle / multi-hop / timeline）（gitignored）
├── hf_cache/      # huggingface_hub / datasets cache（gitignored）
├── samples/       # 小样本，git tracked，用于 unit test 和 dry-run
└── scripts/       # 数据准备脚本
```

## CPT 数据来源（英文为主 8:2，见 spec §4.4）

| 来源 | 工具 | 比例 |
|---|---|---|
| arXiv 全文 | `datasets:arxiv_dataset` / arXiv bulk S3 | 20% |
| Books (PG-19 / Project Gutenberg) | `datasets:pg19` / Gutenberg dump | 15% |
| 多文档主题包（arXiv 同主题 / Wiki topic packs） | `data/scripts/pack_docs.py` | 20% |
| 长技术文档 / Wiki / Pile-of-Law | `datasets:pile-of-law` | 15% |
| 中文长文 (WanJuan-Long, SkyPile-Long) | OpenDataLab / HF | 10% |
| 合成 needle / multi-needle | `data/scripts/synthesize_needle.py` | 8% |
| 合成长推理 (multi-hop / timeline) | `data/scripts/synthesize_long_reasoning.py` | 7% |
| 短高质量 (Tulu-3 / OpenInstruct) | `datasets:` | 5% |

## SFT 数据

- **Short SFT (Stage D)**：Tulu-3 + MetaMathQA + Magicoder 各 5-15K，总 ~50K examples
- **Memory-format SFT (Stage E)**：自建 20K，用 Claude-Opus 或 Qwen3-72B-Instruct 做 teacher 生成 grounded multi-doc QA + citation

## 脚本入口

| 脚本 | 用途 | 何时跑 |
|---|---|---|
| `scripts/prepare_long_docs.py` | 下载、清洗、切 chunk、写 jsonl | W1-2 |
| `scripts/pack_docs.py` | 多文档主题打包 + length upsampling | W2 |
| `scripts/synthesize_needle.py` | NIAH / multi-needle 合成 | W2 |
| `scripts/synthesize_long_reasoning.py` | multi-hop / timeline / 冲突合成 | W3 |
| `scripts/build_memory_sft.py` | memory-format SFT 数据生成（teacher 蒸馏） | W9 |
