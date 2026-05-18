# Retrieval

Hierarchical retrieval 系统：把 10M+ tokens 原始上下文压成 reader 能消化的 1M-4M 输入。

见 spec §5。

## 流程

```
docs/  →  chunk (8K/64K/512K)  →  index (BM25 + FAISS + summary tree)
                                       │
query  → query_processing  →  hybrid retrieve  →  rerank  →  neighbor expand
                                       │
                              query-aware compression
                                       │
                              order-preserving packing
                                       │
                              → reader input
```

## 启动

```bash
# 1. 准备 chunk
uv run python data/scripts/prepare_long_docs.py --source arxiv ...
# 2. 切 chunk
uv run python retrieval/index/chunk_docs.py --config configs/retrieval/chunking.yaml
# 3. 建索引
make build-index
# 4. 端到端检索
uv run python retrieval/pipeline/hybrid_retrieve.py --query "..." --top-k 40
```

## 索引位置

- BM25 (OpenSearch): index 名 `longluxi_raw_chunks`，由 `index/build_bm25.py` 写入
- Dense (FAISS): `retrieval/indices/dense_bge_m3/`
- Summary tree: `retrieval/indices/summary_tree.json`（in-memory，启动加载）
- Metadata (Postgres): schema 见 `index.yaml`

## 配置

见 `configs/retrieval/`:
- `chunking.yaml` — 三层粒度
- `index.yaml` — 索引栈
- `pipeline.yaml` — 检索 / rerank / 压缩 / packing
