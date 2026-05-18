# Pipeline

端到端在线推理。把 retrieval + reader 串成一个 service。

## 启动

```bash
# 1. 启依赖（OpenSearch, FAISS, Postgres）
docker compose up -d  # TODO: docker-compose.yml W3

# 2. 启 reader serving (vLLM / SGLang)
uv run python pipeline/serve.py --reader-ckpt checkpoints/stage_e_memory_sft/latest --port 8001

# 3. 启 router (业务入口)
uv run python pipeline/router.py --port 8000
```

## Router 策略 (spec §12.3)

W11 初版：rule-based
- query 简单 + top evidence 置信度高 → RAG 32K/64K
- query 多文档聚合 → LongRAG 128K/256K
- query 全局/时间线 → hierarchical 512K/1M

W12+ 可选：self-route（query + evidence sufficiency classifier）。
