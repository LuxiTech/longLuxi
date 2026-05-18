4B 级超长上下文模型与 10M Hierarchical Retrieval 系统技术方案

0. 摘要

本项目目标不是在有限算力下暴力训练一个 4M/10M dense full-attention 模型，而是在 16 张 H100 的约束下，建立一套可执行、可扩展、可验证的超长上下文方法论。

核心路线：

使用现有 3B/4B 级开源模型作为 base，通过 YaRN + continued pretraining 训练出 512K/1M long-context reader；再通过 hierarchical retrieval、long retrieval units、query-aware compression、order-preserving packing 和 memory-format SFT，把 4M/10M 级原始上下文转化为 128K–1M 的高证据密度输入，由 long-context reader 完成最终推理和回答。

最终产出包括：

1. 一个 512K/1M 上下文的 3B/4B long-context reader。
2. 一个面向 10M token 原始上下文的 hierarchical retrieval 系统。
3. 一套 dense long-context、普通 RAG、LongRAG、hierarchical retrieval 的系统性对比评测。
4. 一套适用于有限算力下小模型超长上下文扩展的方法论。

⸻

1. 项目背景

近年来，LLM 长上下文能力快速扩展，从 32K、128K 到 1M、4M 甚至更长。但在工程实践中，直接把 context window 拉到 4M/10M 并不总是最优方案。

主要问题包括：

* dense attention 计算复杂度随序列长度平方增长；
* 4M/10M full-attention training 对算力、显存和通信要求极高；
* 真实任务中，10M tokens 中的有效证据通常只占很小比例；
* 长上下文模型即使能“看见”所有 token，也未必能稳定检索、聚合和推理；
* 单纯 needle-in-a-haystack 满分不能代表真实长上下文能力。

因此，本项目将问题重新定义为：

在固定 16×H100 算力下，如何让 3B/4B 级模型有效利用 1M–10M 规模上下文？

这里的“有效利用”不等于模型对 10M tokens 做 dense full attention，而是包括：

* long-context reader 能力；
* 跨 chunk 检索能力；
* 层级摘要和记忆组织能力；
* query-aware evidence compression；
* 多证据聚合与冲突处理能力；
* 可控成本下的推理服务能力。

⸻

2. 资源约束与设计原则

2.1 资源约束

当前硬件约束：

GPU: 16 × H100
目标模型规模: 3B / 4B 为主，7B 作为可选对照
目标上下文: reader 512K/1M，系统有效覆盖 4M/10M
训练方式: continued pretraining + SFT，不从零预训练

2.2 不采用的路线

路线	是否采用	原因
从零训练 4B/8B	否	16 H100 成本过高，质量不可控
重新设计 sparse attention 架构	第一阶段否	需要大规模预训练和 kernel 工程
10M dense full attention	否	计算和通信成本不现实
只做普通 RAG	否	无法体现 long-context reader 的价值
只看 Needle benchmark	否	无法代表真实长上下文能力

2.3 采用的路线

本项目采用：

已有 base model
  → YaRN position scaling
  → long-context continued pretraining
  → short instruction recovery SFT
  → memory-format SFT
  → hierarchical retrieval system
  → 10M evaluation

核心原则：

1. 不改 backbone 或只做极小侵入改造。
2. 先训练一个可靠的 512K/1M reader。
3. 用系统工程覆盖 4M/10M，而不是暴力 dense attention。
4. 评测 retrieval、aggregation、reasoning、compression，而不是只测 needle。
5. 所有设计都要可 ablation、可复现、可落地。

⸻

3. 总体技术架构

整体架构分为三层：

Layer 1: Long-context reader
  - 3B/4B base model
  - YaRN RoPE scaling
  - 512K/1M continued pretraining
  - short SFT + memory-format SFT
Layer 2: Hierarchical retrieval memory system
  - raw chunks
  - large chunks
  - section summaries
  - recursive summary tree
  - sparse/dense hybrid index
  - optional entity/event/code graph
Layer 3: Adaptive inference pipeline
  - query decomposition
  - hybrid retrieval
  - reranking
  - order-preserving packing
  - query-aware compression
  - long-context reader answer
  - citation / verification

系统处理流程：

10M raw context
  ↓
Document parsing / chunking
  ↓
Multi-level indexing
  ↓
Hybrid retrieval
  ↓
Reranking
  ↓
Order-preserving packing
  ↓
Query-aware compression
  ↓
512K/1M long-context reader
  ↓
Answer with evidence

⸻

4. 模型选型

4.1 主模型

推荐主模型：

Qwen3.5-4B / Qwen3.5-4B-Instruct

原因：

* 参数规模适合 16×H100 做多轮 ablation；
* 中文、英文、代码能力相对均衡；
* 比 7B/8B 更适合作为方法学实验载体；
* 训练 512K/1M 上下文的成本更低；
* 后续 recipe 可迁移到 7B/8B。

4.2 可选强模型对照

Qwen2.5-7B / Qwen2.5-7B-Instruct
Llama-3.1-8B-Instruct
Llama-3.2-3B

建议策略：

方法学主线: Qwen3.5-4B
最终效果对照: Qwen2.5-7B 或 Llama-3.1-8B

⸻

5. 长上下文模型训练方案

5.1 训练目标

Reader 模型目标不是 10M dense context，而是：

阶段 1: 256K smoke test
阶段 2: 512K stable reader
阶段 3: 1M main reader
阶段 4: 2M/4M dense boundary proof，optional

推荐主目标：

Qwen3.5-4B-512K-Reader
Qwen3.5-4B-1M-Reader

5.2 Position Encoding

主选：

YaRN RoPE scaling

对照实验：

Linear RoPE scaling
NTK-aware RoPE scaling
YaRN

第一阶段不建议投入过多资源探索复杂 position encoding。核心任务是跑通 reader + hierarchical retrieval methodology。

5.3 Continued Pretraining 数据配比

推荐 CPT 数据总量：

512K reader: 300M–500M tokens
1M reader: 500M–1B tokens

数据配比建议：

数据类型	比例	作用
Code repositories / technical docs	30%	跨文件、结构化依赖、长距离引用
Natural long documents	25%	长文阅读和段落级结构建模
Multi-document corpora	20%	跨文档聚合和多源证据
Synthetic retrieval	10%	位置鲁棒性、needle/multi-needle 能力
Synthetic long reasoning	10%	多跳、时间线、冲突检测
Short high-quality data	5%	防止短任务能力退化

中英比例建议：

英文:中文 = 60:40 或 50:50

5.4 文档长度采样策略

建议采用长度上采样：

<4K tokens: 下采样
4K–8K: 正常采样
8K–64K: 上采样
64K–256K: 强上采样
>256K: 尽量保留

同时保留一定比例短高质量数据，避免模型短上下文能力退化。

5.5 Packing 策略

使用多文档拼接：

<doc id="..." source="...">
...
</doc>
<doc_sep>
<doc id="..." source="...">
...
</doc>

注意：

* 使用专门的 document separator；
* 不对不同文档之间做 cross-document attention mask；
* 不滥用 BOS/EOS 作为每篇文档分隔；
* 保留 metadata 以支持后续 memory-format SFT。

5.6 训练阶段设计

Stage 1: 256K smoke test

Base model: Qwen3.5-4B
Context length: 256K
Training tokens: 50M–100M
Goal: 跑通训练系统、loss、checkpoint、评测链路

成功标准：

* loss 稳定下降；
* NIAH 在 128K/256K 有明显提升；
* 训练无显存、通信和 checkpoint 阻塞问题。

Stage 2: 512K reader

Context length: 512K
Training tokens: 300M–500M
Goal: 得到稳定 long-context reader

成功标准：

* 512K NIAH / passkey 稳定；
* RULER 128K/256K/512K 明显优于 base；
* 短 benchmark 退化可控；
* 可用于 hierarchical retrieval 的 long reader。

Stage 3: 1M reader

Context length: 1M
Training tokens: 500M–1B
Goal: 得到主力 1M reader

成功标准：

* 1M NIAH 基本稳定；
* multi-needle 和 long QA 有提升；
* 能处理 128K–512K evidence context；
* memory-format SFT 后能输出 grounded answer。

Stage 4: 2M/4M dense boundary optional

Context length: 2M / 4M
Training tokens: 100M–300M
Goal: 评估 dense full attention 的边界，不作为第一阶段主线

该阶段主要用于研究：

* 16 H100 下 dense training 的性价比边界；
* 2M/4M 训练是否带来真实任务收益；
* 何时应该从 dense context window 转向 hierarchical retrieval。

⸻

6. SFT 方案

SFT 分为两类：

1. Instruction recovery SFT；
2. Memory-format SFT。

6.1 Instruction Recovery SFT

目的：恢复 continued pretraining 后可能下降的 chat / instruction following 能力。

配置：

Examples: 30K–80K
Max length: 8K–32K
Learning rate: 5e-6
Steps: 100–500

数据类型：

General instruction
Math
Code
Chinese QA
English QA
Tool-use style QA，optional

注意：

* 不要过度 SFT；
* 每 50–100 steps 存 checkpoint；
* 监控 long-context eval，避免 SFT 损害长上下文能力。

6.2 Memory-format SFT

这是本项目的关键特色。

目标：让模型学会读取 hierarchical retrieval 系统输出的结构化 evidence context。

统一输入格式：

<query>
用户问题
</query>
<global_summary>
全局摘要
</global_summary>
<section_summary id="S12" source="...">
section 级摘要
</section_summary>
<evidence id="E01" source="docA" pos="123000-131000">
原文证据或压缩证据
</evidence>
<evidence id="E02" source="docB" pos="912000-920000">
另一条证据
</evidence>
<conflicts>
潜在冲突证据
</conflicts>
<answer>
最终回答，带 evidence 引用
</answer>

训练任务：

任务	说明
Evidence selection	从多个 evidence 中选择相关证据
Answer with citation	基于 evidence 回答并引用 source_id
Multi-hop aggregation	聚合多个 chunk 的信息
Contradiction resolution	处理不同文档之间的冲突
Timeline QA	根据长时间线回答状态变化
Code repo QA	根据跨文件证据回答代码问题
Query-focused summarization	围绕 query 总结超长上下文

建议数据规模：

20K–50K examples
Max length: 64K–256K

训练目标可以包括：

selected_evidence_ids
answer
citations
confidence

不建议强制模型输出长 chain-of-thought。可以输出简短 evidence rationale，但最终部署默认只输出答案和证据引用。

⸻

7. 训练工程选型

7.1 训练框架

首选：

Megatron-LM / NVIDIA NeMo

备选：

DeepSpeed Ulysses + FlashAttention

不建议：

纯 HuggingFace Trainer

原因：

* 512K/1M sequence training 需要 context parallelism；
* 普通 FSDP/ZeRO 无法有效解决 attention activation 和 sequence 维度切分；
* 需要高效 checkpoint、activation recomputation、FlashAttention 和分布式通信优化。

7.2 并行策略

针对 3B/4B：

512K:
  TP = 1 or 2
  CP = 8
  DP = 1 or 2
  micro_batch = 1 sequence
1M:
  TP = 1 or 2
  CP = 16
  DP = 1
  micro_batch = 1 sequence
  gradient_accumulation 调整 global batch tokens

针对 7B：

512K:
  TP = 2
  CP = 8
  DP = 1
1M:
  TP = 2, CP = 8
  或 TP = 4, CP = 4
  activation recomputation 必开

7.3 显存与效率策略

必须开启：

bf16
FlashAttention
Activation recomputation
Sequence packing
Optimizer state sharding
Async checkpointing
Gradient accumulation

建议：

micro_batch = 1 sequence
通过 gradient_accumulation 控制 global batch tokens
不要盲目追求大 batch

7.4 Checkpoint 策略

Smoke stage: 每 10M–20M tokens 存 checkpoint
Main stage: 每 50M–100M tokens 存 checkpoint
SFT stage: 每 50–100 steps 存 checkpoint

每个 checkpoint 都应自动跑：

NIAH
RULER subset
Short benchmark subset
Long retrieval QA subset

⸻

8. Hierarchical Retrieval 系统设计

8.1 Chunking 策略

不要使用传统 512-token chunk 作为主 retrieval unit。对于 long-context reader，chunk 应该更长。

推荐三层粒度：

raw_chunk:   8K tokens
large_chunk: 64K tokens
section:     512K tokens

含义：

粒度	作用
8K raw chunk	embedding / BM25 / rerank 基本单元
64K large chunk	保留局部上下文，适合精读
512K section	构建 section summary，适合全局定位

8.2 多层索引

构建四类索引：

1. Sparse lexical index
   - BM25 / OpenSearch / Elasticsearch
2. Dense embedding index
   - FAISS / Milvus / Qdrant
   - bge-m3 / jina-embeddings / e5 系列
3. Summary tree
   - RAPTOR-style recursive summaries
   - chunk → cluster → summary → higher-level summary
4. Entity/event/code graph，optional
   - entity relation
   - event timeline
   - code symbol graph
   - issue / PR / commit relation

第一版建议：

OpenSearch + FAISS + PostgreSQL

图数据库可第二阶段加入。

8.3 数据 schema

Chunk schema

chunk_id
parent_doc_id
parent_section_id
start_token
end_token
text
embedding
summary
entities
events
timestamp
source_path
prev_chunk_id
next_chunk_id
metadata

Section schema

section_id
parent_doc_id
child_chunk_ids
start_token
end_token
summary
embedding
entities
events
time_range
metadata

Graph schema，optional

node_a
relation
node_b
source_chunk_id
confidence
timestamp
metadata

⸻

9. Retrieval Pipeline

9.1 Query processing

输入 query 后，先进行：

query normalization
query rewrite
query decomposition，optional
keyword/entity extraction
time constraint extraction，optional

示例：

原始 query:
为什么 login timeout 在 v2.3 之后变长了？
解析后:
entities: login timeout, v2.3
intent: cause analysis
time/version constraint: after v2.3
retrieval terms: login timeout, timeout config, v2.3, release notes, migration

9.2 Hybrid recall

召回策略：

BM25 top 200
Dense embedding top 200
Summary tree top 100
Graph neighbor expansion top 100，optional

合并去重后得到 300–500 个 candidates。

不同召回方式的作用：

方法	强项
BM25	实体、函数名、错误码、日期、版本号
Dense	语义改写、模糊问题
Summary tree	全局主题、长文结构
Graph	多跳关系、实体事件、代码依赖

9.3 Reranking

将 candidates 压缩到 top 40–80。

第一版可用：

bge-reranker
jina-reranker
Qwen-based reranker

第二版可用：

long-context reader as listwise reranker

Rerank 输入应包含：

query
chunk_text
chunk_summary
doc_title
section_title
source_path
timestamp
neighbor summaries
metadata

9.4 Neighbor expansion

对 top chunks 做局部扩展：

previous chunk
next chunk
same section summary
same document nearby chunks
graph 1-hop neighbors，optional

目的是避免只取到孤立证据，丢掉上下文。

9.5 Order-preserving packing

不要直接按 relevance score 排序拼接。

推荐：

1. 按 relevance 选择 chunks
2. 合并相邻 chunks 成 continuous spans
3. 按原始文档顺序 / 时间顺序 / 代码依赖顺序重排
4. 每个 span 前加入 metadata header
5. 控制最终 token budget

Packing 格式：

<retrieved_context>
<source id="S01" doc="..." path="..." start="..." end="..." score="...">
...
</source>
<source id="S02" doc="..." path="..." start="..." end="..." score="...">
...
</source>
</retrieved_context>

⸻

10. Query-aware Compression

10.1 目标

Query-aware compression 的目标不是普通摘要，而是：

从 retrieved text 中抽取与 query 相关的证据、事实、时间、实体、代码路径和冲突信息。

输入：

query
chunk_text
chunk_summary
metadata

输出：

relevance
key evidence spans
entities
dates / versions / file paths
claim
supporting quote or line ref
conflicts
uncertainty

10.2 压缩格式

<evidence id="E01" source="S01" relevance="5">
  <claim>...</claim>
  <support_span>...</support_span>
  <why_relevant>...</why_relevant>
  <entities>...</entities>
  <time_or_version>...</time_or_version>
  <conflicts>...</conflicts>
</evidence>

10.3 压缩倍率

目标：

Raw retrieved text: 1M–2M tokens
Compressed evidence: 128K–512K tokens
Reader input: 128K–1M tokens

压缩模型：

第一版: Qwen2.5-7B-Instruct / existing strong instruct model
第二版: self-trained 512K/1M reader
第三版: distilled compressor，optional

10.4 Compression ablation

需要比较：

No compression
Generic summary
Query-aware evidence extraction
Query-aware evidence + citations
Query-aware evidence + conflict detection

⸻

11. Reader 输入格式

最终喂给 reader 的输入应统一格式化。

<task>
请基于给定证据回答问题。回答需要引用 evidence id。如果证据不足，请说明缺失信息。
</task>
<query>
...
</query>
<global_summary>
...
</global_summary>
<section_summaries>
<section id="SEC01">...</section>
<section id="SEC02">...</section>
</section_summaries>
<evidence_list>
<evidence id="E01" source="S01" position="...">
...
</evidence>
<evidence id="E02" source="S02" position="...">
...
</evidence>
</evidence_list>
<answer_format>
- answer
- evidence_ids
- confidence
- missing_information, if any
</answer_format>

输出格式：

{
  "answer": "...",
  "evidence_ids": ["E01", "E05"],
  "confidence": "high|medium|low",
  "missing_information": "..."
}

⸻

12. Inference Serving 方案

12.1 第一版 serving

Retriever service:
  OpenSearch + FAISS
Reranker service:
  reranker model
Compressor service:
  existing instruct model or self-trained reader
Reader service:
  vLLM / SGLang
  TP across GPUs
  chunked prefill if supported

12.2 Reader 长度策略

普通 query: 32K–64K context
复杂 query: 128K–256K context
超长 query: 512K–1M context

12.3 Router 策略

初版规则路由：

if query is simple and top evidence confidence high:
    use RAG 32K/64K
elif query needs multi-doc aggregation:
    use LongRAG 128K/256K
elif query needs global or timeline reasoning:
    use hierarchical retrieval 512K/1M
else:
    retrieve more / ask for clarification / return insufficient evidence

后续可训练 self-route：

query + retrieved evidence
→ evidence sufficiency classifier
→ choose RAG / LongRAG / hierarchical pipeline

⸻

13. 评测方案

评测分两层：

1. Reader 单体评测；
2. 10M 系统评测。

13.1 Reader 单体评测

能力	Benchmark / Task
位置检索	NIAH, passkey
多针检索	multi-needle
长上下文鲁棒性	RULER
长文 QA	LongBench / InfiniteBench / LV-Eval
代码长上下文	repo QA
短能力保持	MMLU, GSM8K, HumanEval, C-Eval / CMMLU

长度设置：

32K
128K
256K
512K
1M

13.2 10M 系统评测

构造 10M-token task instances。

任务类型：

任务	说明
10M passkey	基础位置检索 sanity check
10M multi-needle aggregation	多证据聚合
10M contradiction detection	跨文档冲突识别
10M timeline QA	长时间线状态变化
10M codebase QA	跨文件、issue、PR、release notes 推理
10M query-focused summarization	围绕 query 的全局摘要

13.3 对比系统

比较以下系统：

A. BM25 RAG + 32K reader
B. Dense RAG + 32K reader
C. Hybrid RAG + 64K reader
D. LongRAG-style long chunks + 128K/256K reader
E. RAPTOR-style summary tree + 512K/1M reader
F. Graph/hierarchy + 512K/1M reader
G. Direct dense 1M reader，when input can be truncated to 1M

13.4 指标

Accuracy / F1
Evidence recall
Answer grounding rate
Citation precision
Contradiction resolution accuracy
Compression ratio
Latency
GPU memory
Token cost
End-to-end throughput

核心研究问题：

1. 1M reader 是否显著优于 32K/64K reader？
2. LongRAG-style long chunk 是否优于传统小 chunk RAG？
3. Query-aware compression 是否优于 generic summary？
4. Order-preserving packing 是否优于 relevance-score packing？
5. Hierarchical retrieval 在哪些任务上超过普通 RAG？
6. dense context window 增长到什么时候不再划算？

⸻

14. Ablation 设计

14.1 Reader ablation

变量	设置
Context length	256K / 512K / 1M
CPT tokens	100M / 300M / 500M / 1B
RoPE scaling	Linear / NTK / YaRN
Data mix	balanced / code-heavy / synthetic-heavy / long-doc-heavy
SFT	none / short SFT / memory-format SFT

14.2 Retrieval ablation

变量	设置
Chunk size	4K / 8K / 16K / 32K
Retrieval method	BM25 / dense / hybrid / hybrid+summary
Rerank	none / reranker / LLM rerank
Neighbor expansion	off / prev-next / section / graph
Packing	relevance order / original order / timeline order

14.3 Compression ablation

变量	设置
Compression type	none / generic summary / query-aware evidence
Compression ratio	2× / 4× / 8× / 16×
Compressor	existing instruct / self-trained reader / distilled compressor
Output format	free text / structured evidence / evidence with citations

14.4 System ablation

变量	设置
Reader context	64K / 128K / 256K / 512K / 1M
Raw context	1M / 4M / 10M
Pipeline	RAG / LongRAG / hierarchical / graph+hierarchical
Router	fixed / rule-based / self-route

⸻

15. 里程碑计划

Phase 0: 环境与基线

目标：跑通训练、评测和系统原型。

任务：

搭建 Megatron-LM / NeMo training stack
加载 Qwen3.5-4B
搭建 NIAH/RULER/short benchmark eval
搭建 OpenSearch + FAISS retrieval prototype
构造 10M toy corpus

产出：

Base model eval report
Training smoke test
Retrieval prototype v0

Phase 1: 256K smoke

任务：

256K YaRN CPT 50M–100M tokens
跑 NIAH/RULER/short eval
验证 checkpoint 与 training stability

产出：

Qwen3.5-4B-256K-smoke
训练系统稳定性报告

Phase 2: 512K reader

任务：

512K CPT 300M–500M tokens
instruction recovery SFT
memory-format SFT v0
LongRAG-style retrieval pipeline

产出：

Qwen3.5-4B-512K-Reader
10M → 128K/256K evidence pipeline
Reader eval report

Phase 3: 1M reader

任务：

1M CPT 500M–1B tokens
memory-format SFT v1
RAPTOR-style summary tree
query-aware compression
order-preserving packing

产出：

Qwen3.5-4B-1M-Reader
10M hierarchical retrieval system
10M benchmark v1

Phase 4: Ablation 与报告

任务：

chunk size ablation
retrieval method ablation
compression ablation
reader length ablation
router ablation
optional 2M/4M dense boundary experiment

产出：

technical report
model checkpoint
retrieval system
benchmark suite

⸻

16. 风险与缓解

风险 1: 1M 训练不稳定

缓解：

先完成 256K/512K
降低 learning rate
减少 global batch tokens
检查 YaRN 参数
启用更强 activation recomputation
缩短 sequence 做 debug

风险 2: 长上下文 benchmark 好，但真实任务差

缓解：

不要只看 NIAH
加入 multi-needle、contradiction、timeline、repo QA
构造真实业务任务
监控 evidence grounding

风险 3: SFT 损害长上下文能力

缓解：

SFT 步数少
每 50–100 steps eval
保留 CPT checkpoint
短 SFT 与 memory-format SFT 分开对比

风险 4: Retrieval recall 不足

缓解：

hybrid retrieval
query rewrite
neighbor expansion
summary tree retrieval
graph expansion，optional
提高 top-k 后再压缩

风险 5: Compression 丢失关键信息

缓解：

保留原始 evidence span
压缩结果带 source id
reader 输入混合 compressed evidence + raw snippets
评测 evidence recall

风险 6: 系统复杂度过高

缓解：

第一版只做 OpenSearch + FAISS + summary tree
图记忆放到第二阶段
router 先用规则，后续再训练

⸻

17. 最终推荐技术选型汇总

Model

Base: Qwen3.5-4B / Qwen3.5-4B-Instruct
Optional: Qwen2.5-7B as stronger baseline
Position: YaRN
Context: 512K → 1M
Training: continued pretraining + SFT

Training

Framework: Megatron-LM / NeMo
Precision: bf16
Attention: FlashAttention
Parallelism: TP + CP
Memory: activation recomputation
Batch: micro_batch=1 + gradient accumulation

Data

30% code / technical docs
25% natural long docs
20% multi-doc corpora
10% synthetic retrieval
10% synthetic long reasoning
5% short high-quality data

Retrieval

Chunk: 8K raw / 64K large / 512K section
Index: OpenSearch + FAISS + PostgreSQL
Hierarchy: RAPTOR-style summary tree
Graph: optional second phase
Rerank: bge/jina/Qwen reranker
Packing: order-preserving
Compression: query-aware evidence extraction

Serving

Reader: vLLM / SGLang where possible
Pipeline: retriever → reranker → compressor → reader → verifier
Context tiers: 64K / 128K / 256K / 512K / 1M
Router: rule-based first, self-route later

Evaluation

Reader: NIAH / RULER / InfiniteBench / repo QA / short benchmark
System: 10M passkey / multi-needle / contradiction / timeline / codebase QA / query-focused summarization

⸻

18. 项目一句话定义

本项目不是训练一个暴力 10M dense attention 模型，而是：

在 16×H100 约束下，训练一个 3B/4B 级 512K/1M long-context reader，并通过 hierarchical retrieval、long retrieval units、query-aware compression 和 memory-format SFT，让小模型有效利用 4M/10M 规模上下文。

核心贡献是：

1. 一个有限算力可落地的 long-context reader 训练 recipe；
2. 一个面向 10M 原始上下文的 hierarchical retrieval 系统；
3. 一套证明 dense context、RAG、LongRAG、hierarchical retrieval 边界的方法论；
4. 一个适合小模型超长上下文研究的评测和 ablation 框架。