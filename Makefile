# longluxi Makefile —— 项目常用命令入口
#
# 设计原则：每条 target 一行 uv 命令，方便照搬到 README/notebook。
# 重型依赖通过 `uv sync --extra <name>` 按需安装；不在 make 里自动装重的东西。

UV ?= uv
PYTHON ?= $(UV) run python
MODEL_ID ?= Qwen/Qwen3.5-4B-Instruct

.PHONY: help
help:
	@echo "longluxi targets:"
	@echo "  sync                  install base deps only"
	@echo "  sync-eval             + eval extras"
	@echo "  sync-train            + train extras (heavy: torch, flash-attn, deepspeed)"
	@echo "  sync-retrieval        + retrieval extras (faiss, opensearch, FlagEmbedding)"
	@echo "  sync-inference        + serving extras (vllm, sglang)"
	@echo "  sync-all              install everything"
	@echo "  setup-flame           clone flame + flash-linear-attention into external/"
	@echo "  setup-llamafactory    clone hiyouga/LLaMA-Factory into external/"
	@echo "  setup-ruler           clone NVIDIA/RULER into external/"
	@echo "  setup-external        clone both"
	@echo "  baseline-eval         W1: run Qwen3.5-4B baseline NIAH @ 128K"
	@echo "  smoke-1m              W2: launch 1M CPT smoke run"
	@echo "  stage-a               W2-3: full 1M CPT"
	@echo "  stage-b               W4-5: 2M CPT"
	@echo "  stage-c               W6-7: 4M boundary (optional)"
	@echo "  stage-d               W8: short SFT recovery"
	@echo "  stage-e               W9-10: memory-format SFT"
	@echo "  build-index           build BM25 + FAISS + summary tree on data/processed/"
	@echo "  e2e-eval              W11: 10M end-to-end benchmark"
	@echo "  test                  pytest"
	@echo "  lint                  ruff check"
	@echo "  fmt                   ruff format"

# ---------------- uv sync targets ----------------

.PHONY: sync sync-eval sync-train sync-retrieval sync-inference sync-all sync-dev
sync:
	$(UV) sync

sync-eval:
	$(UV) sync --extra eval

sync-train:
	$(UV) sync --extra train

sync-retrieval:
	$(UV) sync --extra retrieval

sync-inference:
	$(UV) sync --extra inference

sync-all:
	$(UV) sync --extra all

sync-dev:
	$(UV) sync --extra dev

# ---------------- External setup ----------------

.PHONY: setup-flame setup-llamafactory setup-ruler setup-external
setup-flame:
	bash training/flame_wrapper/setup_flame.sh

setup-llamafactory:
	bash training/llamafactory_wrapper/setup_llamafactory.sh

setup-ruler:
	mkdir -p external
	[ -d external/RULER ] || git clone --depth 1 https://github.com/NVIDIA/RULER.git external/RULER
	cd external/RULER && git pull --ff-only

setup-external: setup-flame setup-llamafactory setup-ruler

# ---------------- Baseline eval (W1) ----------------

.PHONY: baseline-eval baseline-eval-1m
baseline-eval:
	$(PYTHON) scripts/baseline_eval.py \
		--model-id $(MODEL_ID) \
		--task niah \
		--max-len 131072 \
		--limit 16

baseline-eval-1m:
	$(PYTHON) scripts/baseline_eval.py \
		--model-id $(MODEL_ID) \
		--task niah \
		--max-len 1010000 \
		--yarn-factor 4.0 \
		--limit 8

# ---------------- Training stages ----------------

.PHONY: smoke-1m stage-a stage-b stage-c stage-d stage-e
smoke-1m:
	bash training/scripts/run_stage_a.sh --smoke

stage-a:
	bash training/scripts/run_stage_a.sh

stage-b:
	bash training/scripts/run_stage_b.sh

stage-c:
	bash training/scripts/run_stage_c.sh

stage-d:
	bash training/scripts/run_short_sft.sh

stage-e:
	bash training/scripts/run_memory_sft.sh

# ---------------- Retrieval ----------------

.PHONY: build-index
build-index:
	$(PYTHON) retrieval/index/build_bm25.py
	$(PYTHON) retrieval/index/build_dense.py
	$(PYTHON) retrieval/index/build_summary_tree.py

# ---------------- E2E eval (W11) ----------------

.PHONY: e2e-eval
e2e-eval:
	$(PYTHON) eval/runners/run_10m_e2e.py --config configs/eval/eval_matrix.yaml

# ---------------- Dev ----------------

.PHONY: test lint fmt
test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .

fmt:
	$(UV) run ruff format .
