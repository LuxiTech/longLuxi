# Training

CPT (Stage A/B/C) + SFT (Stage D/E)。

主框架：[`fla-org/flame`](https://github.com/fla-org/flame)（torchtitan + FLA，对 Qwen3.5 的 GDN 友好）。
后备：Megatron-Core / NeMo（若 flame >1M 不稳）。

## 启动

```bash
# 0. 首次：拉 flame
make setup-flame

# 1. 1M smoke
bash training/scripts/run_stage_a.sh --smoke

# 2. 1M main
bash training/scripts/run_stage_a.sh

# 3. 2M
bash training/scripts/run_stage_b.sh

# 4. 4M (optional)
bash training/scripts/run_stage_c.sh

# 5. short SFT
bash training/scripts/run_short_sft.sh

# 6. memory SFT
bash training/scripts/run_memory_sft.sh
```

## flame 配置映射

`configs/training/stage_*.toml` 的字段会通过 `training/flame_wrapper/` 里的脚本转成 flame 的 CLI 参数。

关键映射：
- `parallelism.context_parallel_degree` → `--experimental.context_parallel_degree`
- `model.attn_impl` → flame `--model.attn_impl flash_attention_2`（standard 层；GDN 层自动 FLA kernel）
- `training.context_length` → `--training.seq_len`
- `training.global_batch_tokens` 通过 `gradient_accumulation` 与 `dp_degree` 反推

## 多节点启动

flame / torchtitan 走 `torchrun --nproc-per-node=8 --nnodes=2 --node-rank={0,1} --master-addr=<head>`，
或在 `slurm` 下用 `srun`。两种启动方式的 wrapper 都在 `training/flame_wrapper/`。

## checkpoint 约定

- 路径：`checkpoints/<stage>/<step or "latest">/`
- 内容：HF-format model + tokenizer + training_state.pt + config.json
- async save 避免阻塞 1M+ step
