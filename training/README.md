# Training

CPT (Stage A/B/C) + SFT (Stage D/E)。**两个框架分工**：

| 阶段 | 框架 | 理由 |
|---|---|---|
| Stage A/B/C — CPT | [`fla-org/flame`](https://github.com/fla-org/flame) (torchtitan + FLA) | GDN/FLA kernel 原生支持；CP 跨节点 IB 走 torchtitan；1M-4M 极长序列 |
| Stage D/E — SFT | [`hiyouga/LLaMA-Factory`](https://github.com/hiyouga/LLaMA-Factory) | HF 生态成熟；数据格式标准化；DeepSpeed Ulysses 序列并行；快速迭代 |

**handoff**：flame 输出 HF-format ckpt → LLaMA-Factory `model_name_or_path` 直接加载。

后备：Megatron-Core (flame fallback at >1M)、TRL `SFTTrainer` (LF fallback at >256K)。

## 启动

```bash
# 0. 首次：拉 flame + LLaMA-Factory
make setup-flame
make setup-llamafactory

# CPT (flame)
bash training/scripts/run_stage_a.sh --smoke   # 1M smoke
bash training/scripts/run_stage_a.sh           # 1M main
bash training/scripts/run_stage_b.sh           # 2M
bash training/scripts/run_stage_c.sh           # 4M (optional)

# SFT (LLaMA-Factory)
bash training/scripts/run_short_sft.sh         # Stage D
bash training/scripts/run_memory_sft.sh        # Stage E
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

## TOML → 框架配置转换

每个 `configs/training/stage_*.toml` 都通过 wrapper 转成对应框架的原生配置：

- `training/flame_wrapper/toml_to_flame_args.py` → flame CLI 参数（Stage A/B/C）
- `training/llamafactory_wrapper/toml_to_lf_yaml.py` → LF yaml（Stage D/E）

转换结果写到 `configs/training/_generated/`（gitignored）。这样 toml 是 source of truth，框架变化只需改 wrapper。
