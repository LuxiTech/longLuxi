"""Convert configs/training/stage_{d,e}_*.toml -> LLaMA-Factory yaml.

LF expects flat yaml with specific keys. Wrapper does:
  TOML (source of truth, our schema)  ->  LF yaml (downstream)

Usage:
    uv run python training/llamafactory_wrapper/toml_to_lf_yaml.py \
        --toml configs/training/stage_d_short_sft.toml \
        --out configs/training/_generated/stage_d_short_sft.lf.yaml

Then launch:
    llamafactory-cli train configs/training/_generated/stage_d_short_sft.lf.yaml
"""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def _stage_template(stage: str) -> dict:
    """LF baseline keys per stage; merged with TOML overrides."""
    if stage == "stage_d":
        return {
            "stage": "sft",
            "do_train": True,
            "cutoff_len": 32768,
            "dataset_dir": "data/processed",
            "template": "qwen",
            "save_strategy": "steps",
            "logging_steps": 10,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "preprocessing_num_workers": 16,
            "report_to": ["wandb"],
        }
    elif stage == "stage_e":
        return {
            "stage": "sft",
            "do_train": True,
            "cutoff_len": 524288,
            "dataset_dir": "data/processed",
            "template": "memory_evidence_qa",   # custom template, register in LF data/template.py
            "save_strategy": "steps",
            "logging_steps": 10,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "preprocessing_num_workers": 16,
            "report_to": ["wandb"],
            # 长上下文必备
            "sequence_parallel_size": 16,        # via DeepSpeed Ulysses
            "neat_packing": True,
        }
    raise ValueError(f"unknown stage: {stage}")


def toml_to_lf(toml_path: Path, out_path: Path) -> None:
    if yaml is None:
        raise RuntimeError("pyyaml missing. `uv sync` first.")
    data = tomllib.loads(toml_path.read_text())
    stage_name = "stage_d" if "stage_d" in toml_path.stem else "stage_e"
    lf = _stage_template(stage_name)

    # ---- model ----
    lf["model_name_or_path"] = data["run"]["resume_from"] or data["model"]["base_id"]
    lf["flash_attn"] = "fa2"
    lf["bf16"] = True

    # ---- finetuning ----
    lf["finetuning_type"] = data["training"]["finetuning_type"]
    if data["training"].get("enable_liger_kernel"):
        lf["enable_liger_kernel"] = True

    # ---- training ----
    t = data["training"]
    lf["cutoff_len"] = t["context_length"]
    lf["per_device_train_batch_size"] = t["micro_batch_size"]
    lf["gradient_accumulation_steps"] = t.get("gradient_accumulation", 1)
    lf["learning_rate"] = t["learning_rate"]
    lf["lr_scheduler_type"] = t.get("lr_scheduler", "cosine").replace("_with_warmup", "")
    lf["max_steps"] = t["max_steps"]
    lf["warmup_steps"] = t.get("warmup_steps", 0)
    lf["weight_decay"] = t.get("weight_decay", 0.01)
    lf["max_grad_norm"] = t.get("max_grad_norm", 1.0)
    lf["gradient_checkpointing"] = data["model"]["gradient_checkpointing"]

    # ---- parallelism ----
    p = data["parallelism"]
    lf["deepspeed"] = f"configs/training/deepspeed_z{p['deepspeed_stage']}.json"
    if p.get("sequence_parallel_size", 1) > 1:
        lf["sequence_parallel_size"] = p["sequence_parallel_size"]

    # ---- checkpoint ----
    lf["output_dir"] = data["run"]["output_dir"]
    lf["save_steps"] = data["checkpoint"]["save_every_steps"]
    lf["save_total_limit"] = data["checkpoint"]["keep_last"]

    # ---- data ----
    lf["dataset"] = Path(data["data"]["mix_file"]).stem   # LF dataset name
    lf["max_samples"] = data["data"]["example_budget"]
    if "template" in data.get("data", {}):
        lf["template"] = data["data"]["template"]

    # ---- eval ----
    e = data.get("eval", {})
    if e.get("eval_every_steps"):
        lf["eval_steps"] = e["eval_every_steps"]
        lf["evaluation_strategy"] = "steps"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("# AUTO-GENERATED from " + str(toml_path) + "\n" + yaml.safe_dump(lf, sort_keys=False, allow_unicode=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toml", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    toml_to_lf(args.toml, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
