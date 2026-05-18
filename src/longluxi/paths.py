"""Centralized project paths. Everything else imports from here so we have one source of truth."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- code / config ----
CONFIGS_DIR = REPO_ROOT / "configs"
YARN_CONFIGS_DIR = CONFIGS_DIR / "yarn"
TRAINING_CONFIGS_DIR = CONFIGS_DIR / "training"
RETRIEVAL_CONFIGS_DIR = CONFIGS_DIR / "retrieval"
EVAL_CONFIGS_DIR = CONFIGS_DIR / "eval"

# ---- data ----
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
HF_CACHE_DIR = DATA_DIR / "hf_cache"

# ---- artifacts ----
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RUNS_DIR = REPO_ROOT / "runs"
LOGS_DIR = REPO_ROOT / "logs"

# ---- retrieval ----
RETRIEVAL_INDICES_DIR = REPO_ROOT / "retrieval" / "indices"

# ---- eval ----
EVAL_OUTPUTS_DIR = REPO_ROOT / "eval" / "_outputs"

# ---- external repos (flame, FLA) ----
EXTERNAL_DIR = REPO_ROOT / "external"
FLAME_DIR = EXTERNAL_DIR / "flame"


def ensure_dirs() -> None:
    """Create runtime dirs that are .gitignored but needed at runtime."""
    for p in (RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, HF_CACHE_DIR,
             CHECKPOINTS_DIR, RUNS_DIR, LOGS_DIR, RETRIEVAL_INDICES_DIR, EVAL_OUTPUTS_DIR):
        p.mkdir(parents=True, exist_ok=True)
