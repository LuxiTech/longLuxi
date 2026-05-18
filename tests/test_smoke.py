"""Smoke tests — must pass without GPUs or heavy deps."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_yarn_configs_valid_json():
    for name in ("yarn_1m.json", "yarn_2m.json", "yarn_4m.json"):
        path = REPO_ROOT / "configs" / "yarn" / name
        data = json.loads(path.read_text())
        assert "rope_parameters" in data
        rp = data["rope_parameters"]
        assert rp["rope_type"] == "yarn"
        assert rp["original_max_position_embeddings"] == 262144
        assert rp["factor"] in (4.0, 8.0, 16.0)


def test_training_configs_present():
    for stage in ("stage_a_1m_cpt", "stage_b_2m_cpt", "stage_c_4m_cpt",
                  "stage_d_short_sft", "stage_e_memory_sft"):
        path = REPO_ROOT / "configs" / "training" / f"{stage}.toml"
        assert path.exists(), f"missing {path}"
        text = path.read_text()
        assert "[run]" in text
        assert "[training]" in text


def test_retrieval_configs_present():
    for name in ("chunking.yaml", "index.yaml", "pipeline.yaml"):
        assert (REPO_ROOT / "configs" / "retrieval" / name).exists()


def test_paths_module():
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from longluxi import paths
    assert paths.REPO_ROOT == REPO_ROOT
    assert paths.YARN_CONFIGS_DIR.exists()


def test_spec_doc_present():
    spec = REPO_ROOT / "docs" / "superpowers" / "specs" / "2026-05-18-qwen3.5-4b-10m-context-design.md"
    assert spec.exists()
    text = spec.read_text()
    assert "Qwen3.5-4B" in text
    assert "flame" in text
    assert "2M-4M" in text


def test_baseline_eval_dry_run():
    """baseline_eval.py --dry-run should print a plan and exit 0 without heavy deps."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "baseline_eval.py"),
         "--task", "niah", "--max-len", "131072", "--limit", "1", "--dry-run"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "plan" in result.stdout
