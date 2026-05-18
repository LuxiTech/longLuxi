"""Model-loader unit tests. Mock the HF config object — no real model load."""
import json
from pathlib import Path
from types import SimpleNamespace

from longluxi.eval.model_loader import apply_yarn_to_config, load_yarn_json


def test_load_yarn_json_returns_rope_params(tmp_path):
    p = tmp_path / "y.json"
    p.write_text(json.dumps({
        "rope_parameters": {
            "rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 262144,
            "mrope_interleaved": True, "mrope_section": [11, 11, 10],
            "rope_theta": 10000000, "partial_rotary_factor": 0.25,
        }
    }))
    rp = load_yarn_json(p)
    assert rp["rope_type"] == "yarn"
    assert rp["factor"] == 4.0


def test_apply_yarn_to_config_with_rope_parameters_attr():
    cfg = SimpleNamespace(rope_parameters={"placeholder": True})
    rp = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}
    apply_yarn_to_config(cfg, rp)
    assert cfg.rope_parameters == rp


def test_apply_yarn_to_config_falls_back_to_rope_scaling():
    cfg = SimpleNamespace()  # no rope_parameters attr
    rp = {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 262144}
    apply_yarn_to_config(cfg, rp)
    assert cfg.rope_scaling["factor"] == 8.0
    assert cfg.rope_scaling["original_max_position_embeddings"] == 262144


def test_repo_yarn_configs_loadable():
    repo_root = Path(__file__).resolve().parent.parent
    for name in ("yarn_1m.json", "yarn_2m.json", "yarn_4m.json"):
        rp = load_yarn_json(repo_root / "configs" / "yarn" / name)
        assert rp["rope_type"] == "yarn"
        assert rp["original_max_position_embeddings"] == 262144
