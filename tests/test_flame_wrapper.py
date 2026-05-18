"""TOML -> flame CLI args wrapper tests. Pure-Python, no GPU."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "training/flame_wrapper/toml_to_flame_args.py"
TOML = REPO_ROOT / "configs/training/stage_a_1m_cpt.toml"


def test_wrapper_emits_expected_cli_flags_for_smoke():
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--toml", str(TOML), "--phase", "smoke"],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr
    args = res.stdout.strip().split()
    # check critical flags appear
    assert any("--training.seq_len" in a for a in args)
    seq_len = next(a for a in args if a.startswith("--training.seq_len="))
    assert seq_len.split("=", 1)[1] == "262144"  # smoke phase override
    assert any("--experimental.context_parallel_degree=4" in a for a in args)
    assert any("--training.dtype=bfloat16" in a for a in args)
    # smoke phase budget
    assert any(("--training.max_steps" in a) or ("--training.steps" in a) for a in args)


def test_wrapper_emits_main_seq_len_when_phase_main():
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--toml", str(TOML), "--phase", "main"],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr
    seq_len = next(a for a in res.stdout.split() if a.startswith("--training.seq_len="))
    assert seq_len.split("=", 1)[1] == "1048576"  # main 1M
