"""Reader serving wrapper.

Default backend: vLLM (supports YaRN via --rope-scaling).
Alternative: SGLang.

启动后暴露 OpenAI-compatible 接口在 --port，router 调它。
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader-ckpt", required=True)
    parser.add_argument("--backend", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-model-len", type=int, default=2097152)
    parser.add_argument("--yarn-config", default="configs/yarn/yarn_2m.json")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    args = parser.parse_args()
    # TODO[W11]: shell out to `vllm serve ...` or `sglang.launch_server ...` with proper YaRN config.
    raise NotImplementedError("W11 task: implement reader serving wrapper")


if __name__ == "__main__":
    main()
