"""Minimal HTTP client for vLLM's OpenAI-compatible API.

Used by run_via_vllm.py (NIAH/RULER eval) and any future scripts that need
to hit a running vLLM server. Stays small on purpose — no streaming, no
retries, no fancy features. Caller scripts wrap retries if they need them.
"""
from __future__ import annotations

from typing import Any

import httpx


class VllmClient:
    def __init__(self, base_url: str = "http://localhost:8001", model: str = "Qwen/Qwen3.5-4B",
                 timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def complete(self, prompt: str, max_new_tokens: int = 128,
                 stop: list[str] | None = None) -> str:
        """Greedy completion from a raw prompt string."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        if stop:
            payload["stop"] = stop
        resp = self._client.post(f"{self.base_url}/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"]

    def complete_chat(self, messages: list[dict[str, str]], max_new_tokens: int = 128,
                       enable_thinking: bool = False,
                       stop: list[str] | None = None) -> str:
        """Greedy chat completion (applies the model's chat template server-side)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if stop:
            payload["stop"] = stop
        resp = self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._client.close()
