"""Tests for the vLLM HTTP client. No GPU; uses a stub HTTP server."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from eval.runners._vllm_client import VllmClient


class _StubHandler(BaseHTTPRequestHandler):
    captured: list[dict] = []
    response_text = "42"

    def log_message(self, *a, **kw):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n))
        type(self).captured.append({"path": self.path, "body": body})
        # Stub returns both the /completions shape (choices[].text) and the
        # /chat/completions shape (choices[].message.content) so the same handler
        # serves both endpoints.
        payload = {
            "choices": [{
                "text": type(self).response_text,
                "message": {"role": "assistant", "content": type(self).response_text},
                "finish_reason": "stop",
            }],
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def stub_server():
    _StubHandler.captured = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = HTTPServer(("127.0.0.1", port), _StubHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{port}", _StubHandler
    finally:
        server.shutdown()


def test_complete_sends_greedy_request(stub_server):
    base_url, handler = stub_server
    client = VllmClient(base_url=base_url, model="Qwen/Qwen3.5-4B")
    out = client.complete("hello", max_new_tokens=8)
    assert out == "42"
    assert len(handler.captured) == 1
    body = handler.captured[0]["body"]
    assert body["model"] == "Qwen/Qwen3.5-4B"
    assert body["prompt"] == "hello"
    assert body["max_tokens"] == 8
    assert body["temperature"] == 0.0


def test_complete_uses_chat_template_path_when_messages_given(stub_server):
    base_url, handler = stub_server
    handler.response_text = "ok"
    client = VllmClient(base_url=base_url, model="Qwen/Qwen3.5-4B")
    out = client.complete_chat(
        [{"role": "user", "content": "what is 6*7?"}],
        max_new_tokens=4,
        enable_thinking=False,
    )
    assert out == "ok"
    body = handler.captured[0]["body"]
    assert "messages" in body
    assert body["messages"][0]["content"] == "what is 6*7?"
    assert body["max_tokens"] == 4
    # vLLM-specific: thinking flag passed as chat_template_kwargs
    assert body.get("chat_template_kwargs", {}).get("enable_thinking") is False
