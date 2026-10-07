"""Fake LM Studio server for offline tests.

Purpose: OpenAI-compatible /models, /chat/completions (plain, streaming, image_url) and /embeddings.
Inputs: none (canned replies). Outputs: HTTP on an ephemeral port.
Data flow: tests start it with start_fake_server() and point config lmstudio.base_url at it.
How to change: edit CANNED_* or the embedding function. Image requests are recorded in server.image_requests.
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CANNED_CHAT = "Canned answer. Verify: check the real code."
CANNED_IMAGE = "Canned figure description: a chart with labelled axes. Values not estimated."
MODELS = ["gemma-4-26b-a4b", "text-embedding-nomic-embed-text-v1.5"]


def fake_embedding(text: str, dim: int = 16) -> list[float]:
    """Deterministic pseudo-embedding from a hash, so tests are repeatable."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return [(b - 128) / 128 for b in h[:dim]]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence request logging
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._json({"data": [{"id": m} for m in MODELS]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path.endswith("/embeddings"):
            inputs = req["input"] if isinstance(req["input"], list) else [req["input"]]
            self._json({"data": [{"index": i, "embedding": fake_embedding(t)} for i, t in enumerate(inputs)]})
        elif self.path.endswith("/chat/completions"):
            msgs = req.get("messages", [])
            script = getattr(self.server, "script", None)   # optional callable(req) -> text for tests
            has_image = any(isinstance(m.get("content"), list) and
                            any(p.get("type") == "image_url" for p in m["content"]) for m in msgs)
            if has_image:
                self.server.image_requests.append(req)
                text = CANNED_IMAGE
            elif script:
                text = script(req)
            else:
                text = CANNED_CHAT
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for word in text.split(" "):
                    chunk = {"choices": [{"delta": {"content": word + " "}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self._json({"choices": [{"message": {"role": "assistant", "content": text}}],
                            "usage": {"completion_tokens": len(text.split())}})
        else:
            self._json({"error": "not found"}, 404)


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):   # clients that time out on purpose cause BrokenPipe: ignore
        pass


def start_fake_server():
    """Start on an ephemeral port in a daemon thread. Returns (server, base_url)."""
    srv = QuietServer(("127.0.0.1", 0), Handler)
    srv.image_requests = []
    srv.script = None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
