"""LM Studio client (OpenAI-compatible API) using only the standard library.

Purpose: list models, chat (optionally streaming), embeddings, image requests.
Inputs: config dict. Outputs: strings / lists of floats.
Data flow: used by rag, embed, vision and the check command.
How to change: endpoints are standard OpenAI paths under lmstudio.base_url.
No cloud calls: base_url should point to localhost.
"""
from __future__ import annotations

import base64
import json
import urllib.request
from typing import Any, Iterator


class LMClient:
    """Thin client. One instance per config; holds no hidden global state."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        lm = cfg["lmstudio"]
        self.base = lm["base_url"].rstrip("/")
        self.timeout = float(lm.get("timeout_s", 120))                # light calls: models list, embeddings
        self.chat_timeout = float(lm.get("chat_timeout_s", 900))      # chat/vision: a local 26B model can be slow
        self.chat_model = lm["chat_model"]
        self.embed_model = lm["embedding_model"]
        self.temperature = float(lm.get("temperature", 0.1))
        self.last_usage: dict[str, Any] = {}      # usage block of the last non-streaming reply

    def _post(self, path: str, payload: dict[str, Any], stream: bool = False, timeout: float | None = None):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def list_models(self) -> list[str]:
        """Return model ids reported by the server."""
        with urllib.request.urlopen(self.base + "/models", timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m["id"] for m in data.get("data", [])]

    def chat(self, messages: list[dict[str, Any]], model: str | None = None) -> str:
        """Non-streaming chat; returns the assistant text."""
        payload = {"model": model or self.chat_model, "messages": messages,
                   "temperature": self.temperature, "stream": False}
        with self._post("/chat/completions", payload, timeout=self.chat_timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        self.last_usage = data.get("usage", {}) or {}
        return data["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict[str, Any]], model: str | None = None) -> Iterator[str]:
        """Streaming chat (server-sent events); yields text deltas."""
        payload = {"model": model or self.chat_model, "messages": messages,
                   "temperature": self.temperature, "stream": True}
        with self._post("/chat/completions", payload, stream=True, timeout=self.chat_timeout) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                delta = json.loads(body)["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts. The caller adds the nomic prefixes (search_document: / search_query: )."""
        payload = {"model": self.embed_model, "input": texts}
        with self._post("/embeddings", payload) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]

    def chat_with_image(self, prompt: str, png_bytes: bytes, model: str | None = None) -> str:
        """Send one image (PNG bytes, as base64 data URI) with a prompt. Output is UNVERIFIED."""
        uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": uri}},
        ]}]
        return self.chat(messages, model=model)
