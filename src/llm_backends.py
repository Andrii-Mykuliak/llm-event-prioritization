"""Language-model backends. Every backend answers one prompt with one text; nothing else differs between LLMs.

    ollama              local Ollama server (/api/chat)
    openai_compatible   any server that speaks the OpenAI chat-completions protocol (OpenAI, vLLM, llama.cpp server,
                        LM Studio, Together, ...), via /chat/completions
    mock                deterministic fake answers, ONLY for testing the pipeline; its outputs are not scientific data
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime, timezone

import requests

from llm_config import LLMConfig


class Backend:
    mock = False

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def check(self) -> dict:
        """Verify that the model is reachable; return environment metadata stored with every response."""
        raise NotImplementedError

    def ask(self, prompt: str) -> dict:
        """Return {text, latency_s, prompt_tokens, output_tokens}."""
        raise NotImplementedError


class OllamaBackend(Backend):
    def check(self) -> dict:
        host = self.cfg.host.rstrip("/")
        v = requests.get(f"{host}/api/version", timeout=10).json()
        tags = requests.get(f"{host}/api/tags", timeout=10).json().get("models", [])
        info = next((m for m in tags if m["name"] == self.cfg.model), None)
        if info is None:
            raise RuntimeError(f"Model {self.cfg.model} is not available on {host}. Run: ollama pull {self.cfg.model}")
        show = requests.post(f"{host}/api/show", json={"model": self.cfg.model}, timeout=30).json()
        template = show.get("template", "")
        thinking_capable = "thinking" in (show.get("capabilities") or []) or "<think>" in template
        if not self.cfg.think and self.cfg.require_no_think_template and thinking_capable and "/no_think" not in template:
            raise RuntimeError(
                f"{self.cfg.model}: the chat template has no /no_think switch. Thinking-only models ignore think=false "
                "and reason inside the answer; use a hybrid model or set require_no_think_template: false in the LLM config "
                "if your model never emits reasoning text.")
        return {"backend": "ollama", "server_version": v.get("version"), "model_digest": info.get("digest"),
                "model_size": info.get("size"), "thinking_capable": thinking_capable, "quantization": info.get("details", {}).get("quantization_level"),
                "template_supports_no_think": "/no_think" in template}

    def ask(self, prompt: str) -> dict:
        body = {"model": self.cfg.model, "messages": [{"role": "user", "content": prompt}], "stream": False,
                "think": self.cfg.think, "options": self.cfg.options}
        t0 = time.perf_counter()
        r = requests.post(f"{self.cfg.host.rstrip('/')}/api/chat", json=body, timeout=self.cfg.timeout_seconds)
        r.raise_for_status()
        d = r.json()
        return {"text": d["message"]["content"], "latency_s": round(time.perf_counter() - t0, 3),
                "prompt_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count")}


class OpenAICompatibleBackend(Backend):
    def _headers(self) -> dict:
        key = os.environ.get(self.cfg.api_key_env, "") if self.cfg.api_key_env else ""
        return {"Authorization": f"Bearer {key}"} if key else {}

    def check(self) -> dict:
        base = self.cfg.host.rstrip("/")
        if not base:
            raise RuntimeError("openai_compatible backend needs 'host' (the base URL, for example http://localhost:8000/v1)")
        env = {"backend": "openai_compatible", "server_version": None, "model_digest": None, "model_size": None,
               "quantization": None, "base_url": base}
        try:   # not every server lists models; this is only informative
            r = requests.get(f"{base}/models", headers=self._headers(), timeout=15)
            if r.ok:
                ids = [m.get("id") for m in r.json().get("data", [])]
                env["model_listed"] = self.cfg.model in ids
        except requests.RequestException:
            pass
        return env

    def ask(self, prompt: str) -> dict:
        o = self.cfg.options
        body = {"model": self.cfg.model, "messages": [{"role": "user", "content": prompt}], "stream": False,
                "temperature": o.get("temperature", 0), "max_tokens": o.get("num_predict", 48)}
        if o.get("seed") is not None:
            body["seed"] = o["seed"]
        body.update(self.cfg.extra_body)
        t0 = time.perf_counter()
        r = requests.post(f"{self.cfg.host.rstrip('/')}/chat/completions", json=body, headers=self._headers(), timeout=self.cfg.timeout_seconds)
        r.raise_for_status()
        d = r.json()
        usage = d.get("usage") or {}
        return {"text": d["choices"][0]["message"].get("content") or "", "latency_s": round(time.perf_counter() - t0, 3),
                "prompt_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}


class MockBackend(Backend):
    """Fake answers for tests and demonstrations. NEVER use these results as evidence about a language model."""
    mock = True

    def check(self) -> dict:
        return {"backend": "mock", "server_version": "mock", "model_digest": "mock", "model_size": None, "quantization": None, "mock": True}

    def ask(self, prompt: str) -> dict:
        rows = [json.loads(l) for l in prompt.splitlines() if l.startswith("{")]
        ids = [r["id"] for r in rows]
        mode = self.cfg.mock_mode
        if mode == "fastest":      # picks the lowest displayed times first, pit events last
            key = lambda r: r.get("sector_time_s", r.get("lap_time_s", 1e9))
            chosen = [r["id"] for r in sorted(rows, key=key)[:3]]
        elif mode == "last":       # picks the last three listed events
            chosen = ids[-3:][::-1]
        else:                      # uniform random choice (a null model)
            rng = random.Random(f"{self.cfg.name}|{prompt}")
            chosen = rng.sample(ids, 3)
        return {"text": ",".join(chosen), "latency_s": 0.0, "prompt_tokens": None, "output_tokens": None}


def make_backend(cfg: LLMConfig) -> Backend:
    return {"ollama": OllamaBackend, "openai_compatible": OpenAICompatibleBackend, "mock": MockBackend}[cfg.backend](cfg)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
