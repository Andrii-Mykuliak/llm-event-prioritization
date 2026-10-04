"""LLM configuration: one YAML file describes one language model.

Minimal example (configs/llm_qwen3-8b.yaml):

    name: qwen3-8b            # results folder name and identifier in tables
    short_label: 8B           # label in figures
    backend: ollama           # ollama | openai_compatible | mock
    model: qwen3:8b

Everything else defaults to the frozen inference settings (temperature 0, seed 42, context 4096, 48 output tokens,
thinking off). Any deviation from the frozen settings is allowed but is recorded in run_meta.json so that a reader
can see it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from paths import MAIN_EXPERIMENT_CONFIG

BACKENDS = ("ollama", "openai_compatible", "mock")


@dataclass
class LLMConfig:
    name: str
    backend: str
    model: str
    short_label: str = ""
    display_name: str = ""
    size_order: float | None = None
    host: str = "http://localhost:11434"          # ollama host, or base URL of the OpenAI-compatible server
    api_key_env: str | None = None                 # name of the environment variable holding the API key
    think: bool = False                            # ollama: pass think=<value>; thinking must stay off for the frozen design
    options: dict = field(default_factory=dict)    # temperature, seed, num_ctx, num_predict (defaults: frozen settings)
    extra_body: dict = field(default_factory=dict) # extra fields merged into the request body (OpenAI-compatible servers)
    timeout_seconds: int = 300
    retries: int = 2                               # extra attempts after a failed request
    require_no_think_template: bool = True         # ollama: refuse thinking-only models when thinking is off
    mock_mode: str = "uniform"                     # mock backend: uniform | fastest | last
    deviations_from_frozen: dict = field(default_factory=dict)

    def label(self) -> str:
        return self.short_label or self.name


def frozen_inference() -> dict:
    cfg = yaml.safe_load(MAIN_EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    return cfg["inference"]


def _guess_size(text: str) -> tuple[str, float | None]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", text.lower().replace("-", " ").replace("_", " ").replace(":", " "))
    if m:
        return f"{m.group(1)}B", float(m.group(1))
    return "", None


def load_llm_config(source: str | Path | dict) -> LLMConfig:
    raw = dict(source) if isinstance(source, dict) else yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    for key in ("name", "backend", "model"):
        if key not in raw:
            raise ValueError(f"LLM config needs '{key}'")
    if raw["backend"] not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    frozen = frozen_inference()
    options = dict(frozen["options"])
    deviations = {}
    for k, v in (raw.get("options") or {}).items():
        if options.get(k) != v:
            deviations[f"options.{k}"] = {"frozen": options.get(k), "used": v}
        options[k] = v
    think = bool(raw.get("think", frozen["think"]))
    if think != bool(frozen["think"]):
        deviations["think"] = {"frozen": frozen["think"], "used": think}
    label, order = _guess_size(raw["name"] + " " + raw["model"])
    cfg = LLMConfig(
        name=str(raw["name"]), backend=raw["backend"], model=str(raw["model"]),
        short_label=str(raw.get("short_label") or label or raw["name"]),
        display_name=str(raw.get("display_name") or raw["name"]),
        size_order=raw.get("size_order", order),
        host=str(raw.get("host") or raw.get("base_url") or ("http://localhost:11434" if raw["backend"] == "ollama" else "")),
        api_key_env=raw.get("api_key_env"), think=think, options=options,
        extra_body=dict(raw.get("extra_body") or {}), timeout_seconds=int(raw.get("timeout_seconds", frozen.get("timeout_seconds", 300))),
        retries=int(raw.get("retries", 2)), require_no_think_template=bool(raw.get("require_no_think_template", True)),
        mock_mode=str(raw.get("mock_mode", "uniform")), deviations_from_frozen=deviations,
    )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cfg.name):
        raise ValueError("name may contain only letters, digits, '.', '_' and '-' (it becomes a folder name)")
    return cfg
