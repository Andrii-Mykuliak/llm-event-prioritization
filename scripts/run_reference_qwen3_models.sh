#!/usr/bin/env bash
# The three reference LLMs, one separate pipeline run each (resumable; ~hours on a GPU). Needs `ollama serve` and the models pulled.
set -e
cd "$(dirname "$0")/.."
for cfg in qwen3-4b-q4_K_M qwen3-8b qwen3-14b; do
  python scripts/run_llm.py --llm "configs/llm_${cfg}.yaml"
done
python scripts/analyze.py --tag my_run
