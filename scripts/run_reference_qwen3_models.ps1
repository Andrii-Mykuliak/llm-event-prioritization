# The three reference LLMs, one separate pipeline run each (resumable). Needs `ollama serve` and the models pulled.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
foreach ($cfg in "qwen3-4b-q4_K_M", "qwen3-8b", "qwen3-14b") {
    python scripts/run_llm.py --llm "configs/llm_$cfg.yaml"
    if ($LASTEXITCODE -ne 0) { throw "run failed for $cfg" }
}
python scripts/analyze.py --tag my_run
