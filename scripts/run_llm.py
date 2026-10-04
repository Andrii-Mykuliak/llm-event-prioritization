"""Run the pipeline for ONE LLM:  python scripts/run_llm.py --llm configs/llms/qwen3-8b.yaml"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
