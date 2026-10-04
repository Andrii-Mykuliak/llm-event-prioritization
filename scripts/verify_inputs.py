"""Run the 16 + 28 structural checks on an input folder (default: the shipped inputs)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from si_cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:] or ["--inputs", str(ROOT / "inputs")]
    raise SystemExit(main(["verify", *args]))
