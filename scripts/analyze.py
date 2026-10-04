"""Tables and figures:  python scripts/analyze.py --with-reference --tag my_run   |   --reference"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["analyze", *sys.argv[1:]]))
