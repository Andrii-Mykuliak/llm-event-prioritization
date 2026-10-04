"""Integrity checks of runs against the inputs:  python scripts/integrity.py [--results results] [--with-reference]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["integrity", *sys.argv[1:]]))
