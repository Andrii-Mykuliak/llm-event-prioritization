"""Create the input data from scratch (downloads FastF1 data) or from a primitive-events folder.

    python scripts/make_inputs.py --out inputs_rebuilt --download --cache .fastf1_cache --compare inputs
    python scripts/make_inputs.py --out inputs_rebuilt --from-primitive inputs --compare inputs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from si_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["build", *sys.argv[1:]]))
