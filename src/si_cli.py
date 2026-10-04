"""
Command line interface of the input builder.

    python scripts/make_inputs.py   --out inputs_rebuilt --from-primitive inputs
    python scripts/make_inputs.py   --out inputs_rebuilt --download --cache .fastf1_cache
    python scripts/verify_inputs.py --inputs inputs

`build` creates the flat set of input files under --out (dataset, main-experiment and controlled timing-rank follow-up files), verifies them and (with --compare)
tells whether they are byte-identical to a reference input set such as the shipped `inputs/`.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path

import si_build_dataset as build_dataset
import si_make_trials as make_trials
from si_controlled_timing_rank_followup import build_followup
from si_verify import verify_main_experiment, verify_followup

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "configs"
EXTRACTOR_CFG = FROZEN / "extractor.yaml"
MAIN_EXPERIMENT_CFG = FROZEN / "main_experiment.yaml"
FOLLOWUP_CFG = FROZEN / "controlled_timing_rank_followup.yaml"

MANIFEST_FILES = ["primitive_events.jsonl", "seed_windows.jsonl", "dataset_manifest.json", "dataset_summary.csv",
                  "main_experiment_trials.jsonl", "controlled_timing_rank_followup_trials.jsonl", "controlled_timing_rank_followup_exclusions.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(inputs: Path) -> Path:
    lines = [f"{sha256(inputs / rel)}  {rel}" for rel in MANIFEST_FILES if (inputs / rel).exists()]
    for name, p in (("configs/extractor.yaml", EXTRACTOR_CFG), ("configs/main_experiment.yaml", MAIN_EXPERIMENT_CFG),
                    ("configs/controlled_timing_rank_followup.yaml", FOLLOWUP_CFG)):
        lines.append(f"{sha256(p)}  {name}")
    out = inputs / "MANIFEST.sha256"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def read_manifest(path: Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, rel = line.split(None, 1)
            out[rel.strip()] = h
    return out


def compare_inputs(a: Path, b: Path) -> list[tuple[str, bool]]:
    """Compare two input directories file by file (byte-identical or not)."""
    res = []
    for rel in MANIFEST_FILES:
        if (a / rel).exists() and (b / rel).exists():
            res.append((rel, sha256(a / rel) == sha256(b / rel)))
        else:
            res.append((rel, False))
    return res


def verify_inputs(inputs: Path, expected_total: int | None = 1040, quiet: bool = False) -> bool:
    inputs = Path(inputs)
    ok = True
    r1 = verify_main_experiment(inputs / "main_experiment_trials.jsonl", inputs, expected_total)
    r2 = verify_followup(FOLLOWUP_CFG, inputs / "controlled_timing_rank_followup_trials.jsonl", inputs / "main_experiment_trials.jsonl", inputs, MAIN_EXPERIMENT_CFG)
    for r in (r1, r2):
        ok &= r.ok
        if not quiet:
            print(r.summary())
        (inputs / ("VERIFY_" + r.name.lower().removesuffix(" trials").replace("follow-up", "followup").replace("-", "_").replace(" ", "_") + ".txt")).write_text(r.report, encoding="utf-8")
    man = inputs / "MANIFEST.sha256"
    if man.exists():
        want = read_manifest(man)
        for rel, h in want.items():
            p = ROOT / rel if rel.startswith("configs/") else inputs / rel
            if p.exists() and sha256(p) != h:
                ok = False
                if not quiet:
                    print(f"HASH MISMATCH: {rel}")
        if not quiet:
            print("manifest hashes checked:", len(want))
    return ok


def cmd_build(a: argparse.Namespace) -> int:
    out = Path(a.out).resolve()
    ds = out
    ds.mkdir(parents=True, exist_ok=True)
    if a.from_primitive:
        src = Path(a.from_primitive).resolve()
        for name in ("primitive_events.jsonl", "dataset_manifest.json"):
            if src.resolve() != ds.resolve():
                shutil.copy(src / name, ds / name)
        argv = ["--config", str(EXTRACTOR_CFG), "--output", str(ds), "--stage", "main", "--from-primitive", "--cache", str(out / "_cache_unused")]
    else:
        argv = ["--config", str(EXTRACTOR_CFG), "--output", str(ds), "--stage", "main", "--cache", str(Path(a.cache).resolve())]
    print("[1/4] building the dataset (windows, candidates, seeds) ...")
    build_dataset.main(argv)
    for handler in logging.root.handlers[:]:   # release build.log so that it can be removed on Windows
        handler.close()
        logging.root.removeHandler(handler)
    shutil.rmtree(out / "_cache_unused", ignore_errors=True)
    if not a.keep_candidates:
        (ds / "candidate_windows.jsonl").unlink(missing_ok=True)
        (ds / "build.log").unlink(missing_ok=True)
    print("[2/4] building main-experiment trials ...")
    make_trials.main(["--dataset", str(ds), "--config", str(MAIN_EXPERIMENT_CFG), "--out", str(out / "main_experiment_trials.jsonl")])
    print("[3/4] building controlled timing-rank follow-up trials ...")
    info = build_followup(ds, MAIN_EXPERIMENT_CFG, FOLLOWUP_CFG, out, allow_source_mismatch=a.allow_different_source)
    print("      ", info)
    print("[4/4] verifying ...")
    ok = verify_inputs(out, expected_total=None if a.allow_different_source else 1040)
    write_manifest(out)
    if a.compare:
        print("comparison with", a.compare)
        same = compare_inputs(out, Path(a.compare).resolve())
        for rel, eq in same:
            print(f"  {'IDENTICAL' if eq else 'DIFFERENT '}  {rel}")
        ok &= all(eq for _, eq in same)
    print("RESULT:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="make_inputs.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the input data")
    b.add_argument("--out", required=True)
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-primitive", help="folder with primitive_events.jsonl and dataset_manifest.json (for example inputs/); no download")
    g.add_argument("--download", action="store_true", help="download the races with FastF1 (needs network)")
    b.add_argument("--cache", default=str(ROOT / "cache" / "fastf1"), help="FastF1 cache directory (download mode)")
    b.add_argument("--compare", help="reference input directory to compare byte-by-byte (for example inputs/)")
    b.add_argument("--keep-candidates", action="store_true", help="keep candidate_windows.jsonl and build.log")
    b.add_argument("--allow-different-source", action="store_true",
                   help="do not require the frozen seed-window hash (use for a deliberately different dataset)")
    v = sub.add_parser("verify", help="verify an input directory")
    v.add_argument("--inputs", default=str(ROOT / "inputs"))
    v.add_argument("--any-size", action="store_true", help="do not require the frozen trial count")
    m = sub.add_parser("manifest", help="write MANIFEST.sha256 for an input directory")
    m.add_argument("--inputs", default=str(ROOT / "inputs"))
    a = ap.parse_args(argv)
    if a.cmd == "build":
        return cmd_build(a)
    if a.cmd == "verify":
        return 0 if verify_inputs(Path(a.inputs), None if a.any_size else 1040) else 1
    print(write_manifest(Path(a.inputs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
