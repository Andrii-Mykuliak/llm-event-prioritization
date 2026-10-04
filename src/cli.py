"""Command line: run one LLM, analyse runs, check integrity.

    python scripts/run_llm.py --llm configs/llm_qwen3-8b.yaml [--parts main_experiment controlled_timing_rank_followup] [--limit N]
    python scripts/analyze.py --tag my_run [--results results] [--with-reference] [--runs NAME ...]
    python scripts/analyze.py --reference
    python scripts/integrity.py [--results results]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from paths import INPUTS, OUTPUTS, REFERENCE, RESULTS


def select_runs(results: Path, with_reference: bool, only_reference: bool, names: list[str] | None, include_smoke: bool, include_mock: bool):
    from an_data import discover_runs

    runs = []
    if not only_reference:
        runs += discover_runs(results, include_smoke=include_smoke, include_mock=include_mock)
    if with_reference or only_reference:
        taken = {r.name for r in runs}
        for r in discover_runs(REFERENCE, reference_roots=(REFERENCE,)):
            if r.name in taken:
                r = dataclasses.replace(r, name=r.name + "__reference", label=r.label + " (ref)")
            runs.append(r)
    if names:
        runs = [r for r in runs if r.name in names]
    runs.sort(key=lambda r: (r.order is None, r.order if r.order is not None else 0, r.name))
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llm pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline for ONE LLM")
    r.add_argument("--llm", required=True, help="path to an LLM config yaml (configs/llm_*.yaml)")
    r.add_argument("--parts", nargs="+", default=["main_experiment", "controlled_timing_rank_followup"], choices=["main_experiment", "controlled_timing_rank_followup"])
    r.add_argument("--limit", type=int, default=None, help="only the first N trials per part; results go to <name>__smoke")
    r.add_argument("--inputs", default=str(INPUTS))
    r.add_argument("--results", default=str(RESULTS))
    a = sub.add_parser("analyze", help="tables and figures for the selected runs")
    a.add_argument("--out", default=str(OUTPUTS), help="folder for the flat output files (default: outputs/)")
    a.add_argument("--tag", default=None, help="prefix of every output file: <tag>__<name> (default: reference / latest)")
    a.add_argument("--results", default=str(RESULTS))
    a.add_argument("--reference", action="store_true", help="analyse only the shipped reference runs")
    a.add_argument("--with-reference", action="store_true", help="add the shipped reference runs to your runs")
    a.add_argument("--runs", nargs="*", default=None)
    a.add_argument("--include-smoke", action="store_true")
    a.add_argument("--include-mock", action="store_true")
    a.add_argument("--no-extras", action="store_true")
    a.add_argument("--inputs", default=str(INPUTS))
    i = sub.add_parser("integrity", help="check runs against the inputs")
    i.add_argument("--results", default=str(RESULTS))
    i.add_argument("--with-reference", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        from run import run_llm

        d = run_llm(args.llm, parts=tuple(args.parts), inputs_dir=Path(args.inputs), results_root=Path(args.results), limit=args.limit)
        print("results folder:", d)
        return 0
    if args.cmd == "analyze":
        from an_report import build_report

        runs = select_runs(Path(args.results), args.with_reference, args.reference, args.runs, args.include_smoke, args.include_mock)
        if not runs:
            print("no runs found; run an LLM first or pass --reference", file=sys.stderr)
            return 2
        tag = args.tag or ("reference" if args.reference else "latest")
        build_report(runs, Path(args.out), tag=tag, inputs=Path(args.inputs), extras=not args.no_extras)
        return 0
    if args.cmd == "integrity":
        from an_integrity import check_runs

        runs = select_runs(Path(args.results), args.with_reference, False, None, False, False)
        t = check_runs(runs)
        print(t.to_string(index=False))
        return 0 if t.ok.all() else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
