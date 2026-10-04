"""Integrity checks of a set of runs against the shipped inputs. A failed check is reported, never silently repaired."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from paths import FROZEN, INPUTS, EXPERIMENT_PARTS, ROOT
from an_data import RunInfo, read_jsonl

EXPECTED = {"main_experiment": 1040, "controlled_timing_rank_followup": 252}   # frozen identifiers
PUBLIC_NAME = {"main_experiment": "main experiment", "controlled_timing_rank_followup": "controlled timing-rank follow-up"}


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_runs(runs: list[RunInfo], inputs: Path = INPUTS) -> pd.DataFrame:
    rows = []

    def add(run, part, check, ok, detail=""):
        rows.append({"run": run, "experiment_part": part, "check": check, "ok": bool(ok), "detail": detail})

    for run in runs:
        for part, (trial_file, resp_file) in EXPERIMENT_PARTS.items():
            path = run.file(part)
            if not path.exists():
                continue
            trials = {t["trial_id"]: t for t in read_jsonl(Path(inputs) / trial_file)}
            resp = read_jsonl(path)
            ids = [r["trial_id"] for r in resp]
            dup = [k for k, c in Counter(ids).items() if c > 1]
            add(run.name, PUBLIC_NAME[part], "no duplicate trial ids", not dup, f"{len(dup)} duplicates")
            unknown = [i for i in ids if i not in trials]
            add(run.name, PUBLIC_NAME[part], "every response matches an input trial", not unknown, f"{len(unknown)} unknown")
            complete = len(set(ids)) == len(trials)
            add(run.name, PUBLIC_NAME[part], "run is complete", complete, f"{len(set(ids))}/{len(trials)} trials answered")
            bad_sha = [r["trial_id"] for r in resp if r.get("prompt_sha") and r["trial_id"] in trials and not trials[r["trial_id"]]["prompt_sha"].startswith(r["prompt_sha"])]
            add(run.name, PUBLIC_NAME[part], "prompt hash equals the input prompt", not bad_sha, f"{len(bad_sha)} mismatches")
            digests = {r.get("model_digest") for r in resp if r.get("model_digest")}
            add(run.name, PUBLIC_NAME[part], "one model digest in the run", len(digests) <= 1, ", ".join(sorted(str(d)[:12] for d in digests)))
            labels_ok = all(set(r["parsed"]) <= set(trials[r["trial_id"]]["labels"]) or not r["format_ok"] for r in resp if r["trial_id"] in trials)
            add(run.name, PUBLIC_NAME[part], "valid answers use only displayed labels", labels_ok)
            n3 = all(len(r["parsed"]) == 3 or not r["format_ok"] for r in resp)
            add(run.name, PUBLIC_NAME[part], "valid answers contain exactly three labels", n3)
        opts = {json.dumps(r.get("options"), sort_keys=True) for ph in EXPERIMENT_PARTS if run.file(ph).exists() for r in read_jsonl(run.file(ph))}
        add(run.name, "all", "one set of inference options in the run", len(opts) == 1, "; ".join(sorted(opts))[:160])
    cur = Path(inputs) / "MANIFEST.sha256"
    if cur.exists():
        bad = []
        for line in cur.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, rel = line.split(None, 1)
            rel = rel.strip().lstrip("*")
            f = (ROOT if rel.startswith("configs/") else Path(inputs)) / rel
            if not f.exists() or sha256(f) != h:
                bad.append(rel)
        add("inputs", "inputs", "inputs match MANIFEST.sha256", not bad, ", ".join(bad))
    return pd.DataFrame(rows)


def frozen_config_hashes() -> pd.DataFrame:
    return pd.DataFrame([{"file": p.name, "sha256": sha256(p)} for p in sorted(Path(FROZEN).glob("*.yaml"))])
