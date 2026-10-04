"""
Run the frozen experiment with ONE language model.

    run_llm(llm_config, parts=("main_experiment", "controlled_timing_rank_followup"))

For every trial the model gets exactly the stored prompt; the answer is parsed with the frozen rule and appended to
results/<llm name>__<experiment part>_responses.jsonl. The run is resumable: trials that already have an answer are skipped.
Run the pipeline once per language model; analysis then compares any set of result folders.
"""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from pathlib import Path

import requests
import yaml

from conditions import CONDITION_OF_DATA_KEY
from paths import __version__
from llm_config import load_llm_config
from llm_backends import make_backend
from llm_backends import now_iso
from paths import INPUTS, MAIN_EXPERIMENT_CONFIG, EXPERIMENT_PARTS, RESULTS, RUN_ERRORS, RUN_META, run_file
from parsing import invalid_labels, parse_labels


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_inputs(inputs: Path) -> dict:
    """Compare the trial files with MANIFEST.sha256 (if present); returns {path: hash} of the files used."""
    import hashlib
    used = {}
    man = Path(inputs) / "MANIFEST.sha256"
    want = {}
    if man.exists():
        for line in man.read_text(encoding="utf-8").splitlines():
            if line.strip():
                h, rel = line.split(None, 1)
                want[rel.strip()] = h
    for rel in (EXPERIMENT_PARTS["main_experiment"][0], EXPERIMENT_PARTS["controlled_timing_rank_followup"][0]):
        p = Path(inputs) / rel
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            used[rel] = h
            if rel in want and want[rel] != h:
                raise SystemExit(f"{rel} differs from MANIFEST.sha256 ({h[:12]} vs {want[rel][:12]}). "
                                 "Use the shipped inputs or rebuild them with scripts/make_inputs.py.")
    return used


def run_llm(llm_config, parts=("main_experiment", "controlled_timing_rank_followup"), inputs_dir: Path = INPUTS, results_root: Path = RESULTS,
            limit: int | None = None, conditions: list[str] | None = None, progress=print, smoke_suffix: bool = True) -> Path:
    """Run the chosen parts for one LLM; returns the results directory (files are <name>__<experiment part>_responses.jsonl)."""
    cfg = load_llm_config(llm_config) if not hasattr(llm_config, "backend") else llm_config
    exp = yaml.safe_load(MAIN_EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    k = exp["prompt"]["k"]
    label_len = exp["trials"]["label_length"]
    used_inputs = check_inputs(inputs_dir)
    backend = make_backend(cfg)
    env = backend.check()
    folder = cfg.name + ("__smoke" if (limit and smoke_suffix) else "")
    run_dir = Path(results_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_file(run_dir, folder, RUN_META)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"history": []}
    meta.update({"name": cfg.name, "short_label": cfg.label(), "display_name": cfg.display_name, "size_order": cfg.size_order,
                 "backend": cfg.backend, "model": cfg.model, "options": cfg.options, "think": cfg.think, "environment": env,
                 "deviations_from_frozen": cfg.deviations_from_frozen, "inputs_sha256": used_inputs,
                 "package_version": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
                 "mock": bool(getattr(backend, "mock", False))})
    if meta["mock"]:
        progress("WARNING: mock backend. These answers are fake and must not be used as scientific results.")
    if cfg.deviations_from_frozen:
        progress(f"NOTE: deviations from the frozen settings are recorded: {cfg.deviations_from_frozen}")
    started = now_iso()
    summary = {}
    for part in parts:
        trials_rel, resp_name = EXPERIMENT_PARTS[part]
        trials = read_jsonl(Path(inputs_dir) / trials_rel)
        if conditions:
            trials = [t for t in trials if CONDITION_OF_DATA_KEY.get(t["family"]) in conditions]
        if limit:
            trials = random.Random(0).sample(trials, min(limit, len(trials)))
        path = run_file(run_dir, folder, resp_name)
        prev = read_jsonl(path) if path.exists() else []
        sha = {t["trial_id"]: t["prompt_sha"] for t in trials}
        stale = [r["trial_id"] for r in prev if r["trial_id"] in sha and r["prompt_sha"] != sha[r["trial_id"]]]
        if stale:
            raise SystemExit(f"{path}: {len(stale)} answers belong to a different version of the prompts (e.g. {stale[0]}). "
                             "Move those files away or choose another results directory.")
        done = {r["trial_id"] for r in prev}
        todo = [t for t in trials if t["trial_id"] not in done]
        progress(f"[{cfg.name}] {part}: {len(done)} done, {len(todo)} to do")
        t_start = time.perf_counter()
        errors = 0
        with path.open("a", encoding="utf-8") as f:
            for i, t in enumerate(todo, 1):
                result, err = None, None
                for attempt in range(cfg.retries + 1):
                    try:
                        result = backend.ask(t["prompt"])
                        break
                    except requests.RequestException as e:
                        err = str(e)[:300]
                        time.sleep(min(2 ** attempt, 20))
                if result is None:
                    errors += 1
                    with run_file(run_dir, folder, RUN_ERRORS).open("a", encoding="utf-8") as ef:
                        ef.write(json.dumps({"trial_id": t["trial_id"], "part": part, "error": err, "time": now_iso()}) + "\n")
                    continue
                valid = set(t["labels"])
                parsed = parse_labels(result["text"], valid, k)
                rec = {"trial_id": t["trial_id"], "llm": cfg.name, "model": cfg.model, **env, "prompt_sha": t["prompt_sha"],
                       "timestamp": now_iso(), "raw": result["text"], "parsed": parsed, "format_ok": len(parsed) == k,
                       "invalid_labels": invalid_labels(result["text"], valid, label_len), "latency_s": result["latency_s"],
                       "prompt_tokens": result["prompt_tokens"], "output_tokens": result["output_tokens"],
                       "options": cfg.options, "think": cfg.think}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(todo):
                    el = time.perf_counter() - t_start
                    progress(f"  {i}/{len(todo)}  last={parsed}  {result['latency_s']}s  elapsed {el / 60:.1f} min")
        answered = len(read_jsonl(path))
        summary[part] = {"trials_in_run": len(trials), "answered": answered, "new_errors": errors}
    meta["history"].append({"started_utc": started, "finished_utc": now_iso(), "parts": list(parts), "limit": limit, "summary": summary})
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    progress(f"[{cfg.name}] finished: {summary}")
    return run_dir
