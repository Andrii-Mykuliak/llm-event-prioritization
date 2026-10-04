"""Loading of run folders and trials into analysis tables."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from conditions import CONDITION_OF_DATA_KEY
from llm_config import _guess_size
from paths import INPUTS, MAIN_EXPERIMENT_CONFIG, EXPERIMENT_PARTS, RUN_META, SEED_WINDOWS, run_file


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class RunInfo:
    name: str
    path: Path
    label: str
    order: float | None
    mock: bool = False
    reference: bool = False

    def file(self, part: str) -> Path:
        return run_file(self.path, self.name, EXPERIMENT_PARTS[part][1])

    @property
    def has_main_experiment(self) -> bool:
        return self.file("main_experiment").exists()

    @property
    def has_followup(self) -> bool:
        return self.file("controlled_timing_rank_followup").exists()


def _run_names(root: Path) -> list[str]:
    names = set()
    for _, suffix in EXPERIMENT_PARTS.values():
        for f in root.glob(f"*__{suffix}"):
            names.add(f.name[: -len(suffix) - 2])
    return sorted(names)


def discover_runs(*roots: Path, include_smoke: bool = False, include_mock: bool = False, reference_roots: tuple[Path, ...] = ()) -> list[RunInfo]:
    """A run = the files <name>__<experiment part>_responses.jsonl in a folder. Sorted by model size when it can be read from the name."""
    runs: list[RunInfo] = []
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for name in _run_names(root):
            if name.endswith("__smoke") and not include_smoke:
                continue
            meta_path = run_file(root, name, RUN_META)
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            if meta.get("mock") and not include_mock:
                continue
            label, order = _guess_size(name)
            label = meta.get("short_label") or label or name
            order = meta.get("size_order", order)
            if name in seen:
                raise ValueError(f"run name {name} appears twice; rename the files of one of them")
            seen.add(name)
            runs.append(RunInfo(name, root, label, order, bool(meta.get("mock")), any(root.resolve() == Path(r).resolve() for r in reference_roots)))
    runs.sort(key=lambda r: (r.order is None, r.order if r.order is not None else 0, r.name))
    return runs


def frozen_positions() -> list[float]:
    return yaml.safe_load(MAIN_EXPERIMENT_CONFIG.read_text(encoding="utf-8"))["trials"]["focal_positions"]


def position_labels() -> list[str]:
    return [str(Fraction(p).limit_denominator(8)) for p in frozen_positions()]


def _pos_class(k: int, n: int) -> str:
    return "first" if k == 0 else "last" if k == n - 1 else "middle"


class TrialStore:
    """main-experiment trials with helpers to read the displayed values from the prompts."""

    def __init__(self, inputs: Path = INPUTS):
        self.trials = {t["trial_id"]: t for t in read_jsonl(Path(inputs) / EXPERIMENT_PARTS["main_experiment"][0])}
        self._cache: dict[str, list[dict]] = {}

    def rows(self, trial_id: str) -> list[dict]:
        if trial_id not in self._cache:
            self._cache[trial_id] = [json.loads(l) for l in self.trials[trial_id]["prompt"].splitlines() if l.startswith("{")]
        return self._cache[trial_id]

    @staticmethod
    def value_of(row: dict):
        return row.get("sector_time_s", row.get("lap_time_s"))


def load_main_experiment(runs: list[RunInfo], inputs: Path = INPUTS) -> pd.DataFrame:
    store = TrialStore(inputs)
    pos_lab = position_labels()
    out = []
    for run in runs:
        if not run.has_main_experiment:
            continue
        for r in read_jsonl(run.file("main_experiment")):
            t = store.trials[r["trial_id"]]
            parsed, n, focal = r["parsed"], t["n_events"], t["focal_label"]
            k = t["focal_position"]
            out.append({
                "llm": run.name, "label": run.label, "trial_id": t["trial_id"], "condition": CONDITION_OF_DATA_KEY[t["family"]], "window_id": t["window_id"],
                "race": t["race"], "level": t["level"], "n_events": n, "perm": t["perm"], "k": k, "pos": _pos_class(k, n),
                "pos5": pos_lab[t["perm"]], "format_ok": bool(r["format_ok"]), "invented": int(bool(r.get("invalid_labels"))),
                "parsed": parsed, "raw": r["raw"], "focal_top1": int(bool(parsed) and parsed[0] == focal), "focal_top3": int(focal in parsed),
                "focal_rank": parsed.index(focal) + 1 if focal in parsed else np.nan, "chance_top1": 1 / n, "chance_top3": min(3, n) / n,
                "robust_z": t["meta"].get("robust_z"), "direction": t["meta"].get("direction"),
                "comp_class": t["meta"].get("comparison_class"), "minority_type": t["meta"].get("minority_type"), "focal_type": t["focal_type"],
                "top1_event": t["labels"].get(parsed[0]) if parsed else None,
                "top3_events": tuple(sorted(t["labels"][p] for p in parsed)),
                "latency_s": r.get("latency_s"), "prompt_tokens": r.get("prompt_tokens"), "output_tokens": r.get("output_tokens"),
                "prompt_sha": r.get("prompt_sha"), "model_digest": r.get("model_digest"),
            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["log_n"] = np.log(df.n_events)
    df["log_z"] = np.log(df.robust_z.astype(float))
    df["slower"] = (df.direction == "slower").astype(int)
    df["y3"] = df.focal_top3
    df["y1"] = df.focal_top1
    return df


def load_followup(runs: list[RunInfo], inputs: Path = INPUTS) -> pd.DataFrame:
    trials = {t["trial_id"]: t for t in read_jsonl(Path(inputs) / EXPERIMENT_PARTS["controlled_timing_rank_followup"][0])}
    out = []
    for run in runs:
        if not run.has_followup:
            continue
        for r in read_jsonl(run.file("controlled_timing_rank_followup")):
            t = trials[r["trial_id"]]
            p = r["parsed"]
            out.append({"llm": run.name, "label": run.label, "src": t["source_seed_window_id"], "content": t["content_condition"],
                        "position": t["position_condition"], "orig_dir": t["original_direction"], "cls": t["comparison_class"],
                        "n": t["n_events"], "format_ok": bool(r["format_ok"]), "invented": int(bool(r.get("invalid_labels"))),
                        "driver_answer": int(bool(re.search(r"\bD\d{2}\b", r["raw"]))),
                        "top1": int(bool(p) and p[0] == t["focal_label"]), "top3": int(t["focal_label"] in p),
                        "chance1": 1 / t["n_events"], "chance3": min(3, t["n_events"]) / t["n_events"],
                        "rank_status": t.get("focal_rank_status")})
    return pd.DataFrame(out)


def load_seeds(inputs: Path = INPUTS) -> list[dict]:
    return read_jsonl(Path(inputs) / SEED_WINDOWS)
