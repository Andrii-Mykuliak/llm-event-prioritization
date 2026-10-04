"""
Verification suites for the generated input data.

verify_main_experiment():       16 checks on the main-experiment trials (counts and balance, one comparison class in numerical trials,
                       anonymization, no hidden fields in the prompt).
verify_followup():  28 checks on the controlled timing-rank follow-up trials (source integrity, schema, content, position, factorial
                       completeness, leakage, reproducibility).
Both return a VerifyResult with a Markdown report.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from conditions import CONDITION_OF_DATA_KEY
from si_build_dataset import read_jsonl
from si_controlled_timing_rank_followup import build_followup

REAL_CODES = re.compile(r'"driver":\s*"(?!D\d{2}")')
HIDDEN_WORDS = ["original", "faster", "slower", "counterfactual", "content_condition", "position_condition",
                "first", "last", "focal", "p2r", "manipulat"]


@dataclass
class VerifyResult:
    name: str
    results: list[bool] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    report: str = ""

    @property
    def ok(self) -> bool:
        return all(self.results)

    def summary(self) -> str:
        return f"{self.name}: {'ALL CHECKS PASSED' if self.ok else 'CHECKS FAILED'} ({sum(self.results)}/{len(self.results)})"


def events_of(prompt: str) -> list[dict]:
    return [json.loads(line) for line in prompt.splitlines() if line.startswith("{")]


def template_of(prompt: str) -> str:
    return re.sub(r"Events:\n.*?\n\n", "Events:\n{events}\n\n", prompt, flags=re.S)


# --------------------------------------------------------------------------- main experiment

def verify_main_experiment(trials_path: Path, dataset_dir: Path, expected_total: int | None = 1040) -> VerifyResult:
    trials = read_jsonl(Path(trials_path))
    seeds = {s["window_id"]: s for s in read_jsonl(Path(dataset_dir) / "seed_windows.jsonl")}
    real_drivers = {e["driver"] for s in seeds.values() for e in s["events"]}
    v = VerifyResult("Main experiment trials")

    def check(name: str, ok: bool, detail: str = "") -> None:
        v.results.append(bool(ok))
        v.lines.append(f"| {'PASS' if ok else 'FAIL'} | {name} | {detail} |")

    fam = Counter(t["family"] for t in trials)
    check("Number of trials" + (f" (frozen design: {expected_total})" if expected_total else ""),
          expected_total is None or len(trials) == expected_total,
          f"{len(trials)}: " + ", ".join(f"{CONDITION_OF_DATA_KEY[k]} {n}" for k, n in sorted(fam.items())))
    ids = [t["trial_id"] for t in trials]
    check("Unique trial_id", len(ids) == len(set(ids)))
    for f in sorted(fam):
        per_pos = Counter(t["perm"] for t in trials if t["family"] == f)
        check(f"{CONDITION_OF_DATA_KEY[f]}: five focal positions in equal numbers", len(set(per_pos.values())) == 1 and len(per_pos) == 5,
              " / ".join(str(per_pos[k]) for k in sorted(per_pos)))
        per_seed = Counter((t["window_id"], t["level"]) for t in trials if t["family"] == f)
        check(f"{CONDITION_OF_DATA_KEY[f]}: every seed x level presentation has 5 trials", set(per_seed.values()) == {5}, f"{len(per_seed)} combinations")
    dens = Counter(t["n_events"] for t in trials if t["family"] == "density")
    check("candidate_set_size: levels of 5/10/20 events", set(dens) == {5, 10, 20}, ", ".join(f"n={k}: {n}" for k, n in sorted(dens.items())))
    zs = Counter("2.5-4" if t["meta"]["robust_z"] < 4 else "4-7" if t["meta"]["robust_z"] < 7 else ">=7"
                 for t in trials if t["family"] == "numerical_outlier" and t["perm"] == 0)
    check("numerical_value: z strata with equal numbers of seeds", len(set(zs.values())) == 1, ", ".join(f"{k}: {n}" for k, n in zs.items()))
    bad = []
    for t in trials:
        if t["family"] != "numerical_outlier":
            continue
        key = {(e["type"], e.get("sector")) for e in events_of(t["prompt"])}
        if len(key) != 1:
            bad.append(t["trial_id"])
    check("numerical_value: only one comparison class in the prompt", not bad, f"violations: {len(bad)}")
    n_num = Counter(t["n_events"] for t in trials if t["family"] == "numerical_outlier")
    check("numerical_value: at least 5 events per presentation", min(n_num) >= 5, ", ".join(f"n={k}: {n}" for k, n in sorted(n_num.items())))
    leaks_code = [t["trial_id"] for t in trials if REAL_CODES.search(t["prompt"])]
    check("No real driver codes in the driver field", not leaks_code, f"violations: {len(leaks_code)}")
    words = {w for t in trials for w in re.findall(r"\b[A-Z]{3}\b", t["prompt"])}
    labels = {lab for t in trials for lab in t["labels"]}
    leaked = sorted((words - labels) & real_drivers)
    check("No driver code appears anywhere in a prompt", not leaked, ", ".join(leaked) or "none")
    pos_leaks = [t["trial_id"] for t in trials if '"position"' in t["prompt"]]
    check("The position field is absent from the model-visible JSON", not pos_leaks, f"violations: {len(pos_leaks)}")
    kept = all(any(x["driver"] in real_drivers for x in t["hidden_attrs"].values()) for t in trials)
    check("Real driver/position kept in hidden_attrs", kept)
    fields = Counter(k for t in trials for e in events_of(t["prompt"]) for k in e)
    rng = random.Random(7)
    examples = []
    for f in ("type_outlier", "numerical_outlier", "density"):
        pool = [t for t in trials if t["family"] == f]
        if not pool:
            continue
        if f == "density":
            picks = [rng.choice([t for t in pool if t["n_events"] == n]) for n in (5, 10, 20) if any(t["n_events"] == n for t in pool)]
        else:
            picks = rng.sample(pool, min(3, len(pool)))
        for t in picks:
            examples.append(f"### {CONDITION_OF_DATA_KEY[f]}: `{t['trial_id']}`\n\nfocal = `{t['focal_label']}` (position {t['focal_position'] + 1} of {t['n_events']})\n\n"
                            f"```text\n{t['prompt']}\n```\n")
    v.report = "\n".join(["# Main experiment trial verification", "", f"Result: **{v.summary()}**", "", "| | Check | Details |", "|---|---|---|",
                          *v.lines, "", f"Model-visible fields (occurrences): {dict(fields)}", "", "## Example prompts", "", *examples])
    return v


# --------------------------------------------------------------------------- controlled timing-rank follow-up

def verify_followup(config: Path, trials_path: Path, main_experiment_trials: Path, dataset_dir: Path, main_experiment: Path) -> VerifyResult:
    config = Path(config)
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    trials = read_jsonl(Path(trials_path))
    main_trials = [t for t in read_jsonl(Path(main_experiment_trials)) if t["family"] == "numerical_outlier"]
    seeds = {s["window_id"]: s for s in read_jsonl(Path(dataset_dir) / "seed_windows.jsonl")}
    by_src = defaultdict(dict)
    for t in trials:
        by_src[t["source_seed_window_id"]][(t["content_condition"], t["position_condition"])] = t
    v = VerifyResult("Controlled timing-rank follow-up trials")

    def check(no, name, ok, detail=""):
        v.results.append(bool(ok))
        v.lines.append(f"| {no} | {'PASS' if ok else 'FAIL'} | {name} | {detail} |")

    check(1, "every source id exists among the frozen main-experiment seeds", all(b in seeds for b in by_src))
    check(2, "every source is a numerical window", all("numerical_outlier" in seeds[b]["selected_for"] for b in by_src))
    check(3, "focal id equals the main-experiment focal id", all(t["source_focal_event_id"] == seeds[t["source_seed_window_id"]]["analysis_metadata"]["numerical_outlier"]["extreme_event_id"] for t in trials))
    check(4, "comparison class equals the main experiment", all(t["comparison_class"] == seeds[t["source_seed_window_id"]]["analysis_metadata"]["numerical_outlier"]["comparison_class"] for t in trials))
    main_schema = {tuple(r.keys()) for t in main_trials for r in events_of(t["prompt"])}
    p2_schema = {tuple(r.keys()) for t in trials for r in events_of(t["prompt"])}
    check(5, "visible JSON schema equals the main experiment (same keys and key order)", p2_schema <= main_schema, f"{sorted(p2_schema)}")
    main_types = {r["type"] for t in main_trials for r in events_of(t["prompt"])}
    check(6, "event types unchanged", {r["type"] for t in trials for r in events_of(t["prompt"])} <= main_types)
    check(7, "field names unchanged", {k for t in trials for r in events_of(t["prompt"]) for k in r} <= {k for s in main_schema for k in s})
    check(8, "no artificial `value` / `entity` fields", not any(k in ("value", "entity", "score", "metric") for s in p2_schema for k in s))
    p1_val = {}
    for t in main_trials:
        for r in events_of(t["prompt"]):
            if r["id"] == t["focal_label"]:
                p1_val[t["window_id"]] = r.get("sector_time_s", r.get("lap_time_s"))

    def fval(t):
        r = next(r for r in events_of(t["prompt"]) if r["id"] == t["focal_label"])
        return r.get("sector_time_s", r.get("lap_time_s"))
    check(9, "original value equals the main-experiment value (in the prompt)", all(fval(x[("original", p)]) == p1_val[b] for b, x in by_src.items() for p in cfg["positions"]))
    check(10, "faster < original", all(fval(x[("faster", "first")]) < fval(x[("original", "first")]) for x in by_src.values()))
    check(11, "slower > original", all(fval(x[("slower", "first")]) > fval(x[("original", "first")]) for x in by_src.values()))

    def bgv(t):
        return [r.get("sector_time_s", r.get("lap_time_s")) for r in events_of(t["prompt"]) if r["id"] != t["focal_label"]]

    def expected(t):
        if t["content_condition"] == "faster":
            return min(t["original_value"], min(bgv(t))) - t["rank_epsilon_ms"] / 1000
        return max(t["original_value"], max(bgv(t))) + t["rank_epsilon_ms"] / 1000
    e = cfg["manipulation"]["R"]["epsilon_scale"]
    moved = [t for t in trials if t["content_condition"] != "original"]
    exact = all(abs(t["counterfactual_value"] - expected(t)) <= 0.0005 for t in moved)
    rank_ok = all(fval(t) < min(bgv(t)) if t["content_condition"] == "faster" else fval(t) > max(bgv(t)) for t in moved)
    check(12, f"rank rule applied exactly (epsilon = {e} x robust background scale); faster strictly fastest, slower strictly slowest", exact and rank_ok)

    def bg(t):
        return [r for r in events_of(t["prompt"]) if r["id"] != t["focal_label"]]

    def seed_bg_values(b):
        n = seeds[b]["analysis_metadata"]["numerical_outlier"]
        evs = {x["event_id"]: x for x in seeds[b]["events"]}
        return sorted(round((evs[i].get("sector_time_ms") or evs[i].get("lap_time_ms")) / 1000, 3)
                      for i in n["group_event_ids"] if i != n["extreme_event_id"])
    same_variants = all(len({json.dumps(bg(t)) for t in x.values()}) == 1 for x in by_src.values())
    same_as_seed = all(sorted(r.get("sector_time_s", r.get("lap_time_s")) for r in bg(x[("original", "first")])) == seed_bg_values(b) for b, x in by_src.items())
    check(13, "background values unchanged (across the 6 variants and against the main-experiment seed)", same_variants and same_as_seed)
    check(14, "participants unchanged", all(len({json.dumps(sorted((r["id"], r["driver"]) for r in events_of(t["prompt"]))) for t in x.values()}) == 1 for x in by_src.values()))
    check(15, "ids unchanged", all(len({json.dumps(t["labels"], sort_keys=True) for t in x.values()}) == 1 for x in by_src.values()))
    check(16, "first: focal index = 0", all(events_of(t["prompt"])[0]["id"] == t["focal_label"] for t in trials if t["position_condition"] == "first"))
    check(17, "last: focal index = n - 1", all(events_of(t["prompt"])[-1]["id"] == t["focal_label"] for t in trials if t["position_condition"] == "last"))
    check(18, "relative order of the background is the same", all(len({json.dumps([r["id"] for r in bg(t)]) for t in x.values()}) == 1 for x in by_src.values()))
    cells = {(c, p) for c in cfg["content_conditions"] for p in cfg["positions"]}
    check(19, "exactly 6 variants per eligible source", all(len(x) == 6 for x in by_src.values()))
    ids = [t["trial_id"] for t in trials]
    check(20, "no duplicated condition cells", len(ids) == len(set(ids)))
    check(21, "no missing condition cells", all(set(x) == cells for x in by_src.values()))
    visible = ["\n".join(l for l in t["prompt"].splitlines() if l.startswith("{")) for t in trials]
    leak = sorted({w for vis in visible for w in HIDDEN_WORDS if re.search(w, vis, re.I)})
    real = {x["driver"] for s in seeds.values() for x in s["events"]}
    codes = {w for t in trials for w in re.findall(r"\b[A-Z]{3}\b", t["prompt"])} - {l for t in trials for l in t["labels"]}
    check(22, "hidden metadata is not in the prompt (no position, no real driver codes)",
          not any("position" in r for t in trials for r in events_of(t["prompt"])) and not (codes & real))
    check(23, "condition names and source labels are not shown to the model", not leak, ", ".join(leak) or "none")
    with tempfile.TemporaryDirectory() as d:
        build_followup(dataset_dir, main_experiment, config, Path(d), allow_source_mismatch=True)
        same = (Path(d) / "controlled_timing_rank_followup_trials.jsonl").read_bytes() == Path(trials_path).read_bytes()
    check(24, "regeneration is byte-identical", same)
    cfg_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    check(25, "config hash is stable (equals the hash in every trial)", all(t["config_hash"] == cfg_hash for t in trials), cfg_hash[:12])
    src_sha = hashlib.sha256((Path(dataset_dir) / "seed_windows.jsonl").read_bytes()).hexdigest()
    check(26, "Main-experiment source hash equals the hash in the configuration", src_sha == cfg["source"]["phase1_seed_windows_sha256"], src_sha[:12])
    check(27, "prompt template equals the main experiment (byte-wise)", {template_of(t["prompt"]) for t in trials} == {template_of(t["prompt"]) for t in main_trials})
    check(28, "prompt hash = sha256(prompt)", all(hashlib.sha256(t["prompt"].encode()).hexdigest() == t["prompt_hash"] for t in trials))
    v.report = "\n".join(["# Controlled timing-rank follow-up trial verification", "", f"Result: **{v.summary()}**", "",
                          f"Eligible source windows: {len(by_src)}; trials: {len(trials)} (6 x {len(by_src)}).", "",
                          "| # | | Check | Details |", "|---|---|---|---|", *v.lines])
    return v
