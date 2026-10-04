"""
Controlled timing-rank follow-up: trials built on the frozen numerical seed windows of the main experiment.

For every eligible source window there are 6 variants: content {original, faster, slower} x position {first, last}.
Between variants only the timing value of the focal event and its position change; the schema, fields, events,
participants, labels and background order are identical and have exactly the same form as in the main experiment (the same
serialization functions are used).

Manipulation rule (rank):
    faster = min(original, min(background)) - epsilon     -> the focal event is strictly the fastest in the list
    slower = max(original, max(background)) + epsilon     -> the focal event is strictly the slowest in the list
    epsilon = epsilon_scale * robust scale of the background (1.4826 * MAD, floor 50 ms)
Counterfactual values must lie in the observed range of valid values of the same class in the same race
(plausibility rule); windows that fail it are excluded before any model is run.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

from si_build_dataset import race_key_of, read_jsonl, write_jsonl
from si_f1windows import comparison_class, metric_of
from si_make_trials import build_prompt, make_labels, serialize_event

FIELD = {"SectorCross": ("sector_time_ms", "sector_time_s"), "LapComplete": ("lap_time_ms", "lap_time_s")}


def file_sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def robust(values, mad_scale=1.4826, floor=50.0):
    med = st.median(values)
    mad = st.median([abs(v - med) for v in values])
    return med, mad, max(mad_scale * mad, floor)


def load_sources(dataset_dir: Path):
    """Numerical seed windows and the observed [min, max] of valid values per (race, comparison class)."""
    dataset_dir = Path(dataset_dir)
    sha = file_sha(dataset_dir / "seed_windows.jsonl")
    seeds = [s for s in read_jsonl(dataset_dir / "seed_windows.jsonl") if "numerical_outlier" in s["selected_for"]]
    ranges: dict[tuple, list] = defaultdict(list)
    for e in read_jsonl(dataset_dir / "primitive_events.jsonl"):
        cls = comparison_class(e)
        if cls and e["numeric_valid"] and metric_of(e):
            ranges[(race_key_of(e), cls)].append(metric_of(e))
    return seeds, {k: (min(v), max(v)) for k, v in ranges.items()}, sha


def build(cfg: dict, main_cfg: dict, cfg_path: Path, dataset_dir: Path) -> tuple[list[dict], list[dict]]:
    seeds, ranges, _ = load_sources(dataset_dir)
    m = cfg["manipulation"]
    assert m["rule"] == "R", "only the rank rule (R) is implemented in this release"
    # The section holding the robust-scale parameters is called "B" in the frozen configuration file
    # (the file is kept byte-identical because its hash is stored in every trial); only mad_scale and mad_floor_ms are used.
    scale_cfg = m["B"]
    prefix = cfg["trial_prefix"]
    tcfg = main_cfg["trials"]
    hidden = frozenset(tcfg.get("hidden_fields", []))
    all_drivers = {e["driver"] for s in seeds for e in s["events"]}   # as in the main experiment: labels never equal driver codes
    cfg_hash = file_sha(cfg_path)
    trials, exclusions = [], []
    for s in sorted(seeds, key=lambda s: s["window_id"]):
        n = s["analysis_metadata"]["numerical_outlier"]
        evs = {e["event_id"]: e for e in s["events"]}
        focal = evs[n["extreme_event_id"]]
        group = [evs[i] for i in n["group_event_ids"]]
        background = [e for e in group if e is not focal]
        x = metric_of(focal)
        bg_vals = [metric_of(e) for e in background]
        _, _, scale = robust(bg_vals, scale_cfg["mad_scale"], scale_cfg["mad_floor_ms"])
        eps = round(m["R"]["epsilon_scale"] * scale)
        values = {"original": x, "faster": min(x, min(bg_vals)) - eps, "slower": max(x, max(bg_vals)) + eps}
        lo, hi = ranges[(s["window_id"].rsplit("_", 1)[0], n["comparison_class"])]
        bad = [c for c in ("faster", "slower") if not lo <= values[c] <= hi]
        if bad or len(group) < 2:
            exclusions.append({"source_seed_window_id": s["window_id"], "reason":
                               f"implausible {'/'.join(bad)} (outside [{lo}; {hi}] ms of the class in the race)" if bad else "list too short"})
            continue
        # Once per source window: background order, labels and anonymization are shared by all 6 variants.
        rng = random.Random(f"{cfg['random_seed']}|{s['window_id']}")
        rng.shuffle(background)
        labels = make_labels(rng, len(group), tcfg["label_alphabet"], tcfg["label_length"], all_drivers)
        label_of = {e["event_id"]: lab for e, lab in zip([focal] + background, labels)}
        drivers = sorted({e["driver"] for e in group})
        ids = [f"D{i + 1:02d}" for i in range(len(drivers))]
        rng.shuffle(ids)
        dmap = dict(zip(drivers, ids)) if tcfg["driver_repr"] == "anon" else None
        ms_field, s_field = FIELD[focal["type"]]
        for content in cfg["content_conditions"]:
            cf = dict(focal)
            cf[ms_field] = values[content]
            for position in cfg["positions"]:
                order = [cf] + background if position == "first" else background + [cf]
                rows = [serialize_event(e, label_of[e["event_id"]], dmap, None, hidden) for e in order]
                prompt = build_prompt(main_cfg, rows)
                sha = hashlib.sha256(prompt.encode()).hexdigest()
                extra = {"rank_epsilon_ms": eps,
                         "focal_rank_status": "fastest" if values[content] < min(bg_vals) else
                                              "slowest" if values[content] > max(bg_vals) else "middle"}
                trials.append({
                    "trial_id": f"{prefix}|{s['window_id']}|{content}|{position}",
                    "phase": "phase2_realdata", "source_seed_window_id": s["window_id"],
                    "source_focal_event_id": focal["event_id"], "source_race": s["session"]["event"],
                    "source_session": s["session"]["session"], "comparison_class": n["comparison_class"],
                    "event_type": focal["type"], "manipulated_field": s_field,
                    "original_value": round(x / 1000, 3), "counterfactual_value": round(values[content] / 1000, 3),
                    "content_condition": content, "position_condition": position,
                    "direction": {"original": "none", "faster": "lower", "slower": "higher"}[content],
                    "semantic_direction": {"original": "none", "faster": "faster", "slower": "slower"}[content],
                    "original_direction": n["direction"], "original_robust_z": n["robust_z"],
                    "manipulation_absolute": round((values[content] - x) / 1000, 3),
                    "manipulation_relative": round((values[content] - x) / x, 5),
                    "background_scale_ms": round(scale, 3), **extra, "n_events": len(order),
                    "focal_label": label_of[focal["event_id"]], "focal_position": order.index(cf),
                    "labels": {label_of[e["event_id"]]: e["event_id"] for e in order},
                    "driver_map": dmap,
                    "prompt": prompt, "prompt_sha": sha[:12], "prompt_hash": sha, "config_hash": cfg_hash,
                    "family": "phase2_realdata", "meta": {},   # compatibility with the runner
                })
    return trials, exclusions


def build_followup(dataset_dir: Path, main_experiment: Path, config: Path, out_dir: Path,
                      allow_source_mismatch: bool = False) -> dict:
    """Write controlled_timing_rank_followup_trials.jsonl and controlled_timing_rank_followup_exclusions.json into out_dir; returns a small summary."""
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    main_cfg = yaml.safe_load(Path(main_experiment).read_text(encoding="utf-8"))
    sha = file_sha(Path(dataset_dir) / "seed_windows.jsonl")
    expected = cfg["source"]["phase1_seed_windows_sha256"]
    if sha != expected and not allow_source_mismatch:
        raise SystemExit(f"seed_windows.jsonl hash {sha[:12]} differs from the frozen {expected[:12]}; "
                         "use allow_source_mismatch=True only for a deliberately different dataset")
    trials, exclusions = build(cfg, main_cfg, Path(config), Path(dataset_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "controlled_timing_rank_followup_trials.jsonl", trials)
    (out_dir / "controlled_timing_rank_followup_exclusions.json").write_text(json.dumps(exclusions, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"eligible_sources": len(trials) // 6, "excluded": len(exclusions), "trials": len(trials)}
