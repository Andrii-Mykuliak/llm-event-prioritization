"""
Build LLM trials from the frozen seed windows.

For every seed, K presentations (permutations) are created with a controlled focal-event position
and random opaque labels. For the candidate-set-size condition the background
is drawn only from real events of the same type and the same race.

Usage:
    python make_trials.py --dataset ./inputs --config experiment.yaml --out ./inputs/main_experiment_trials.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from si_build_dataset import race_key_of, read_jsonl, write_jsonl


def fmt_seconds(ms: int | None) -> float | None:
    return None if ms is None else round(ms / 1000, 3)


def serialize_event(e: dict, label: str, driver_map: dict | None, offset_ms: int | None,
                    hidden: frozenset = frozenset()) -> dict:
    """Model-visible representation: only the factual event fields, without service metadata.
    Fields in `hidden` (e.g. position) are not shown to the model; they are only stored in the trial metadata."""
    out = {"id": label, "type": e["type"], "driver": driver_map[e["driver"]] if driver_map else e["driver"]}
    if e["type"] == "SectorCross":
        out.update(sector=e["sector"], lap=e["lap_number"], sector_time_s=fmt_seconds(e.get("sector_time_ms")))
    elif e["type"] == "LapComplete":
        out.update(lap=e["lap_number"], lap_time_s=fmt_seconds(e.get("lap_time_ms")), position=e.get("position"))
    if offset_ms is not None:
        out["t_s"] = fmt_seconds(offset_ms)
    return {k: v for k, v in out.items() if v is not None and k not in hidden}


def make_labels(rng: random.Random, n: int, alphabet: str, length: int, forbidden: set[str]) -> list[str]:
    labels: list[str] = []
    while len(labels) < n:
        cand = "".join(rng.choice(alphabet) for _ in range(length))
        if cand not in forbidden and cand not in labels:
            labels.append(cand)
    return labels


def arrange(rng: random.Random, focal: dict, others: list[dict], frac: float) -> tuple[list[dict], int]:
    """Random order of background events + focal event at a controlled position."""
    others = others[:]
    rng.shuffle(others)
    pos = round(frac * len(others))
    return others[:pos] + [focal] + others[pos:], pos


def build_prompt(cfg: dict, rows: list[dict]) -> str:
    p = cfg["prompt"]
    events = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    return p["template"].format(events=events, k=p["k"]).strip()


def candidate_set_size_background(seed: dict, focal: dict, stream: list[dict], level: int, max_ctx_ms: int) -> list[dict] | None:
    """Real events of the majority type nearest in time to the window end, from the past only (no future information)."""
    majority = seed["analysis_metadata"]["type_outlier"]["majority_type"]
    end = seed["time"]["end_ms"]
    pool = [e for e in stream
            if e["type"] == majority and end - max_ctx_ms <= e["session_time_ms"] <= end
            and e["event_id"] != focal["event_id"]]
    if len(pool) < level:
        return None
    pool.sort(key=lambda e: (end - e["session_time_ms"], e["event_id"]))
    return [dict(e) for e in sorted(pool[:level], key=lambda e: e["session_time_ms"])]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./dataset")
    ap.add_argument("--config", default="experiment.yaml")
    ap.add_argument("--out", default="./inputs/main_experiment_trials.jsonl")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tcfg, dcfg = cfg["trials"], cfg["density"]
    hidden_fields = frozenset(tcfg.get("hidden_fields", []))
    ds = Path(args.dataset)
    seeds = read_jsonl(ds / "seed_windows.jsonl")
    streams: dict[str, list[dict]] = defaultdict(list)
    for e in read_jsonl(ds / "primitive_events.jsonl"):
        streams[race_key_of(e)].append(e)
    all_drivers = {e["driver"] for s in streams.values() for e in s}

    trials, skipped = [], Counter()

    def emit(family, seed, focal, background, level, extra_meta):
        n = len(background) + 1
        for k, frac in enumerate(tcfg["focal_positions"]):
            key = f"{family}|{seed['window_id']}|L{level}|P{k}"
            rng = random.Random(f"{cfg['random_seed']}|{key}")
            ordered, pos = arrange(rng, focal, background, frac)
            labels = make_labels(rng, n, tcfg["label_alphabet"], tcfg["label_length"], all_drivers)
            driver_map = None
            if tcfg["driver_repr"] == "anon":
                ds_ = sorted({e["driver"] for e in ordered})
                ids = [f"D{i + 1:02d}" for i in range(len(ds_))]
                rng.shuffle(ids)
                driver_map = dict(zip(ds_, ids))
            rows = [serialize_event(e, lab, driver_map, e.get("offset_ms") if tcfg["show_offsets"] else None,
                                    hidden_fields) for e, lab in zip(ordered, labels)]
            prompt = build_prompt(cfg, rows)
            trials.append({
                "trial_id": key, "family": family, "window_id": seed["window_id"],
                "race": seed["window_id"].rsplit("_", 1)[0], "level": level, "n_events": n, "perm": k,
                "focal_position": pos, "focal_position_frac": round(pos / (n - 1), 3) if n > 1 else 0.0,
                "focal_event_id": focal["event_id"], "focal_label": labels[pos], "focal_type": focal["type"],
                "labels": dict(zip(labels, [e["event_id"] for e in ordered])),
                "types_in_order": [e["type"] for e in ordered],
                # Real attributes hidden from the model, used only for confounder analysis.
                "hidden_attrs": {lab: {"driver": e["driver"], "position": e.get("position")}
                                 for lab, e in zip(labels, ordered)},
                "driver_map": driver_map,
                "prompt": prompt, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:12],
                "meta": extra_meta,
            })

    for seed in seeds:
        evs = {e["event_id"]: e for e in seed["events"]}
        meta = seed["analysis_metadata"]
        if "type_outlier" in seed["selected_for"]:
            t = meta["type_outlier"]
            focal = evs[t["minority_event_id"]]
            emit("type_outlier", seed, focal, [e for e in seed["events"] if e is not focal],
                 seed["statistics"]["event_count"] - 1,
                 {"pattern": t["pattern"], "minority_type": t["minority_type"], "majority_type": t["majority_type"]})
            # Density: the same focal event, a controlled number of real majority-type background events.
            if dcfg["source_family"] == "type_outlier":
                for level in dcfg["background_levels"]:
                    bg = candidate_set_size_background(seed, focal, streams[seed["window_id"].rsplit("_", 1)[0]], level,
                                            int(dcfg["max_context_seconds"] * 1000))
                    if bg is None:
                        skipped[f"density_L{level}"] += 1
                        continue
                    for e in bg:  # align the format with the seed events
                        e.setdefault("offset_ms", e["session_time_ms"] - (seed["time"]["end_ms"] - seed["time"]["duration_ms"]))
                    emit("density", seed, focal, bg, level,
                         {"minority_type": t["minority_type"], "majority_type": t["majority_type"]})
        if "numerical_outlier" in seed["selected_for"]:
            nmeta = meta["numerical_outlier"]
            focal = evs[nmeta["extreme_event_id"]]
            # Only the comparison class of the focal event is shown: other event types
            # are not shown, so numerical salience is not mixed with type rarity.
            group = [evs[i] for i in nmeta["group_event_ids"] if i != focal["event_id"]]
            emit("numerical_outlier", seed, focal, group, len(group),
                 {**{k: nmeta[k] for k in ("comparison_class", "robust_z", "rel_deviation", "direction", "group_size")},
                  # Race position is not shown to the model (hidden_fields) but is stored for analysis.
                  "focal_race_position": focal.get("position")})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, trials)
    by_family = Counter(t["family"] for t in trials)
    print(json.dumps({"trials": len(trials), "by_family": by_family, "skipped": skipped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
