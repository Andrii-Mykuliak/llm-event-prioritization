"""
CLI for building the seed-window dataset.

Examples:
    python build_dataset.py --config config.yaml --output ./dataset --stage pilot
    python build_dataset.py --config config.yaml --output ./dataset --stage main
    python build_dataset.py --config config.yaml --output ./dataset --from-primitive   # without downloading again
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import yaml

from si_f1windows import (
    EXTRACTOR_VERSION, CandidateClassifier, DatasetValidator, EventStreamBuilder,
    FastF1SessionLoader, PrimitiveEventExtractor, SeedSampler, WindowGenerator, config_hash, slugify,
)


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def race_key_of(ev: dict) -> str:
    p = ev["provenance"]
    return f"{p['season']}_{slugify(p['event'])}_{p['session']}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output", default="./dataset")
    ap.add_argument("--stage", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--from-primitive", action="store_true",
                    help="rebuild windows from an existing primitive_events.jsonl")
    ap.add_argument("--sampling-seed", type=int,
                    help="override sampling.random_seed (only for sensitivity checks of the sample)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.sampling_seed is not None:
        cfg["sampling"]["random_seed"] = args.sampling_seed
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.cache).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "build.log", mode="w", encoding="utf-8"),
                                  logging.StreamHandler()])
    for noisy in ("fastf1", "urllib3", "requests_cache"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("build")

    rounds = cfg["source"][f"{args.stage}_rounds"]
    extractor = PrimitiveEventExtractor(cfg)
    fastf1_version = "n/a"

    # 1. primitive events per race
    streams: dict[str, list[dict]] = {}
    if args.from_primitive:
        by_race = defaultdict(list)
        for ev in read_jsonl(out / "primitive_events.jsonl"):
            by_race[race_key_of(ev)].append(ev)
        streams = {k: EventStreamBuilder.build(v) for k, v in by_race.items()}
        prev = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
        fastf1_version = prev.get("fastf1_version", "n/a")
    else:
        loader = FastF1SessionLoader(args.cache)
        fastf1_version = loader.version
        for rnd in rounds:
            sd = loader.load(cfg["source"]["season"], rnd, cfg["source"]["session"])
            stream = EventStreamBuilder.build(extractor.extract(sd))
            key = f"{sd.season}_{sd.slug}_{sd.session}"
            streams[key] = stream
            log.info("round %s %s: %d primitive events", rnd, sd.event_name, len(stream))

    # 2. windows + classification
    gen = WindowGenerator(cfg)
    clf = CandidateClassifier(cfg)
    validator = DatasetValidator(cfg)
    all_candidates, summary = [], []
    for key, stream in streams.items():
        windows = [clf.classify(w) for w in gen.generate(stream, key)]
        candidates = [w for w in windows if w["candidate_for"]]
        all_candidates.extend(candidates)
        summary.append({
            "race": key, "primitive_events": len(stream), "windows_min_events": len(windows),
            "candidate_windows": len(candidates),
            "type_outlier_candidates": sum("type_outlier" in w["candidate_for"] for w in candidates),
            "numerical_outlier_candidates": sum("numerical_outlier" in w["candidate_for"] for w in candidates),
            "median_events_per_window": sorted(w["statistics"]["event_count"] for w in windows)[len(windows) // 2] if windows else 0,
            "max_events_per_window": max((w["statistics"]["event_count"] for w in windows), default=0),
        })

    # 3. sampling
    seeds, report = SeedSampler(cfg).sample(all_candidates)
    for fam, r in report.items():
        level = logging.INFO if r["target_met"] else logging.WARNING
        log.log(level, "condition %s: selected %d / target %d (pool %d)%s", fam, r["selected"], r["target"],
                r["pool"], "" if r["target_met"] else "  -> TARGET NOT MET, windows are NOT duplicated (§38)")
    for row in summary:
        row["selected_seed_windows"] = sum(1 for s in seeds if s["window_id"].startswith(row["race"] + "_"))

    # 4. validation (§47–49)
    errors = {}
    for w in all_candidates:
        e = validator.validate_window(w)
        if e:
            errors[w["window_id"]] = e
    if errors:
        log.error("validation failed for %d windows: %s", len(errors), list(errors.items())[:5])
    else:
        log.info("validation passed for %d candidate windows", len(all_candidates))

    # 5. export (§43–46)
    primitive = [ev for s in streams.values() for ev in s]
    write_jsonl(out / "primitive_events.jsonl", primitive)
    write_jsonl(out / "candidate_windows.jsonl", all_candidates)
    write_jsonl(out / "seed_windows.jsonl", seeds)
    manifest = {
        "dataset_version": cfg["dataset_version"], "extractor_version": EXTRACTOR_VERSION,
        "fastf1_version": fastf1_version, "config_hash": config_hash(cfg), "stage": args.stage,
        "sessions_processed": len(streams), "races": list(streams), "primitive_events": len(primitive),
        "candidate_windows": len(all_candidates), "seed_windows": len(seeds),
        "sampling_report": report, "validation_errors": len(errors),
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    log.info("done: %s", json.dumps({k: manifest[k] for k in ("primitive_events", "candidate_windows", "seed_windows")}))


if __name__ == "__main__":
    main()
