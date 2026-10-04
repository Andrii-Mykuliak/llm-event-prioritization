"""Design diagnostics: properties of the input data that are independent of any LLM.

These tables document what the experiment can and cannot separate (for example that the numerical focal event is always
the lowest or highest value of its list, that pit events have a different representation, and how candidate-set sizes differ).
"""

from __future__ import annotations

import json
import warnings
import statistics as st
from collections import Counter

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

from conditions import CANDIDATE_SET_SIZE, CONDITION_OF_DATA_KEY, EVENT_TYPE, NUMERICAL_VALUE, condition_of
from paths import DATASET_MANIFEST, DATASET_SUMMARY, EXTRACTOR, INPUTS, MAIN_EXPERIMENT_TRIALS, FOLLOWUP_TRIALS, PRIMITIVE_EVENTS, SEED_WINDOWS
from an_data import read_jsonl

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _value(e: dict):
    return e.get("sector_time_ms") or e.get("lap_time_ms")


def numeric_seeds(seeds: list[dict]) -> pd.DataFrame:
    z_strata = yaml.safe_load(EXTRACTOR.read_text(encoding="utf-8"))["numeric_outlier"]["z_strata"]
    rows = []
    for s in seeds:
        if "numerical_outlier" not in s["selected_for"]:
            continue
        m = s["analysis_metadata"]["numerical_outlier"]
        evs = {e["event_id"]: e for e in s["events"]}
        vals = [_value(evs[i]) for i in m["group_event_ids"]]
        focal = evs[m["extreme_event_id"]]
        rows.append({"window_id": s["window_id"], "race": s["window_id"].rsplit("_", 1)[0], "comparison_class": m["comparison_class"],
                     "direction": m["direction"], "robust_z": m["robust_z"], "relative_deviation": abs(m["rel_deviation"]),
                     "abs_deviation_ms": abs(m["abs_deviation_ms"]), "group_size": m["group_size"], "lap": focal.get("lap_number"),
                     "stratum": int(sum(m["robust_z"] >= b for b in z_strata)), "focal_is_lowest": m["value_ms"] == min(vals),
                     "focal_is_highest": m["value_ms"] == max(vals)})
    return pd.DataFrame(rows)


def direction_balance(ns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("robust_z", "relative_deviation", "abs_deviation_ms", "group_size", "lap"):
        a, b = ns[ns.direction == "faster"][col], ns[ns.direction == "slower"][col]
        rows.append({"property": col, "faster_median": a.median(), "slower_median": b.median(), "test": "Mann-Whitney",
                     "p": sps.mannwhitneyu(a, b).pvalue})
    for col in ("comparison_class", "stratum", "race"):
        ct = pd.crosstab(ns.direction, ns[col])
        rows.append({"property": col, "faster_median": np.nan, "slower_median": np.nan, "test": "chi-square", "p": sps.chi2_contingency(ct)[1]})
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"property": "focal is the lowest value of its list (faster seeds)", "faster_median": ns[ns.direction == "faster"].focal_is_lowest.mean(),
                         "slower_median": np.nan, "test": "share", "p": np.nan}
    out.loc[len(out)] = {"property": "focal is the highest value of its list (slower seeds)", "faster_median": np.nan,
                         "slower_median": ns[ns.direction == "slower"].focal_is_highest.mean(), "test": "share", "p": np.nan}
    return out


def seeds_per_race(seeds: list[dict]) -> pd.DataFrame:
    rows = [{"race": s["window_id"].rsplit("_", 1)[0], "condition": CONDITION_OF_DATA_KEY[f]} for s in seeds for f in s["selected_for"]]
    t = pd.DataFrame(rows).groupby(["race", "condition"]).size().unstack(fill_value=0)
    t["total"] = t.sum(axis=1)
    return t.reset_index()


def representation(trials: list[dict]) -> pd.DataFrame:
    rows = []
    for t in trials:
        lines = [l for l in t["prompt"].splitlines() if l.startswith("{")]
        for line in lines:
            e = json.loads(line)
            rows.append({"event_type": e["type"], "keys": len(e), "characters": len(line),
                         "has_lap": "lap" in e, "has_timing_value": ("lap_time_s" in e or "sector_time_s" in e),
                         "is_focal_pit": e["id"] == t["focal_label"] and e["type"] in ("PitIn", "PitOut")})
    d = pd.DataFrame(rows)
    return d.groupby("event_type").agg(events=("keys", "size"), keys_mean=("keys", "mean"), characters_mean=("characters", "mean"),
                                       characters_min=("characters", "min"), characters_max=("characters", "max"),
                                       share_with_lap=("has_lap", "mean"), share_with_timing_value=("has_timing_value", "mean")).reset_index()


def candidate_set_size_availability(trials: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame([{"window_id": t["window_id"], "n": t["n_events"]} for t in trials if condition_of(t) == CANDIDATE_SET_SIZE]).drop_duplicates()
    t = d.groupby("n").window_id.nunique().rename("seeds").reset_index()
    t["trials"] = t.n.map(Counter(t_["n_events"] for t_ in trials if condition_of(t_) == CANDIDATE_SET_SIZE))
    return t


def candidate_set_size_composition(trials: list[dict], inputs=INPUTS) -> pd.DataFrame:
    """Age, heterogeneity and type of the retrieved background events by list size."""
    prim = {e["event_id"]: e for e in read_jsonl(inputs / PRIMITIVE_EVENTS)}
    seeds = {s["window_id"]: s for s in read_jsonl(inputs / SEED_WINDOWS)}
    rows = []
    for t in trials:
        if condition_of(t) != CANDIDATE_SET_SIZE or t["perm"] != 0:
            continue
        s = seeds[t["window_id"]]
        end = s["time"]["end_ms"]
        win_ids = {e["event_id"] for e in s["events"]}
        bg = [prim[i] for i in t["labels"].values() if i != t["focal_event_id"]]
        vals = []
        for e in bg:
            v = e.get("lap_time_ms") if e["type"] == "LapComplete" else e.get("sector_time_ms")
            vals.append(v / 1000 if v else np.nan)
        arr = np.array(vals, dtype=float)
        med = np.nanmedian(arr)
        scale = max(1.4826 * np.nanmedian(np.abs(arr - med)), 0.05)
        zs = np.abs(arr - med) / scale
        rows.append({"n_events": t["n_events"], "window_id": t["window_id"], "majority": s["analysis_metadata"]["type_outlier"]["majority_type"],
                     "share_inside_source_window": np.mean([e["event_id"] in win_ids for e in bg]),
                     "mean_age_s": np.mean([(end - e["session_time_ms"]) / 1000 for e in bg]),
                     "max_age_s": max((end - e["session_time_ms"]) / 1000 for e in bg),
                     "cv_of_timing_values": np.nanstd(arr) / np.nanmean(arr), "share_outlier_like": float(np.mean(zs >= 2.5)),
                     "share_invalid_for_numerical": 1 - np.mean([bool(e.get("numeric_valid")) for e in bg]),
                     "unique_drivers": len({e["driver"] for e in bg})})
    d = pd.DataFrame(rows)
    out = d.groupby("n_events").agg(seeds=("window_id", "nunique"), share_lap_majority=("majority", lambda x: np.mean(x == "LapComplete")),
                                    share_inside_source_window=("share_inside_source_window", "mean"), mean_age_s=("mean_age_s", "mean"),
                                    max_age_s=("max_age_s", "max"), cv_of_timing_values=("cv_of_timing_values", "mean"),
                                    share_outlier_like=("share_outlier_like", "mean"),
                                    share_invalid_for_numerical=("share_invalid_for_numerical", "mean"), unique_drivers=("unique_drivers", "mean")).reset_index()
    return out


def dataset_table(inputs=INPUTS) -> pd.DataFrame:
    man = json.loads((inputs / DATASET_MANIFEST).read_text(encoding="utf-8"))
    summ = pd.read_csv(inputs / DATASET_SUMMARY)
    trials = read_jsonl(inputs / MAIN_EXPERIMENT_TRIALS)
    followup_trials = read_jsonl(inputs / FOLLOWUP_TRIALS)
    fam = pd.Series([condition_of(t) for t in trials]).value_counts()
    rows = [("Race sessions (2025 season, rounds 4-15)", man["sessions_processed"]), ("Primitive events", man["primitive_events"]),
            ("Windows with at least five events", int(summ.windows_min_events.sum())), ("Candidate windows", man["candidate_windows"]),
            ("Seed windows (event type / numerical value)", f"{man['seed_windows']} (45 / 45)"),
            ("Main experiment presentations: event type", int(fam[EVENT_TYPE])), ("Main experiment presentations: numerical value", int(fam[NUMERICAL_VALUE])),
            ("Main experiment presentations: candidate-set size", int(fam[CANDIDATE_SET_SIZE])), ("Main experiment presentations in total", len(trials)),
            ("Controlled timing-rank follow-up source windows (eligible)", len(followup_trials) // 6), ("Controlled timing-rank follow-up presentations", len(followup_trials))]
    return pd.DataFrame(rows, columns=["component", "value"])
