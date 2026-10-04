"""
Build real seed windows from FastF1 data.

Pipeline:
    FastF1SessionLoader -> PrimitiveEventExtractor -> EventStreamBuilder -> WindowGenerator
    -> CandidateClassifier -> SeedSampler -> DatasetValidator -> DatasetExporter

This module does NOT run any LLM, build prompts, assign labels, reorder events
or change numeric values. The terms important/salient/interesting/critical
are not used in the metadata.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pandas as pd

EXTRACTOR_VERSION = "1.0.0"
EVENT_TYPES = ("SectorCross", "LapComplete", "PitIn", "PitOut")
TYPE_CODES = {"SectorCross": "SC", "LapComplete": "LAP", "PitIn": "PIN", "PitOut": "POUT"}

# Fields that must never appear in the model-visible representation.
MODEL_HIDDEN_KEYS = {
    "event_id", "minority_type", "numeric_outlier_score", "robust_z", "density_bucket",
    "candidate_for", "source_field", "provenance", "analysis_metadata", "numeric_valid",
    "session_time_ms", "track_status",
}


# --------------------------------------------------------------------------- utils

def td_to_ms(value) -> int | None:
    """pandas Timedelta/NaT -> int milliseconds or None."""
    if value is None or pd.isna(value):
        return None
    return int(round(pd.Timedelta(value).total_seconds() * 1000))


def slugify(name: str) -> str:
    name = re.sub(r"grand prix", "", name, flags=re.IGNORECASE)
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _is_true(value) -> bool:
    return bool(value) if value is not None and not (isinstance(value, float) and math.isnan(value)) else False


# --------------------------------------------------------------------------- loader

@dataclass
class SessionData:
    season: int
    round: int
    event_name: str
    slug: str
    session: str
    laps: pd.DataFrame


class FastF1SessionLoader:
    """Loads Race sessions only, without telemetry."""

    def __init__(self, cache_dir: str):
        import fastf1  # imported here so unit tests do not need network access

        self.fastf1 = fastf1
        fastf1.Cache.enable_cache(cache_dir)

    def load(self, season: int, rnd: int, session: str = "R") -> SessionData:
        if session != "R":
            raise ValueError("This version supports Race sessions only")
        s = self.fastf1.get_session(season, rnd, session)
        s.load(laps=True, telemetry=False, weather=False, messages=False)
        name = str(s.event["EventName"])
        return SessionData(season, rnd, name, slugify(name), session, s.laps.copy())

    @property
    def version(self) -> str:
        return self.fastf1.__version__


# --------------------------------------------------------------------------- extraction

class PrimitiveEventExtractor:
    """Turns lap rows into primitive events."""

    def __init__(self, cfg: dict):
        ev = cfg["events"]
        self.sectors = tuple(ev["sector_cross"]["sectors"])
        self.lap_complete = ev["lap_complete"]
        self.pit_in = ev["pit_in"]
        self.pit_out = ev["pit_out"]

    @staticmethod
    def _numeric_ok(row, value_ms, require_accurate: bool) -> bool:
        """Valid numeric value: not null, finite, not deleted, not FastF1-generated; no pit laps or out-laps."""
        if value_ms is None or value_ms <= 0:
            return False
        if _is_true(row.get("Deleted")) or _is_true(row.get("FastF1Generated")):
            return False
        if not pd.isna(row.get("PitInTime")) or not pd.isna(row.get("PitOutTime")):
            return False
        if require_accurate and not _is_true(row.get("IsAccurate")):
            return False
        return True

    def extract(self, sd: SessionData) -> list[dict]:
        events: dict[str, dict] = {}

        def add(ev_type, driver, lap, t_ms, source_field, extra, sector=None, numeric_valid=None, track_status=None):
            code = TYPE_CODES[ev_type] + (str(sector) if sector else "")
            eid = f"{sd.season}_{sd.slug}_{sd.session}_{driver}_{code}_L{lap}_{t_ms}"
            event = {"event_id": eid, "type": ev_type, "driver": driver}
            if sector is not None:
                event["sector"] = sector
            event.update(extra)
            event["session_time_ms"] = t_ms
            event["numeric_valid"] = numeric_valid
            event["provenance"] = {
                "season": sd.season, "round": sd.round, "event": sd.event_name,
                "session": sd.session, "driver": driver, "lap_number": lap,
                "source_field": source_field,
                "track_status": None if track_status is None or pd.isna(track_status) else str(track_status),
            }
            events[eid] = event  # de-duplication by stable event id

        for _, row in sd.laps.iterrows():
            driver = str(row["Driver"])
            if pd.isna(row["LapNumber"]):
                continue
            lap = int(row["LapNumber"])
            ts = row.get("TrackStatus")
            for sector in self.sectors:
                t_ms = td_to_ms(row.get(f"Sector{sector}SessionTime"))
                if t_ms is None:
                    continue
                s_ms = td_to_ms(row.get(f"Sector{sector}Time"))
                # Sector 1 of lap 1 (standing start) is not valid for the numerical experiment.
                ok = self._numeric_ok(row, s_ms, require_accurate=False) and lap > 1
                add("SectorCross", driver, lap, t_ms, f"Sector{sector}SessionTime",
                    {"lap_number": lap, "sector_time_ms": s_ms}, sector=sector, numeric_valid=ok, track_status=ts)
            if self.lap_complete:
                t_ms = td_to_ms(row.get("Time"))
                if t_ms is not None:
                    l_ms = td_to_ms(row.get("LapTime"))
                    pos = None if pd.isna(row.get("Position")) else int(row["Position"])
                    ok = self._numeric_ok(row, l_ms, require_accurate=True) and lap > 1
                    add("LapComplete", driver, lap, t_ms, "Time",
                        {"lap_number": lap, "lap_time_ms": l_ms, "position": pos}, numeric_valid=ok, track_status=ts)
            if self.pit_in:
                t_ms = td_to_ms(row.get("PitInTime"))
                if t_ms is not None:
                    add("PitIn", driver, lap, t_ms, "PitInTime", {}, track_status=ts)
            if self.pit_out:
                t_ms = td_to_ms(row.get("PitOutTime"))
                if t_ms is not None:
                    add("PitOut", driver, lap, t_ms, "PitOutTime", {}, track_status=ts)
        return list(events.values())


class EventStreamBuilder:
    """Global stream: session_time_ms ascending, then type, driver, event_id."""

    @staticmethod
    def build(events: list[dict]) -> list[dict]:
        return sorted(events, key=lambda e: (e["session_time_ms"], e["type"], e["driver"], e["event_id"]))


# --------------------------------------------------------------------------- windows

class WindowGenerator:
    """10-second trailing windows [t - 10 s, t] for every anchor event."""

    def __init__(self, cfg: dict):
        self.duration_ms = int(cfg["window"]["duration_seconds"] * 1000)
        self.min_events = int(cfg["window"]["min_events"])

    def generate(self, stream: list[dict], race_key: str) -> list[dict]:
        times = [e["session_time_ms"] for e in stream]
        seen: set[tuple] = set()
        windows = []
        for anchor in stream:
            t = anchor["session_time_ms"]
            lo = bisect.bisect_left(times, t - self.duration_ms)
            hi = bisect.bisect_right(times, t)  # events after t are not included
            members = stream[lo:hi]
            if len(members) < self.min_events:
                continue
            key = (t, tuple(e["event_id"] for e in members))
            if key in seen:  # anchors with the same t give the same window
                continue
            seen.add(key)
            windows.append(self._make_window(race_key, t, members))
        return windows

    def _make_window(self, race_key: str, end_ms: int, members: list[dict]) -> dict:
        start_ms = end_ms - self.duration_ms
        first = members[0]["provenance"]
        type_counts = Counter(e["type"] for e in members)
        visible, provenance = [], {}
        for e in members:
            v = {k: val for k, val in e.items() if k not in ("session_time_ms", "provenance", "numeric_valid")}
            v["offset_ms"] = e["session_time_ms"] - start_ms
            visible.append(v)
            provenance[e["event_id"]] = {
                **e["provenance"], "session_time_ms": e["session_time_ms"], "numeric_valid": e["numeric_valid"],
            }
        return {
            "window_id": f"{race_key}_{end_ms:08d}",
            "session": {"season": first["season"], "round": first["round"], "event": first["event"], "session": first["session"]},
            "time": {"start_ms": start_ms, "end_ms": end_ms, "duration_ms": self.duration_ms},
            "events": visible,
            "statistics": {
                "event_count": len(members),
                "unique_drivers": len({e["driver"] for e in members}),
                "type_counts": dict(sorted(type_counts.items())),
                "sector1_count": sum(1 for e in members if e["type"] == "SectorCross" and e.get("sector") == 1),
                "sector2_count": sum(1 for e in members if e["type"] == "SectorCross" and e.get("sector") == 2),
                "duration_ms": self.duration_ms,
                "events_per_second": round(len(members) / (self.duration_ms / 1000), 3),
            },
            "candidate_for": [],
            "analysis_metadata": {},
            "provenance": provenance,
        }


# --------------------------------------------------------------------------- classification

def comparison_class(ev: dict) -> str | None:
    """Comparison classes for the numerical experiment."""
    if ev["type"] == "SectorCross":
        return f"SectorCross_S{ev['sector']}"
    if ev["type"] == "LapComplete":
        return "LapComplete"
    return None


def metric_of(ev: dict) -> int | None:
    return ev.get("sector_time_ms") if ev["type"] == "SectorCross" else ev.get("lap_time_ms")


class CandidateClassifier:
    def __init__(self, cfg: dict):
        self.t_cfg = cfg["type_outlier"]
        self.n_cfg = cfg["numeric_outlier"]
        self.strata = cfg["density"]["strata"]

    # §19–21
    def type_outlier(self, window: dict) -> dict | None:
        counts = window["statistics"]["type_counts"]
        if len(counts) != 2:
            return None
        (t1, c1), (t2, c2) = sorted(counts.items(), key=lambda kv: kv[1])
        if c1 != self.t_cfg["minority_count"] or c2 < self.t_cfg["min_majority_count"]:
            return None
        minority = next(e for e in window["events"] if e["type"] == t1)
        return {
            "majority_type": t2, "majority_count": c2,
            "minority_type": t1, "minority_event_id": minority["event_id"],
            "pattern": f"{t2}x{c2}+{t1}",
            "preferred_pattern": t1 in self.t_cfg["preferred_minority_types"],
        }

    # §23–28
    def numerical_outlier(self, window: dict) -> dict | None:
        groups: dict[str, list[dict]] = defaultdict(list)
        for e in window["events"]:
            cls = comparison_class(e)
            if cls and window["provenance"][e["event_id"]]["numeric_valid"] and metric_of(e):
                groups[cls].append(e)
        extremes = []
        for cls, members in groups.items():
            if len(members) < self.n_cfg["min_group_size"]:
                continue
            values = [metric_of(e) for e in members]
            med = _median(values)
            mad = _median([abs(v - med) for v in values])
            scale = max(self.n_cfg["mad_scale"] * mad, self.n_cfg["mad_floor_ms"])
            for e, v in zip(members, values):
                z = abs(v - med) / scale
                rel = abs(v - med) / med if med else 0.0
                if z >= self.n_cfg["threshold"] and rel >= self.n_cfg.get("min_rel_deviation", 0.0):
                    extremes.append((cls, e, v, med, mad, z, [m["event_id"] for m in members]))
        if len(extremes) != 1:  # exactly one extreme candidate per window
            return None
        cls, e, v, med, mad, z, group_ids = extremes[0]
        return {
            "comparison_class": cls, "group_size": len(group_ids), "group_event_ids": group_ids,
            "extreme_event_id": e["event_id"], "value_ms": v, "median_ms": med,
            "abs_deviation_ms": abs(v - med), "rel_deviation": round((v - med) / med, 5),
            "mad_ms": mad, "robust_z": round(z, 3), "direction": "slower" if v > med else "faster",
        }

    def density_bucket(self, n: int) -> str:
        for name, (lo, hi) in self.strata.items():
            if n >= lo and (hi is None or n <= hi):
                return name
        return "below_min"

    def classify(self, window: dict) -> dict:
        meta = {"density_bucket": self.density_bucket(window["statistics"]["event_count"])}
        t = self.type_outlier(window)
        if t:
            window["candidate_for"].append("type_outlier")
            meta["type_outlier"] = t
        n = self.numerical_outlier(window)
        if n:
            window["candidate_for"].append("numerical_outlier")
            meta["numerical_outlier"] = n
        if t or n:
            # density is flagged for windows where the focal event is already defined.
            window["candidate_for"].append("density")
        window["analysis_metadata"] = meta
        return window


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# --------------------------------------------------------------------------- sampling

class SeedSampler:
    """Deterministic sampling without event reuse within an experimental condition."""

    FAMILIES = ("type_outlier", "numerical_outlier")

    def __init__(self, cfg: dict):
        s = cfg["sampling"]
        self.seed = s["random_seed"]
        self.max_per_race = s["max_windows_per_race"]
        self.forbid_reuse = s["forbid_event_reuse"]
        self.target = s["target_per_family"]

        self.z_strata = cfg["numeric_outlier"].get("z_strata")

    def _priority(self, family: str, w: dict) -> int:
        if family == "type_outlier":
            return 0 if w["analysis_metadata"]["type_outlier"]["preferred_pattern"] else 1
        return 0

    def _stratify(self, family: str, pool: list[dict]) -> list[dict]:
        """Round-robin over robust-z strata, to obtain a response curve and not only strong outliers."""
        if family != "numerical_outlier" or not self.z_strata:
            return pool
        buckets: dict[int, list[dict]] = defaultdict(list)
        for w in pool:
            z = w["analysis_metadata"]["numerical_outlier"]["robust_z"]
            buckets[sum(z >= b for b in self.z_strata)].append(w)
        order = sorted(buckets)
        out, i = [], 0
        while any(buckets[b] for b in order):
            b = order[i % len(order)]
            if buckets[b]:
                out.append(buckets[b].pop(0))
            i += 1
        return out

    def sample(self, candidates: list[dict]) -> tuple[list[dict], dict]:
        selected: dict[str, dict] = {}
        report = {}
        for family in self.FAMILIES:
            pool = sorted((w for w in candidates if family in w["candidate_for"]), key=lambda w: w["window_id"])
            rng = random.Random(f"{self.seed}:{family}")
            rng.shuffle(pool)
            pool.sort(key=lambda w: self._priority(family, w))  # stable sort keeps the shuffle
            pool = self._stratify(family, pool)
            used_events: set[str] = set()
            per_race: Counter = Counter()
            chosen = []
            for w in pool:
                race = w["window_id"].rsplit("_", 1)[0]
                ids = {e["event_id"] for e in w["events"]}
                if per_race[race] >= self.max_per_race:
                    continue
                if self.forbid_reuse and ids & used_events:
                    continue
                chosen.append(w)
                used_events |= ids
                per_race[race] += 1
                if len(chosen) >= self.target:
                    break
            report[family] = {"selected": len(chosen), "target": self.target, "pool": len(pool),
                              "target_met": len(chosen) >= self.target}
            for w in chosen:
                entry = selected.setdefault(w["window_id"], json.loads(json.dumps(w)))
                entry.setdefault("selected_for", []).append(family)
        seeds = sorted(selected.values(), key=lambda w: w["window_id"])
        return seeds, report


# --------------------------------------------------------------------------- validation

class DatasetValidator:
    def __init__(self, cfg: dict):
        self.duration_ms = int(cfg["window"]["duration_seconds"] * 1000)
        self.cfg = cfg

    def validate_window(self, w: dict) -> list[str]:
        errs = []
        start, end = w["time"]["start_ms"], w["time"]["end_ms"]
        if end - start != self.duration_ms:
            errs.append("duration mismatch")
        ids = [e["event_id"] for e in w["events"]]
        if len(ids) != len(set(ids)):
            errs.append("duplicate event ids")
        abs_t = [w["provenance"][i]["session_time_ms"] for i in ids]
        if any(not (start <= t <= end) for t in abs_t):
            errs.append("event outside window / future event")
        if abs_t != sorted(abs_t):
            errs.append("events not sorted by real time")
        if any(e["type"] not in EVENT_TYPES for e in w["events"]):
            errs.append("invalid event type")
        if _contains_nan(w):
            errs.append("NaN in window")
        meta = w["analysis_metadata"]
        if "type_outlier" in meta:
            counts = w["statistics"]["type_counts"]
            c = sorted(counts.values())
            if not (len(counts) == 2 and c[0] == 1 and c[1] >= self.cfg["type_outlier"]["min_majority_count"]):
                errs.append("type_outlier rule violated")
        if "numerical_outlier" in meta:
            n = meta["numerical_outlier"]
            evs = {e["event_id"]: e for e in w["events"]}
            group = [evs[i] for i in n["group_event_ids"]]
            classes = {comparison_class(e) for e in group}
            if len(group) < self.cfg["numeric_outlier"]["min_group_size"]:
                errs.append("numerical group too small")
            if len(classes) != 1:
                errs.append("numerical group mixes metrics/sectors")
            if any(metric_of(e) is None or not math.isfinite(metric_of(e)) for e in group):
                errs.append("non-finite numerical value")
        return errs


def _contains_nan(obj) -> bool:
    if isinstance(obj, float):
        return math.isnan(obj) or math.isinf(obj)
    if isinstance(obj, dict):
        return any(_contains_nan(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_nan(v) for v in obj)
    return False


# --------------------------------------------------------------------------- model-visible view

def model_visible_events(window: dict) -> list[dict]:
    """Events without service fields. Used by the downstream experiment module."""
    return [{k: v for k, v in e.items() if k not in MODEL_HIDDEN_KEYS} for e in window["events"]]


@dataclass
class BuildResult:
    primitive_events: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    seeds: list[dict] = field(default_factory=list)
    summary_rows: list[dict] = field(default_factory=list)
    sampling_report: dict = field(default_factory=dict)
    validation_errors: dict = field(default_factory=dict)
