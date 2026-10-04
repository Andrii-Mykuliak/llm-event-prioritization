"""Controlled timing-rank follow-up statistics.

Each source window is shown six times: the focal timing value is kept (original), made faster than every background
value, or made slower, and the focal event is placed first or last. Effects are paired contrasts over source windows
with a 95% percentile bootstrap (B = 5000). The bootstrap is vectorised; with frozen_random_stream=True the random stream is the one
of the frozen analysis, so the tables equal the frozen output_rank tables exactly.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from paths import INPUTS, FOLLOWUP_EXCLUSIONS

CONTENT = ["original", "faster", "slower"]
POS = ["first", "last"]
CELLS = [(c, p) for c in CONTENT for p in POS]
B = 5000
EFFECT_SEED, DESC_SEED = 31, 5


def valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.format_ok]


def cell_array(d: pd.DataFrame, y: str) -> np.ndarray:
    tab = valid(d).pivot_table(index="src", columns=["content", "position"], values=y, aggfunc="mean")
    return tab.reindex(columns=pd.MultiIndex.from_tuples(CELLS)).to_numpy(dtype=float)


def _mean(v: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(v, axis=axis)


def contrasts(m: np.ndarray) -> dict[str, np.ndarray]:
    """m[..., 6] cell means in CELLS order -> every contrast of the protocol."""
    def g(c, p):
        return m[..., CELLS.index((c, p))]

    e: dict[str, np.ndarray] = {}
    for c in CONTENT:
        e[f"last-first | {c}"] = g(c, "last") - g(c, "first")
    e["last-first | mean"] = _mean(np.stack([e[f"last-first | {c}"] for c in CONTENT]), 0)
    for a, b in (("faster", "original"), ("slower", "original"), ("faster", "slower")):
        for p in POS:
            e[f"{a}-{b} | {p}"] = g(a, p) - g(b, p)
        e[f"{a}-{b} | mean"] = _mean(np.stack([e[f"{a}-{b} | {p}"] for p in POS]), 0)
    e["interaction: (last-first|faster) - (last-first|original)"] = e["last-first | faster"] - e["last-first | original"]
    e["interaction: (last-first|slower) - (last-first|original)"] = e["last-first | slower"] - e["last-first | original"]
    return e


def effects(d: pd.DataFrame, y: str, label: str, rng: np.random.Generator) -> pd.DataFrame:
    v = cell_array(d, y)
    point = contrasts(_mean(v, 0))
    idx = rng.integers(0, len(v), (B, len(v)))
    boot = contrasts(_mean(v[idx], 1))
    rows = []
    for k, val in point.items():
        bs = boot[k]
        rows.append({"group": label, "outcome": y, "effect": k, "pp": 100 * float(val), "ci_lo_pp": 100 * float(np.nanpercentile(bs, 2.5)),
                     "ci_hi_pp": 100 * float(np.nanpercentile(bs, 97.5)), "n_sources": len(v)})
    return pd.DataFrame(rows)


def _seed_for(name: str, base: int) -> int:
    return base * 1_000_003 + sum((i + 1) * ord(ch) for i, ch in enumerate(name))


def descriptive(df: pd.DataFrame, frozen_random_stream: bool = False) -> pd.DataFrame:
    shared = np.random.default_rng(DESC_SEED)
    rows = []
    for (m, c, p), g in df.groupby(["llm", "content", "position"]):
        rng = shared if frozen_random_stream else np.random.default_rng(_seed_for(f"{m}|{c}|{p}", DESC_SEED))
        ok = valid(g)
        r = {"llm": m, "label": g.label.iloc[0], "content": c, "position": p, "N": len(g), "N_valid": len(ok)}
        per = ok.groupby("src")[["top1", "top3"]].mean()
        for y in ("top1", "top3"):
            bs = per[y].to_numpy()[rng.integers(0, len(per), (B, len(per)))].mean(1)
            r.update({f"{y}_pct": 100 * ok[y].mean(), f"{y}_ci_lo": 100 * np.percentile(bs, 2.5), f"{y}_ci_hi": 100 * np.percentile(bs, 97.5)})
        r["chance3_pct"] = 100 * ok.chance3.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def all_effects(df: pd.DataFrame, order: list[str], frozen_random_stream: bool = False) -> pd.DataFrame:
    shared = np.random.default_rng(EFFECT_SEED)

    def rng_for(tag):
        return shared if frozen_random_stream else np.random.default_rng(_seed_for(tag, EFFECT_SEED))

    parts = []
    for m in order:
        if m not in set(df.llm):
            continue
        for y in ("top3", "top1"):
            parts.append(effects(df[df.llm == m], y, m, rng_for(f"{m}|{y}")))
    for m in order:
        if m not in set(df.llm):
            continue
        for d in ("faster", "slower"):
            parts.append(effects(df[(df.llm == m) & (df.orig_dir == d)], "top3", f"{m} | original {d}", rng_for(f"{m}|{d}")))
    return pd.concat(parts, ignore_index=True)


def quality(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("llm")[["format_ok", "invented", "driver_answer"]].mean().reset_index()


def _clean(t: str) -> str:
    import re
    t = re.sub(r"C\(content, Treatment\('original'\)\)\[T\.(\w+)\]", r"content[\1]", t)
    t = re.sub(r"C\(position, Treatment\('first'\)\)\[T\.(\w+)\]", r"position[\1]", t)
    t = re.sub(r"C\(llm, Treatment\('[^']*'\)\)\[T\.([^\]]*)\]", r"llm[\1]", t)
    return t


def gee_followup(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    ok = valid(df)
    multi = ok.llm.nunique() > 1
    base = "top3 ~ C(content, Treatment('original')) * C(position, Treatment('first'))"
    specs = [("additive_llm", base + (f" + C(llm, Treatment('{order[0]}'))" if multi else ""))]
    if multi:
        specs.append(("full_llm_interaction", base.replace(" * ", " * ") + f" * C(llm, Treatment('{order[0]}'))"))
    rows = []
    for name, f in specs:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                res = smf.gee(f, "src", ok, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            except Exception as e:  # noqa: BLE001
                rows.append({"spec": name, "term": f"GEE failed: {e}"[:90]})
                continue
            ci = res.conf_int()
            tab = pd.DataFrame({"OR": np.exp(res.params), "ci_lo": np.exp(ci[0]), "ci_hi": np.exp(ci[1]), "p": res.pvalues})
            unstable = (not np.isfinite(tab.values).all()) or (tab.OR > 1e4).any() or (tab.OR < 1e-4).any() or any("overflow" in str(x.message) for x in w)
        for term, r in tab.iterrows():
            rows.append({"spec": name, "term": _clean(term), "OR": r.OR, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "p": r.p, "n": len(ok),
                         "clusters": ok.src.nunique(), "unstable_do_not_interpret": bool(unstable)})
    return pd.DataFrame(rows)


def crossover(desc: pd.DataFrame) -> pd.DataFrame:
    return desc[["llm", "label", "content", "position", "top3_pct", "top3_ci_lo", "top3_ci_hi", "top1_pct", "chance3_pct"]]


def exclusions(inputs: Path = INPUTS) -> pd.DataFrame:
    p = Path(inputs) / FOLLOWUP_EXCLUSIONS
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data["excluded"] if isinstance(data, dict) and "excluded" in data else data
    return pd.DataFrame(items)


def frozen_reference_check(desc: pd.DataFrame, eff: pd.DataFrame, frozen_dir: Path) -> dict:
    """Compare against the frozen follow-up descriptive and marginal-effect tables (model names qwen3:xb are mapped to run names)."""
    names = {"qwen3:4b-q4_K_M": "qwen3-4b-q4_K_M", "qwen3:8b": "qwen3-8b", "qwen3:14b": "qwen3-14b"}
    f1 = pd.read_csv(Path(frozen_dir) / "controlled_timing_rank_followup_frozen_descriptive.csv")
    f1["llm"] = f1.model.map(names)
    m = desc.merge(f1, on=["llm", "content", "position"], suffixes=("", "_f"))
    d1 = max((m[c] - m[c + "_f"]).abs().max() for c in ("top1_pct", "top1_ci_lo", "top1_ci_hi", "top3_pct", "top3_ci_lo", "top3_ci_hi"))
    f2 = pd.read_csv(Path(frozen_dir) / "controlled_timing_rank_followup_frozen_marginal_effects_pp.csv")
    f2["group"] = f2.group.map(lambda g: names[g.split(" | ", 1)[0]] + (" | " + g.split(" | ", 1)[1] if " | " in g else ""))
    e = eff.merge(f2, on=["group", "outcome", "effect"], suffixes=("", "_f"))
    followup_df = max((e[c] - e[c + "_f"]).abs().max() for c in ("pp", "ci_lo_pp", "ci_hi_pp"))
    return {"descriptive_rows": len(m), "descriptive_max_abs_diff_pp": float(d1), "effect_rows": len(e), "effect_max_abs_diff_pp": float(followup_df)}
