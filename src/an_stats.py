"""Main-experiment statistics. Every function takes the table from data.load_main_experiment() and returns DataFrames.

Conventions (identical to the manuscript):
  * rates use format-valid responses only; denominators of all-trial rates are given separately
  * confidence intervals of rates: 95% percentile cluster bootstrap, clusters = seed windows, 2,000 resamples, seed 7
  * models: logistic GEE, exchangeable correlation, clusters = seed windows; position reference = middle positions
    (1/4, 1/2, 3/4); LLM reference = the first run in size order
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy import stats as sps

from conditions import CANDIDATE_SET_SIZE, CONDITIONS, EVENT_TYPE, NUMERICAL_VALUE
warnings.filterwarnings("ignore")


def valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.format_ok]


def cluster_boot(d: pd.DataFrame, col: str, cluster: str = "window_id", B: int = 2000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in d.groupby(cluster)]
    if len(groups) < 2:
        return (np.nan, np.nan)
    stats = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        stats.append(np.concatenate([groups[i] for i in pick]).mean())
    return tuple(np.percentile(stats, [2.5, 97.5]))


# --------------------------------------------------------------------------- descriptive tables

def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, llm), g in valid(df).groupby(["condition", "llm"]):
        lo1, hi1 = cluster_boot(g, "focal_top1")
        lo3, hi3 = cluster_boot(g, "focal_top3")
        rows.append({"condition": fam, "llm": llm, "label": g.label.iloc[0], "trials": len(g), "seeds": g.window_id.nunique(),
                     "top1": g.focal_top1.mean(), "top1_ci_lo": lo1, "top1_ci_hi": hi1, "top1_chance": g.chance_top1.mean(),
                     "top1_lift": g.focal_top1.mean() / g.chance_top1.mean(),
                     "top3": g.focal_top3.mean(), "top3_ci_lo": lo3, "top3_ci_hi": hi3, "top3_chance": g.chance_top3.mean(),
                     "top3_lift": g.focal_top3.mean() / g.chance_top3.mean()})
    return pd.DataFrame(rows)


def direction_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    num = valid(df[df.condition == NUMERICAL_VALUE])
    for (llm, d), g in num.groupby(["llm", "direction"]):
        lo, hi = cluster_boot(g, "focal_top1")
        rows.append({"llm": llm, "label": g.label.iloc[0], "direction": d, "n": len(g), "top1": g.focal_top1.mean(), "top1_ci_lo": lo,
                     "top1_ci_hi": hi, "top1_chance": g.chance_top1.mean(), "top3": g.focal_top3.mean(), "top3_chance": g.chance_top3.mean()})
    return pd.DataFrame(rows)


def position_curve(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, llm, p), g in valid(df).groupby(["condition", "llm", "pos5"]):
        lo, hi = cluster_boot(g, "focal_top3")
        rows.append({"condition": fam, "llm": llm, "label": g.label.iloc[0], "pos": p, "n": len(g), "top3": g.focal_top3.mean(),
                     "ci_lo": lo, "ci_hi": hi, "chance": g.chance_top3.mean()})
    return pd.DataFrame(rows)


def candidate_set_size_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dens = valid(df[df.condition == CANDIDATE_SET_SIZE])
    for (llm, n), g in dens.groupby(["llm", "n_events"]):
        lo, hi = cluster_boot(g, "focal_top3")
        ch = g.chance_top3.mean()
        rows.append({"llm": llm, "label": g.label.iloc[0], "n_events": int(n), "trials": len(g), "seeds": g.window_id.nunique(),
                     "top3": g.focal_top3.mean(), "ci_lo": lo, "ci_hi": hi, "chance": ch, "lift": g.focal_top3.mean() / ch,
                     "lift_lo": lo / ch, "lift_hi": hi / ch})
    return pd.DataFrame(rows)


def candidate_set_size_complete_seeds(df: pd.DataFrame, trials_design: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidate-set size restricted to the seeds that have all three list sizes (same seeds at every level)."""
    dens = df[df.condition == CANDIDATE_SET_SIZE]
    levels = dens.groupby("window_id").n_events.apply(set)
    complete = [w for w, s in levels.items() if s == {5, 10, 20}]
    dc = valid(dens[dens.window_id.isin(complete)])
    rows, ln_rows = [], []
    for llm, g0 in dc.groupby("llm"):
        for n, g in g0.groupby("n_events"):
            rows.append({"llm": llm, "label": g.label.iloc[0], "n_events": int(n), "top3": g.focal_top3.mean(), "chance": g.chance_top3.mean(),
                         "lift": g.focal_top3.mean() / g.chance_top3.mean(), "trials": len(g), "seeds": len(complete)})
        try:
            res = smf.gee("y3 ~ C(pos, Treatment('middle')) + log_n", "window_id", g0, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
            ci = res.conf_int()
            ln_rows.append({"llm": llm, "label": g0.label.iloc[0], "OR_log_n": np.exp(res.params["log_n"]), "ci_lo": np.exp(ci.loc["log_n", 0]),
                            "ci_hi": np.exp(ci.loc["log_n", 1]), "p": res.pvalues["log_n"], "seeds": len(complete)})
        except Exception:
            ln_rows.append({"llm": llm, "label": g0.label.iloc[0], "OR_log_n": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "seeds": len(complete)})
    return pd.DataFrame(rows), pd.DataFrame(ln_rows)


def pit_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for llm, g in valid(df[df.condition == EVENT_TYPE]).groupby("llm"):
        lo, hi = cluster_boot(g, "focal_top3")
        rows.append({"llm": llm, "label": g.label.iloc[0], "n": len(g), "top3": g.focal_top3.mean(), "ci_lo": lo, "ci_hi": hi,
                     "chance": g.chance_top3.mean(), "top1": g.focal_top1.mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- GEE models

def _formula(df: pd.DataFrame, fam: str, order: list[str], direction: bool, outcome: str = "y3") -> str:
    multi = df.llm.nunique() > 1
    f = f"{outcome} ~ " + (f"C(llm, Treatment('{order[0]}')) + " if multi else "") + "C(pos, Treatment('middle')) + log_n"
    if fam == NUMERICAL_VALUE:
        f += " + log_z"
        if direction:
            f += " + slower"
    return f


def clean_term(t: str) -> str:
    t = t.replace("C(pos, Treatment('middle'))", "pos")
    if "C(llm," in t:
        import re
        t = re.sub(r"C\(llm, Treatment\('[^']*'\)\)\[T\.([^\]]*)\]", r"llm[\1]", t)
    return t


def gee_models(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Pooled GEE per condition (numerical both without and with the direction term)."""
    d = valid(df)
    rows = []
    specs = [(EVENT_TYPE, "canonical"), (NUMERICAL_VALUE, "canonical"), (NUMERICAL_VALUE, "with_direction"), (CANDIDATE_SET_SIZE, "canonical")]
    for fam, variant in specs:
        g = d[d.condition == fam]
        if len(g) < 30:
            continue
        f = _formula(g, fam, order, direction=variant == "with_direction")
        try:
            res = smf.gee(f, "window_id", g, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        except Exception as e:  # noqa: BLE001
            rows.append({"condition": fam, "spec": variant, "term": f"GEE failed: {e}"[:80]})
            continue
        ci = res.conf_int()
        for term in res.params.index:
            rows.append({"condition": fam, "spec": variant, "term": clean_term(term), "OR": np.exp(res.params[term]), "ci_lo": np.exp(ci.loc[term, 0]),
                         "ci_hi": np.exp(ci.loc[term, 1]), "p": res.pvalues[term], "n": len(g), "clusters": g.window_id.nunique(),
                         "working_correlation": float(np.ravel(res.cov_struct.dep_params)[0]), "formula": f})
    return pd.DataFrame(rows)


def position_by_llm(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Position effects fitted separately for each LLM and the joint Position x LLM interaction test."""
    d = valid(df)
    rows, inter = [], []
    for fam in CONDITIONS:
        for llm, g in d[d.condition == fam].groupby("llm"):
            f = "y3 ~ C(pos, Treatment('middle')) + log_n" + (" + log_z + slower" if fam == NUMERICAL_VALUE else "")
            try:
                res = smf.gee(f, "window_id", g, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
                ci = res.conf_int()
                for pos in ("first", "last"):
                    t = f"C(pos, Treatment('middle'))[T.{pos}]"
                    rows.append({"condition": fam, "llm": llm, "label": g.label.iloc[0], "pos": pos, "OR": np.exp(res.params[t]), "ci_lo": np.exp(ci.loc[t, 0]),
                                 "ci_hi": np.exp(ci.loc[t, 1]), "p": res.pvalues[t], "n": len(g)})
            except Exception:
                for pos in ("first", "last"):
                    rows.append({"condition": fam, "llm": llm, "label": g.label.iloc[0], "pos": pos, "OR": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "n": len(g)})
        gg = d[d.condition == fam]
        if gg.llm.nunique() < 2:
            continue
        order = list(gg.groupby("llm").size().index)
        full = (f"y3 ~ C(llm, Treatment('{order[0]}')) * C(pos, Treatment('middle')) + log_n"
                + (" + log_z + slower" if fam == NUMERICAL_VALUE else ""))
        try:
            res = smf.gee(full, "window_id", gg, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            names = [n for n in res.params.index if "C(llm" in n and "C(pos" in n]
            R = np.zeros((len(names), len(res.params)))
            for i, n in enumerate(names):
                R[i, list(res.params.index).index(n)] = 1
            w = res.wald_test(R, scalar=True)
            inter.append({"condition": fam, "wald_stat": float(w.statistic), "df": len(names), "p": float(w.pvalue)})
        except Exception:
            inter.append({"condition": fam, "wald_stat": np.nan, "df": np.nan, "p": np.nan})
    return pd.DataFrame(rows), pd.DataFrame(inter)


# --------------------------------------------------------------------------- estimator comparison

def _fit_three(formula: str, d: pd.DataFrame) -> pd.DataFrame:
    y, X = dmatrices(formula, d, return_type="dataframe")
    groups = pd.factorize(d.loc[y.index, "window_id"])[0]
    A = sm.Logit(y, X).fit(disp=0, maxiter=200)
    B = sm.Logit(y, X).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    C = sm.GEE(y, X, groups=groups, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    rows = []
    for name in X.columns:
        for tag, res in (("naive logit", A), ("cluster-robust logit", B), ("GEE exchangeable", C)):
            b, se = float(res.params[name]), float(res.bse[name])
            rows.append({"term": clean_term(name), "estimator": tag, "beta": b, "se": se, "OR": np.exp(b), "ci_lo": np.exp(b - 1.96 * se),
                         "ci_hi": np.exp(b + 1.96 * se), "p": float(res.pvalues[name])})
    return pd.DataFrame(rows)


_SPEC_SORT_KEY = {CANDIDATE_SET_SIZE: 0, NUMERICAL_VALUE: 1, NUMERICAL_VALUE + " (with direction)": 2, EVENT_TYPE: 3}


def estimator_comparison(df: pd.DataFrame, order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Naive logit vs seed-clustered robust logit vs GEE on the same design matrix. Returns (long table, summary)."""
    d = valid(df)
    long_rows = []
    for fam, variant in ((EVENT_TYPE, False), (NUMERICAL_VALUE, False), (NUMERICAL_VALUE, True), (CANDIDATE_SET_SIZE, False)):
        g = d[d.condition == fam]
        if len(g) < 30:
            continue
        spec = fam + (" (with direction)" if variant else "")
        try:
            t = _fit_three(_formula(g, fam, order, direction=variant), g)
        except Exception:
            continue
        t.insert(0, "spec", spec)
        long_rows.append(t)
    if not long_rows:
        return pd.DataFrame(), pd.DataFrame()
    long = pd.concat(long_rows, ignore_index=True)
    wide = long[long.term != "Intercept"].pivot_table(index=["spec", "term"], columns="estimator", values=["beta", "se", "p", "OR"]).reset_index()
    wide.columns = ["_".join([c for c in col if c]) for col in wide.columns]
    spec_order = {s: i for i, s in enumerate(long.spec.drop_duplicates().sort_values(key=lambda s: s.map(_SPEC_SORT_KEY)))}
    wide = wide.assign(_spec_order=wide.spec.map(spec_order)).sort_values(["_spec_order", "term"], kind="stable").drop(columns="_spec_order").reset_index(drop=True)
    wide["dOR_pct_naive_vs_gee"] = 100 * (wide["OR_naive logit"] - wide["OR_GEE exchangeable"]) / wide["OR_GEE exchangeable"]
    wide["dOR_pct_cluster_vs_gee"] = 100 * (wide["OR_cluster-robust logit"] - wide["OR_GEE exchangeable"]) / wide["OR_GEE exchangeable"]
    wide["SE_ratio_cluster_vs_naive"] = wide["se_cluster-robust logit"] / wide["se_naive logit"]
    wide["SE_ratio_gee_vs_naive"] = wide["se_GEE exchangeable"] / wide["se_naive logit"]
    wide["sig_naive"] = wide["p_naive logit"] < 0.05
    wide["sig_cluster"] = wide["p_cluster-robust logit"] < 0.05
    wide["sig_gee"] = wide["p_GEE exchangeable"] < 0.05
    wide["inference_changes"] = ~((wide.sig_naive == wide.sig_cluster) & (wide.sig_cluster == wide.sig_gee))
    wide["core_term"] = wide.term.str.startswith(("pos[", "llm[")) | (wide.term == "slower")
    return long, wide


def estimator_headline(wide: pd.DataFrame) -> dict:
    if wide.empty:
        return {}
    core = wide[wide.core_term]
    se_terms = wide[wide.term.isin(["log_z", "log_n", "slower"]) & wide.spec.str.startswith("numerical")]
    return {
        "max_abs_dOR_pct_naive_vs_gee_core": float(core.dOR_pct_naive_vs_gee.abs().max()),
        "max_abs_dOR_pct_cluster_vs_gee_core": float(core.dOR_pct_cluster_vs_gee.abs().max()),
        "sign_disagreements_core": int(((np.sign(core["beta_naive logit"]) != np.sign(core["beta_GEE exchangeable"]))).sum()),
        "se_ratio_seed_level_covariates_min": float(se_terms.SE_ratio_cluster_vs_naive.min()) if len(se_terms) else np.nan,
        "se_ratio_seed_level_covariates_max": float(se_terms.SE_ratio_cluster_vs_naive.max()) if len(se_terms) else np.nan,
        "se_ratio_position_min": float(core[core.term.str.startswith("pos[")].SE_ratio_cluster_vs_naive.min()),
        "se_ratio_position_max": float(core[core.term.str.startswith("pos[")].SE_ratio_cluster_vs_naive.max()),
        "inference_changes": [f"{r.spec}: {r.term}" for r in wide[wide.inference_changes].itertuples()],
    }


# --------------------------------------------------------------------------- quality, denominators, parser, min-value

def top1_agreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, llm, win, lvl), g in valid(df).groupby(["condition", "llm", "window_id", "level"]):
        if len(g) < 2:
            continue
        top1 = g.top1_event.value_counts().iloc[0] / len(g)
        sets = [set(s) for s in g.top3_events]
        jac = np.mean([len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)])
        rows.append({"condition": fam, "llm": llm, "top1_agreement": top1, "top3_jaccard": jac})
    return pd.DataFrame(rows).groupby(["condition", "llm"])[["top1_agreement", "top3_jaccard"]].mean().reset_index()


def quality_table(df: pd.DataFrame) -> pd.DataFrame:
    stab = top1_agreement(df)
    rows = []
    for llm, g in df.groupby("llm"):
        s = stab[stab.llm == llm].top1_agreement
        rows.append({"llm": llm, "label": g.label.iloc[0], "responses": len(g), "valid": int(g.format_ok.sum()), "valid_rate": g.format_ok.mean(),
                     "invalid": int((~g.format_ok).sum()), "responses_with_invented_ids": int(g.invented.sum()),
                     "top1_agreement_min": s.min() if len(s) else np.nan, "top1_agreement_max": s.max() if len(s) else np.nan,
                     "mean_latency_s": g.latency_s.mean(), "max_output_tokens": g.output_tokens.max(), "max_prompt_tokens": g.prompt_tokens.max()})
    return pd.DataFrame(rows)


def denominators(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, llm), g in df.groupby(["condition", "llm"]):
        v = g[g.format_ok]
        rows.append({"condition": fam, "llm": llm, "label": g.label.iloc[0], "n_all": len(g), "n_valid": len(v),
                     "top1_valid": v.focal_top1.mean(), "top1_all": g.focal_top1.mean(),
                     "top3_valid": v.focal_top3.mean(), "top3_all": g.focal_top3.mean(),
                     "lift3_valid": v.focal_top3.mean() / v.chance_top3.mean(), "lift3_all": g.focal_top3.mean() / g.chance_top3.mean()})
    return pd.DataFrame(rows)


def parser_audit(df: pd.DataFrame, store) -> pd.DataFrame:
    """Independent re-parse of every raw response against the recorded parse."""
    import re
    strict = re.compile(r"^\s*([A-Z]{3})\s*,\s*([A-Z]{3})\s*,\s*([A-Z]{3})\s*\.?\s*$")
    rows = []
    for llm, g in df.groupby("llm"):
        mism = strict_ok = extra = more = inv_valid = 0
        reasons = {"invented_ids": 0, "fewer_than_3": 0}
        for r in g.itertuples():
            labels = set(store.trials[r.trial_id]["labels"])
            txt = re.sub(r"<think>.*?</think>", "", r.raw, flags=re.S)
            found = []
            for tok in re.findall(r"[A-Z]{2,}", txt.upper()):
                if tok in labels and tok not in found:
                    found.append(tok)
            ok = len(found[:3]) == 3
            mism += (ok != r.format_ok) or (found[:3] != list(r.parsed))
            m = strict.match(txt)
            s_ok = bool(m) and len(set(m.groups())) == 3 and all(x in labels for x in m.groups())
            strict_ok += s_ok
            extra += ok and not s_ok
            more += len(found) > 3
            inv = [x for x in re.findall(r"\b[A-Z]{3}\b", txt) if x not in labels]
            inv_valid += bool(inv) and ok
            if not ok:
                reasons["invented_ids" if inv else "fewer_than_3"] += 1
        rows.append({"llm": llm, "label": g.label.iloc[0], "responses": len(g), "recorded_vs_independent_mismatches": mism,
                     "exact_format_A,B,C": strict_ok, "valid_with_extra_text": extra, "more_than_3_valid_labels": more,
                     "valid_but_contains_invented_id": inv_valid, "invalid_due_to_invented_id": reasons["invented_ids"],
                     "invalid_other": reasons["fewer_than_3"]})
    return pd.DataFrame(rows)


def min_value_preference(df: pd.DataFrame, store) -> pd.DataFrame:
    """How often the first-ranked event is the lowest displayed timing value (numerical, event-type and candidate-set-size conditions)."""
    rows = []
    for r in valid(df).itertuples():
        evs = store.rows(r.trial_id)
        vals = {e["id"]: store.value_of(e) for e in evs}
        numeric = {k: v for k, v in vals.items() if v is not None}
        if not numeric:
            continue
        lo = min(numeric, key=numeric.get)
        first = r.parsed[0]
        rows.append({"llm": r.llm, "label": r.label, "condition": r.condition, "direction": r.direction if r.condition == NUMERICAL_VALUE else "",
                     "first_is_lowest": first == lo, "lowest_in_top3": lo in r.parsed[:3], "first_is_valueless": first not in numeric,
                     "focal_is_lowest": (store.trials[r.trial_id]["focal_label"] == lo), "chance_first": 1 / len(vals)})
    t = pd.DataFrame(rows)
    return t.groupby(["condition", "direction", "llm", "label"]).agg(
        trials=("first_is_lowest", "size"), first_is_lowest=("first_is_lowest", "mean"), lowest_in_top3=("lowest_in_top3", "mean"),
        first_is_valueless=("first_is_valueless", "mean"), focal_is_lowest=("focal_is_lowest", "mean"), chance_first=("chance_first", "mean")).reset_index()
