"""Manuscript figures (colour + marker + line style redundant encoding, grayscale proofs, fixed 5.28 in width).

Figures use descriptive file names. Extra figures (candidate-set-size complete seeds, lowest-value preference, estimator comparison, serial position by model, design balance) go beyond the manuscript.
Every function takes pre-computed tables and the list of runs (for the per-LLM style) and writes PNG/PDF/EPS.
"""

from __future__ import annotations

import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from conditions import CANDIDATE_SET_SIZE, CONDITION_LABEL, CONDITIONS, EVENT_TYPE, NUMERICAL_VALUE, condition_of
from paths import INPUTS, EXPERIMENT_PARTS
import an_style as S
from an_data import read_jsonl

FAMS = [(c, CONDITION_LABEL[c]) for c in CONDITIONS]
POS5 = ["0", "1/4", "1/2", "3/4", "1"]
FONT, INK, MS, LW, CAP, CHANCE, WIDTH, HEIGHT = S.FONT, S.INK, S.MS, S.LW, S.CAP, S.CHANCE, S.WIDTH, S.HEIGHT


def _line(ax, st, x, y, lo=None, hi=None, dx=0.0, label=True):
    xs = np.asarray(x, dtype=float) + dx
    y = np.asarray(y, dtype=float)
    if lo is not None:
        ax.errorbar(xs, y, yerr=[y - np.asarray(lo), np.asarray(hi) - y], fmt="none", ecolor=st["color"], elinewidth=1.0, capsize=CAP, capthick=1.0, zorder=2)
    ax.plot(xs, y, color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["mfc"], mec=INK, mew=0.6, ms=MS, lw=LW, label=st["label"] if label else None, zorder=3)


def _offsets(n):
    return [(k - (n - 1) / 2) * 0.08 for k in range(n)]


def plot_timing_direction(d, runs, out):
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    n = len(runs)
    specs = [("faster", "o", True, "-", S.FASTER, "Focal faster"), ("slower", "s", False, "--", S.SLOWER, "Focal slower")]
    for k, (direction, marker, filled, ls, color, lab) in enumerate(specs):
        g = d[d.direction == direction].set_index("llm").reindex([r.name for r in runs])
        x = np.arange(n) + (k - 0.5) * 0.14
        ax.errorbar(x, g.top1, yerr=[g.top1 - g.top1_ci_lo, g.top1_ci_hi - g.top1], fmt=marker + ls, color=color, mfc=color if filled else "#ffffff",
                    mec=color, ms=MS + 1, lw=LW, capsize=CAP, elinewidth=1.0, mew=1.4, label=lab)
    chance = d.top1_chance.mean()
    ax.axhline(chance, **CHANCE, label="Chance")
    ax.text(n - 0.55, chance + 0.025, "chance", ha="right", va="bottom", fontsize=FONT - 0.5, color=S.CHANCE_COLOR)
    ax.text(0.0, 0.80, "Focal faster", color=S.FASTER, ha="center", fontsize=FONT, fontweight="bold")
    ax.text(0.0, 0.26, "Focal slower", color=S.SLOWER, ha="center", fontsize=FONT, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels([r.label for r in runs])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1.02)
    S.clean(ax, "Focal event ranked first", "LLM")
    S.save(fig, out, "timing_direction_top1_selection")


def plot_serial_position(pc, runs, out):
    st = S.styles(runs)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.3), sharey=True)
    off = _offsets(len(runs))
    for ax, (fam, title) in zip(axes, FAMS):
        g = pc[pc.condition == fam]
        for k, r in enumerate(runs):
            h = g[g.llm == r.name].set_index("pos").reindex(POS5)
            if h.top3.notna().any():
                _line(ax, st[r.name], np.arange(5), h.top3, h.ci_lo, h.ci_hi, dx=off[k])
        ax.plot(np.arange(5), g.groupby("pos").chance.mean().reindex(POS5), **CHANCE, label="Chance")
        ax.set_xticks(range(5))
        ax.set_xticklabels(POS5, fontsize=FONT - 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=FONT)
        S.clean(ax, xlabel="Focal position")
    axes[0].set_ylabel("Focal event in top three")
    axes[0].legend(frameon=False, loc="upper left", handlelength=2.6, fontsize=FONT - 1, borderaxespad=0.1)
    S.save(fig, out, "serial_position_top3_inclusion")


def plot_pooled_gee_forest(gee, out):
    marker = {EVENT_TYPE: "D", NUMERICAL_VALUE: "s", CANDIDATE_SET_SIZE: "^"}
    rows = []
    for fam, nm in FAMS:
        spec = "with_direction" if fam == NUMERICAL_VALUE else "canonical"
        for pos in ("first", "last"):
            r = gee[(gee.condition == fam) & (gee.spec == spec) & (gee.term == f"pos[T.{pos}]")]
            if len(r):
                r = r.iloc[0]
                rows.append((f"{nm}, {pos} position", r.OR, r.ci_lo, r.ci_hi, pos, fam))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    ypos = np.arange(len(rows))[::-1]
    xmax = max(9, max(r[3] for r in rows) * 1.15)
    xmin = min(0.04, min(r[2] for r in rows) * 0.85)
    for y, (lab, o, lo, hi, pos, fam) in zip(ypos, rows):
        color = "#B03A6F" if pos == "first" else "#00798C"
        ax.errorbar(o, y, xerr=[[o - lo], [hi - o]], fmt=marker[fam], color=color, mfc=color if pos == "last" else "#ffffff", mec=color, ms=MS + 0.5,
                    capsize=CAP, elinewidth=1.3, mew=1.4)
        ax.text(xmax, y, f"{o:.2f} [{lo:.2f}, {hi:.2f}]", va="center", ha="left", fontsize=FONT - 0.5)
    ax.axvline(1, color=INK, lw=1.3)
    ax.text(1.04, len(rows) - 0.35, "OR = 1", fontsize=FONT - 1, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.5, 1, 2, 4, 10, 20, 50) if xmin <= t <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=FONT - 0.5)
    ax.set_xlabel("Odds ratio vs. middle positions (log scale)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=S.GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    S.save(fig, out, "pooled_gee_position_effects_forest")


def _candidate_set_size_plot(d, runs, out, name, col, lo, hi, ylabel, chance_label, ylim, loc, chance_line=None):
    st = S.styles(runs)
    levels = sorted(d.n_events.unique())
    xs = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    off = _offsets(len(runs))
    for k, r in enumerate(runs):
        h = d[d.llm == r.name].set_index("n_events").reindex(levels)
        if lo:
            _line(ax, st[r.name], xs, h[col], h[lo], h[hi], dx=off[k])
        else:
            _line(ax, st[r.name], xs, h[col], dx=off[k])
    if chance_line is None:
        ax.plot(xs, d.groupby("n_events").chance.mean().reindex(levels).values, **CHANCE, marker="x", ms=MS, mew=1.4, label=chance_label)
    else:
        ax.axhline(chance_line, **CHANCE, label=chance_label)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in levels])
    ax.set_xlim(-0.4, len(levels) - 0.6)
    S.clean(ax, ylabel, "Events per stream")
    ax.set_ylim(*ylim) if ylim is not None else ax.set_ylim(0, None)
    ax.legend(frameon=False, loc=loc, ncol=2 if loc == "upper right" else 1, columnspacing=1.0, handlelength=2.6)
    S.save(fig, out, name)


def plot_candidate_set_size_top3(d, runs, out):
    _candidate_set_size_plot(d, runs, out, "candidate_set_size_top3_inclusion", "top3", "ci_lo", "ci_hi", "Focal event in top three", "Chance", (0, 1.02), "upper right")


def plot_candidate_set_size_lift(d, runs, out):
    _candidate_set_size_plot(d, runs, out, "candidate_set_size_chance_adjusted_lift", "lift", "lift_lo", "lift_hi", "Lift over chance", "Chance (lift = 1)", None, "upper left", chance_line=1)


def plot_pit_event_selection(p, runs, out):
    st = S.styles(runs)
    p = p.set_index("llm").reindex([r.name for r in runs])
    n = len(runs)
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT - 0.1))
    for i, r in enumerate(runs):
        s, v = st[r.name], p.loc[r.name]
        ax.errorbar(i, v.top3, yerr=[[v.top3 - v.ci_lo], [v.ci_hi - v.top3]], fmt=s["marker"], color=s["color"], mfc=s["mfc"], mec=INK, mew=0.6, ms=MS + 2,
                    capsize=CAP + 0.5, elinewidth=1.3)
        ax.hlines(v.chance, i - 0.3, i + 0.3, **CHANCE)
        ax.text(i + 0.12, v.top3, f"{100 * v.top3:.0f}%", va="center", ha="left", fontsize=FONT)
    ax.plot([], [], **CHANCE, label="Chance")
    ax.set_xticks(range(n))
    ax.set_xticklabels([r.label for r in runs])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1.0)
    S.clean(ax, "Pit event in top three", "LLM")
    ax.legend(frameon=False, loc="upper left")
    S.save(fig, out, "event_type_pit_event_selection")


def plot_response_quality(q, runs, out):
    st = S.styles(runs)
    q = q.set_index("llm").reindex([r.name for r in runs])
    n = len(runs)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.8))

    def bars(ax, vals, fmt, top):
        for i, r in enumerate(runs):
            ax.bar(i, vals[i], color=st[r.name]["color"], edgecolor=INK, linewidth=0.9, width=0.62, hatch=st[r.name]["hatch"])
            ax.text(i, vals[i], fmt(vals[i]), ha="center", va="bottom", fontsize=FONT - 0.5)
        ax.set_ylim(0, max(top, max(vals) * 1.25 + 1e-9))

    bars(axes[0], ((1 - q.valid_rate) * 100).tolist(), lambda v: f"{v:.1f}", 1)
    S.clean(axes[0], "Invalid responses (%)")
    bars(axes[1], q.responses_with_invented_ids.tolist(), lambda v: f"{int(v)}", 5)
    S.clean(axes[1], "Responses with invented IDs")
    ax = axes[2]
    for i, r in enumerate(runs):
        c = st[r.name]["color"]
        ax.vlines(i, q.top1_agreement_min[r.name], q.top1_agreement_max[r.name], color=c, lw=3.0)
        ax.plot(i, q.top1_agreement_min[r.name], st[r.name]["marker"], color=c, mfc="#ffffff", mec=c, mew=1.6, ms=MS)
        ax.plot(i, q.top1_agreement_max[r.name], st[r.name]["marker"], color=c, mec=INK, ms=MS)
    ax.set_ylim(0.4, 1.0)
    S.clean(ax, "Top-1 agreement (range)")
    for a in axes:
        a.set_xticks(range(n))
        a.set_xticklabels([r.label for r in runs])
        a.set_xlabel("LLM")
    S.save(fig, out, "response_format_quality_by_model")


def _box(ax, xy, w, h, fc, ls="-", lw=1.3):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.0,rounding_size=0.025", fc=fc, ec=INK, ls=ls, lw=lw))


def _arrow(ax, p, q, ls="-", lw=1.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=lw, ls=ls, color=INK, shrinkA=0, shrinkB=0))


def plot_study_framework(out):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [("1", "Operational structured event stream", "#EEF3FA", "-", "Formula 1 timing feed"),
              ("2", "Structured event representation", "#EEF3FA", "-", "id, type, driver, lap and time fields"),
              ("3", "Temporal event window", "#EEF3FA", "-", "events of the last 10 s, no later events"),
              ("4", "LLM-based implicit prioritization", "#FDF1DC", (0, (5, 2)), "no importance rules in the prompt"),
              ("5", "Selected, reportable events", "#F1F1F1", "-", "the three events ranked as most important"),
              ("6", "Monitoring, reporting, commentary", "#F1F1F1", "-", "downstream use, not evaluated in this study")]
    w, h, x0 = 0.58, 0.108, 0.035
    ys = [0.80 - i * 0.153 for i in range(len(stages))]
    for (num, title, fc, ls, sub), y in zip(stages, ys):
        _box(ax, (x0, y), w, h, fc, ls, 2.0 if num == "4" else 1.2)
        ax.text(x0 + 0.03, y + h * 0.72, title, ha="left", va="center", fontsize=FONT, fontweight="bold")
        ax.text(x0 + 0.03, y + h * 0.27, sub, ha="left", va="center", fontsize=FONT - 1.5)
        ax.text(x0 - 0.012, y + h + 0.005, num, ha="center", va="center", fontsize=FONT - 1, fontweight="bold", bbox=dict(boxstyle="circle,pad=0.25", fc="#ffffff", ec=INK, lw=1.0))
    for y_hi, y_lo in zip(ys[:-1], ys[1:]):
        _arrow(ax, (x0 + w / 2, y_hi), (x0 + w / 2, y_lo + h))
    fx, fw = 0.68, 0.31
    top, bottom = ys[3] + h + 0.13, ys[3] - 0.17
    _box(ax, (fx, bottom), fw, top - bottom, "#FFFFFF", (0, (2, 2)), 1.3)
    ax.text(fx + fw / 2, top - 0.035, "Factors studied", ha="center", va="center", fontsize=FONT, fontweight="bold")
    for i, f in enumerate(["Event content (type)", "Numerical direction", "Candidate-set size", "Serial position", "Model configuration"]):
        yy = top - 0.095 - i * 0.065
        ax.text(fx + 0.02, yy, f"{chr(97 + i)})", ha="left", va="center", fontsize=FONT, fontweight="bold")
        ax.text(fx + 0.07, yy, f, ha="left", va="center", fontsize=FONT - 1)
    _arrow(ax, (fx, ys[3] + h / 2), (x0 + w, ys[3] + h / 2), (0, (5, 2)), 1.6)
    ax.add_patch(Rectangle((0.008, ys[2] - 0.02), w + 0.055, ys[0] + h - ys[2] + 0.075, fc="none", ec=INK, lw=1.0, ls=(0, (1, 2))))
    ax.text(x0 + w, ys[0] + h + 0.04, "real Formula 1 data (this study)", ha="right", va="center", fontsize=FONT - 1.5, fontstyle="italic")
    S.save(fig, out, "study_framework")


def plot_example_presentation(out, inputs=INPUTS):
    trials = read_jsonl(inputs / EXPERIMENT_PARTS["main_experiment"][0])
    t = next(t for t in trials if condition_of(t) == EVENT_TYPE and t["n_events"] in (7, 8) and abs(t["focal_position_frac"] - 0.5) < 1e-9
             and t["focal_type"] == "PitIn" and t["types_in_order"].count("LapComplete") == t["n_events"] - 1)
    events = [json.loads(l.strip()) for l in t["prompt"].split("\n") if l.strip().startswith("{") and '"id"' in l]
    abbr = {"LapComplete": "LAP", "SectorCross": "SEC", "PitIn": "PIT-IN", "PitOut": "PIT-OUT"}
    chip_ls = {"LapComplete": "-", "SectorCross": (0, (4, 2)), "PitIn": "-", "PitOut": "-"}
    chip_fc = {"LapComplete": "#EEF3FA", "SectorCross": "#EEF3FA", "PitIn": "#FDF1DC", "PitOut": "#FDF1DC"}
    chip_lw = {"LapComplete": 1.0, "SectorCross": 1.0, "PitIn": 2.2, "PitOut": 2.2}
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.17, 1)
    ax.axis("off")
    _box(ax, (0.01, 0.905), 0.98, 0.085, "#F6F6F6", lw=1.0)
    ax.text(0.025, 0.947, "Prompt: select the 3 events most important to report right now.", ha="left", va="center", fontsize=FONT)
    cols = {"pos": 0.04, "id": 0.12, "type": 0.27, "driver": 0.47, "lap": 0.59, "time": 0.72}
    heads = {"pos": "No.", "id": "Id", "type": "Type", "driver": "Driver", "lap": "Lap", "time": "Time (s)"}
    y = 0.835
    for k, v in cols.items():
        ax.text(v, y, heads[k], ha="left", va="center", fontsize=FONT - 0.5, fontweight="bold")
    ax.plot([0.02, 0.84], [y - 0.035, y - 0.035], color=INK, lw=0.9)
    rh = 0.088
    for i, e in enumerate(events):
        yy = y - 0.085 - i * rh
        focal = e["id"] == t["focal_label"]
        if focal:
            ax.add_patch(Rectangle((0.02, yy - rh / 2 + 0.006), 0.82, rh - 0.012, fc="#FFF7E6", ec=INK, lw=2.0, zorder=0))
        ax.text(cols["pos"], yy, f"{i + 1}", ha="left", va="center", fontsize=FONT)
        ax.text(cols["id"], yy, e["id"], ha="left", va="center", fontsize=FONT, family="monospace")
        ax.add_patch(FancyBboxPatch((cols["type"] - 0.01, yy - 0.026), 0.135, 0.052, boxstyle="round,pad=0.0,rounding_size=0.012", fc=chip_fc[e["type"]], ec=INK,
                                    ls=chip_ls[e["type"]], lw=chip_lw[e["type"]]))
        ax.text(cols["type"] + 0.0575, yy, abbr[e["type"]], ha="center", va="center", fontsize=FONT - 1.5, fontweight="bold")
        ax.text(cols["driver"], yy, e["driver"], ha="left", va="center", fontsize=FONT)
        ax.text(cols["lap"], yy, str(e.get("lap", "")), ha="left", va="center", fontsize=FONT)
        tv = e.get("lap_time_s", e.get("sector_time_s", ""))
        ax.text(cols["time"], yy, f"{tv:.3f}" if tv != "" else "-", ha="left", va="center", fontsize=FONT)
        if focal:
            ax.annotate("focal event\n(not marked in\nthe prompt)", xy=(0.845, yy), xytext=(0.88, yy), ha="left", va="center", fontsize=FONT - 1.5,
                        arrowprops=dict(arrowstyle="-|>", lw=1.4, color=INK))
    S.save(fig, out, "example_presentation_event_type")


def plot_candidate_set_size_complete_seeds(dc, runs, out):
    _candidate_set_size_plot(dc, runs, out, "candidate_set_size_lift_complete_seeds", "lift", None, None, "Lift over chance (complete seeds)", "Chance (lift = 1)", None, "upper left", chance_line=1)


def plot_lowest_displayed_value_preference(mv, runs, out):
    g = mv[mv.condition == NUMERICAL_VALUE]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharey=True)
    n = len(runs)
    for ax, (col, title) in zip(axes, [("first_is_lowest", "Ranked first is the lowest"), ("lowest_in_top3", "Lowest is in top three")]):
        for k, (direction, color, marker) in enumerate([("faster", S.FASTER, "o"), ("slower", S.SLOWER, "s")]):
            h = g[g.direction == direction].set_index("llm").reindex([r.name for r in runs])
            ax.plot(np.arange(n) + (k - 0.5) * 0.12, h[col], marker, color=color, mfc=color if k == 0 else "#ffffff", mec=color, ms=MS + 1, mew=1.4,
                    label=f"Focal {direction}")
        ax.set_xticks(range(n))
        ax.set_xticklabels([r.label for r in runs])
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=FONT - 0.5)
        S.clean(ax, xlabel="LLM")
    axes[0].set_ylabel("Share of presentations")
    axes[0].legend(frameon=False, loc="upper left", fontsize=FONT - 1)
    S.save(fig, out, "lowest_displayed_value_preference")


def plot_estimator_comparison(wide, out):
    w = wide[wide.term.str.startswith("pos")]
    if w.empty:
        return
    est = [("OR_naive logit", "Naive logit", "o", S.COLORS[5]), ("OR_cluster-robust logit", "Cluster-robust logit", "s", S.COLORS[1]),
           ("OR_GEE exchangeable", "GEE exchangeable", "^", S.COLORS[0])]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    labels = [f"{r.spec}, {r.term}" for r in w.itertuples()]
    ys = np.arange(len(w))[::-1]
    for k, (col, lab, mk, c) in enumerate(est):
        ax.plot(w[col].values, ys + (k - 1) * 0.18, mk, color=c, mec=INK, mew=0.6, ms=MS - 1, label=lab)
    ax.axvline(1, color=INK, lw=1.0)
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=FONT - 2)
    ax.set_xlabel("Odds ratio (log scale)")
    ax.grid(axis="x", color=S.GRID, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=FONT - 1.5, loc="lower right")
    S.save(fig, out, "estimator_comparison")


def plot_serial_position_by_model(pm, runs, out):
    st = S.styles(runs)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9), sharey=True)
    for ax, (fam, title) in zip(axes, FAMS):
        g = pm[pm.condition == fam]
        for k, pos in enumerate(("first", "last")):
            for j, r in enumerate(runs):
                h = g[(g.llm == r.name) & (g.pos == pos)]
                if h.empty:
                    continue
                h = h.iloc[0]
                x = k + (j - (len(runs) - 1) / 2) * 0.2
                ax.errorbar(x, h.OR, yerr=[[h.OR - h.ci_lo], [h.ci_hi - h.OR]], fmt=st[r.name]["marker"], color=st[r.name]["color"], mec=INK, mew=0.6, ms=MS - 1,
                            capsize=CAP, label=st[r.name]["label"] if k == 0 else None)
        ax.axhline(1, color=INK, lw=1.0)
        ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["first", "last"])
        ax.set_title(title, fontsize=FONT)
        S.clean(ax)
    axes[0].set_ylabel("Odds ratio vs. middle")
    axes[0].legend(frameon=False, fontsize=FONT - 1.5, loc="lower left")
    S.save(fig, out, "serial_position_effects_by_model")


def plot_design_balance(ns, per_race, out):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    races = per_race.race.str.replace("2025_", "").str.replace("_R", "").str.title()
    x = np.arange(len(per_race))
    ax.bar(x, per_race.event_type, color=S.COLORS[0], edgecolor=INK, width=0.7, label="Event type")
    ax.bar(x, per_race.numerical_value, bottom=per_race.event_type, color="#ffffff", edgecolor=INK, hatch="////", width=0.7, label="Numerical value")
    ax.set_xticks(x)
    ax.set_xticklabels([r[:9] for r in races], rotation=70, ha="right", fontsize=FONT - 2.5)
    S.clean(ax, "Seed windows")
    ax.legend(frameon=False, fontsize=FONT - 1.5)
    ax = axes[1]
    rng = np.random.default_rng(3)
    for k, (d, c, mk) in enumerate([("faster", S.FASTER, "o"), ("slower", S.SLOWER, "s")]):
        z = ns[ns.direction == d].robust_z
        ax.plot(np.full(len(z), k) + rng.uniform(-0.15, 0.15, len(z)), z, mk, color=c, mfc="none", ms=MS - 2, mew=1.0)
        ax.hlines(z.median(), k - 0.3, k + 0.3, color=INK, lw=1.6)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["faster", "slower"])
    ax.set_xlim(-0.6, 1.6)
    S.clean(ax, "Robust z of the focal value")
    S.save(fig, out, "design_balance")


def _p2_axes(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(WIDTH, HEIGHT + 0.2), sharey=True, squeeze=False)
    return fig, axes[0]


def followup_cells(desc, runs, y, out, name, ylabel):
    st = S.styles(runs)
    fig, axes = _p2_axes(runs)
    for ax, r in zip(axes, runs):
        for p, marker, off in (("first", "o", -0.1), ("last", "s", 0.1)):
            d = desc[(desc.llm == r.name) & (desc.position == p)].set_index("content").reindex(["original", "faster", "slower"])
            ax.errorbar(np.arange(3) + off, d[f"{y}_pct"], yerr=[d[f"{y}_pct"] - d[f"{y}_ci_lo"], d[f"{y}_ci_hi"] - d[f"{y}_pct"]], fmt=marker,
                        color=st[r.name]["color"], mfc=st[r.name]["color"] if p == "last" else "#ffffff", mec=st[r.name]["color"], ms=MS, capsize=CAP, lw=LW,
                        mew=1.3, label=p)
        if y == "top3":
            ax.axhline(desc[desc.llm == r.name].chance3_pct.mean(), **CHANCE)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["original", "faster", "slower"], fontsize=FONT - 1.5)
        ax.set_ylim(0, 122)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_title(r.label, fontsize=FONT)
        S.clean(ax, xlabel="Focal time")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(title="position", fontsize=FONT - 2, title_fontsize=FONT - 2, frameon=False, loc="upper left", ncol=2)
    S.save(fig, out, name)


def followup_effects(eff, runs, out):
    """Central follow-up figure: numerical-rank effect (fastest - slowest) versus serial-position effect (last - first), per LLM, 95% paired-bootstrap CI."""
    from matplotlib.lines import Line2D
    st = S.styles(runs)
    series = [("faster-slower | mean", -0.17, True, "-", S.FASTER), ("last-first | mean", 0.17, False, "--", S.SLOWER)]
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT + 0.4))
    for i, r in enumerate(runs):
        d = eff[(eff.group == r.name) & (eff.outcome == "top3")].set_index("effect")
        for key, dx, filled, ls, col in series:
            v = d.loc[key]
            x = i + dx
            bars = ax.errorbar(x, v.pp, yerr=[[v.pp - v.ci_lo_pp], [v.ci_hi_pp - v.pp]], fmt=st[r.name]["marker"], color=col, mfc=col if filled else "#ffffff", mec=INK if filled else col,
                               ms=MS + 1, mew=1.5, capsize=CAP + 1, elinewidth=1.6, capthick=1.4, zorder=3)
            bars[2][0].set_linestyle(ls)
            ax.text(x + 0.07, v.pp, f"{v.pp:+.1f}", ha="left", va="center", fontsize=FONT - 1.5, color=INK)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"Qwen3 {r.label}" if r.label[:-1].replace(".", "").isdigit() else r.label for r in runs])
    ax.set_xlim(-0.5, len(runs) - 0.38)
    S.clean(ax, "Change in Top-3 selection probability" + chr(10) + "(percentage points)")
    ax.set_ylim(-4, 100)
    handles = [Line2D([0], [0], marker="o", color=S.FASTER, mfc=S.FASTER, mec=INK, ls="-", lw=1.6, ms=MS - 1, label="Numerical rank (fastest − slowest)"),
               Line2D([0], [0], marker="o", color=S.SLOWER, mfc="#ffffff", mec=S.SLOWER, mew=1.5, ls="--", lw=1.6, ms=MS - 1, label="Serial position (last − first)")]
    ax.legend(handles=handles, frameon=False, fontsize=FONT - 1.5, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    S.save(fig, out, "timing_rank_vs_serial_position_effects")


def followup_crossover(desc, runs, out):
    fig, axes = _p2_axes(runs)
    cstyle = {"original": ("#555555", "o", "-"), "faster": (S.FASTER, "^", "--"), "slower": (S.SLOWER, "v", "-.")}
    for ax, r in zip(axes, runs):
        for c, (col, mk, ls) in cstyle.items():
            d = desc[(desc.llm == r.name) & (desc.content == c)].set_index("position").reindex(["first", "last"])
            ax.plot([0, 1], d.top3_pct, mk + ls, color=col, ms=MS, lw=LW, mec=INK, mew=0.6, label=c)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["first", "last"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(0, 122)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_title(r.label, fontsize=FONT)
        S.clean(ax, xlabel="Focal position")
    axes[0].set_ylabel("Focal event in top three (%)")
    axes[0].legend(frameon=False, fontsize=FONT - 2, loc="upper left", title="focal time", title_fontsize=FONT - 2, ncol=3, columnspacing=0.8, handlelength=1.8)
    S.save(fig, out, "timing_rank_vs_serial_position_crossover")
