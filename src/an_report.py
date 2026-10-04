"""One call that produces every figure and table for a set of run folders.

    from an_report import build_report
    build_report(runs, "outputs", tag="my_run")

Writes flat files <out>/<tag>__<name>.<ext> (figures, tables, <tag>__report.txt, <tag>__REPORT_MANIFEST.json).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from paths import INPUTS, MAIN_EXPERIMENT_TRIALS, REFERENCE
import an_design as design
import an_figures as F
import an_integrity as integrity
import an_controlled_timing_rank_followup as followup
import an_stats as stats
import an_style as style
import an_tables as T
from an_data import RunInfo, TrialStore, load_main_experiment, load_followup, load_seeds, read_jsonl

REFERENCE_TRIO = ["qwen3-14b", "qwen3-4b-q4_K_M", "qwen3-8b"]


def _is_reference_trio(runs: list[RunInfo]) -> bool:
    return sorted(r.name for r in runs) == REFERENCE_TRIO


def build_report(runs: list[RunInfo], out: str | Path, tag: str = "latest", inputs: Path = INPUTS, extras: bool = True, run_main_experiment: bool = True, run_followup: bool = True,
                 frozen_random_stream: bool | None = None, log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = tab_dir = out
    style.PREFIX = f"{tag}__"
    style.apply_theme()
    order = [r.name for r in runs]
    frozen_stream = _is_reference_trio(runs) if frozen_random_stream is None else frozen_random_stream
    made: dict[str, list[str]] = {"figures": [], "tables": []}
    titles: dict[str, str] = {}
    t0 = time.time()

    def fig(name):
        made["figures"].append(name)
        log(f"  figure {name}")

    def tab(df, name, title, note="", nd=3):
        if "llm" in df.columns:
            df = df.assign(_o=df.llm.map({n: i for i, n in enumerate(order)})).sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
        T.write_table(df, tab_dir, name, title, note, nd)
        titles[name] = title
        made["tables"].append(name)
        log(f"  table  {name}")

    seeds = load_seeds(inputs)
    main_trials = read_jsonl(Path(inputs) / MAIN_EXPERIMENT_TRIALS)
    ic = integrity.check_runs(runs, inputs)
    tab(ic, "integrity_checks", "Integrity checks of the runs against the inputs", "Every row must be ok = True before any result is interpreted.")
    tab(integrity.frozen_config_hashes(), "frozen_config_hashes", "SHA-256 of the frozen configuration files")

    summary: dict = {"runs": [r.name for r in runs], "integrity_all_ok": bool(ic.ok.all()), "frozen_random_stream": frozen_stream}

    F.plot_study_framework(fig_dir)
    fig("study_framework")
    F.plot_example_presentation(fig_dir, inputs)
    fig("example_presentation_event_type")
    tab(design.dataset_table(inputs), "dataset_composition", "Dataset and experimental composition")

    if run_main_experiment:
        df = load_main_experiment(runs, inputs)
        store = TrialStore(inputs)
        if df.empty:
            log("  no main-experiment responses in the selected runs")
        else:
            fs = stats.condition_summary(df)
            dt = stats.direction_table(df)
            pc = stats.position_curve(df)
            dens = stats.candidate_set_size_table(df)
            pit = stats.pit_table(df)
            q = stats.quality_table(df)
            gee = stats.gee_models(df, order)
            pm, inter = stats.position_by_llm(df)
            dc, dc_log = stats.candidate_set_size_complete_seeds(df)
            tab(fs, "focal_selection_by_condition_and_model", "Focal selection by experimental condition and LLM", "Rates over format-valid responses; 95% cluster bootstrap CI (seed windows, 2,000 resamples, seed 7).")
            tab(dt, "timing_direction_by_model", "Numerical-value condition by direction of the focal value")
            tab(T.gee_wide(gee), "pooled_gee_effects", "Pooled logistic GEE (exchangeable, clusters = seed windows)", "Odds ratios; positions vs. middle; first listed LLM is the reference.")
            tab(gee, "pooled_gee_effects_full", "Pooled logistic GEE, all terms and numeric columns")
            tab(pm, "serial_position_effects_by_model", "Position effects fitted separately per LLM")
            tab(inter, "serial_position_by_model_interaction", "Wald test of the Position x LLM interaction")
            tab(pc, "serial_position_top3_curve", "Top-3 rate by focal position (five positions)")
            tab(dens, "candidate_set_size_top3_and_lift", "Candidate-set size: Top-3 and lift over chance")
            tab(dc, "candidate_set_size_complete_seeds", "Candidate-set size restricted to seeds available at every list size")
            tab(dc_log, "candidate_set_size_complete_seeds_log_odds", "Log list-size odds ratio on complete seeds")
            tab(pit, "event_type_pit_events", "Pit events (event-type condition)")
            tab(q, "response_quality_and_top1_agreement", "Response quality and Top-1 agreement")
            tab(stats.denominators(df), "valid_only_versus_all_trial_rates", "Valid-only versus all-trial rates")
            tab(stats.parser_audit(df, store), "parser_audit", "Independent re-parse of every raw response")
            tab(stats.min_value_preference(df, store), "lowest_displayed_value_preference", "Preference for the lowest displayed timing value")
            tab(stats.top1_agreement(df), "top1_agreement_by_condition_and_model", "Top-1 agreement across the five focal positions")
            long, wide = stats.estimator_comparison(df, order)
            tab(long, "estimator_comparison_long", "Naive logit, cluster-robust logit and GEE")
            tab(wide, "estimator_comparison_wide", "Estimator comparison, one row per term")
            summary["estimator_headline"] = stats.estimator_headline(wide)
            ns = design.numeric_seeds(seeds)
            tab(design.direction_balance(ns), "design_direction_balance", "Is the faster/slower contrast confounded with seed properties?")
            tab(design.seeds_per_race(seeds), "design_seeds_per_race", "Seed windows per race and condition")
            tab(design.representation(main_trials), "design_event_representation", "How each event type is written in the prompt")
            tab(design.candidate_set_size_availability(main_trials), "design_candidate_set_size_availability", "Seeds and presentations per list size")
            tab(design.candidate_set_size_composition(main_trials, inputs), "design_candidate_set_size_composition", "Composition of the background by list size")

            F.plot_timing_direction(dt, runs, fig_dir)
            fig("timing_direction_top1_selection")
            F.plot_serial_position(pc, runs, fig_dir)
            fig("serial_position_top3_inclusion")
            F.plot_pooled_gee_forest(gee, fig_dir)
            fig("pooled_gee_position_effects_forest")
            F.plot_candidate_set_size_top3(dens, runs, fig_dir)
            fig("candidate_set_size_top3_inclusion")
            F.plot_candidate_set_size_lift(dens, runs, fig_dir)
            fig("candidate_set_size_chance_adjusted_lift")
            F.plot_pit_event_selection(pit, runs, fig_dir)
            fig("event_type_pit_event_selection")
            F.plot_response_quality(q, runs, fig_dir)
            fig("response_format_quality_by_model")
            if extras:
                F.plot_candidate_set_size_complete_seeds(dc, runs, fig_dir)
                fig("candidate_set_size_lift_complete_seeds")
                F.plot_lowest_displayed_value_preference(stats.min_value_preference(df, store), runs, fig_dir)
                fig("lowest_displayed_value_preference")
                F.plot_estimator_comparison(wide, fig_dir)
                fig("estimator_comparison")
                F.plot_serial_position_by_model(pm, runs, fig_dir)
                fig("serial_position_effects_by_model")
                F.plot_design_balance(ns, design.seeds_per_race(seeds), fig_dir)
                fig("design_balance")
            summary["main_experiment"] = {"responses": int(len(df)), "valid": int(df.format_ok.sum()),
                                 "top3_by_condition": {f"{r.condition}|{r.llm}": round(float(r.top3), 4) for r in fs.itertuples()}}

    if run_followup:
        followup_df = load_followup(runs, inputs)
        if followup_df.empty:
            log("  no controlled timing-rank follow-up responses in the selected runs")
        else:
            followup_runs = [r for r in runs if r.has_followup]
            desc = followup.descriptive(followup_df, frozen_random_stream=frozen_stream)
            eff = followup.all_effects(followup_df, [r.name for r in followup_runs], frozen_random_stream=frozen_stream)
            tab(desc, "controlled_timing_rank_followup_descriptive", "Controlled timing-rank follow-up: Top-1 and Top-3 by focal time and position", "Percent, 95% paired bootstrap over source windows (B = 5000).", nd=2)
            tab(eff, "controlled_timing_rank_followup_marginal_effects_pp", "Controlled timing-rank follow-up: marginal effects in percentage points", nd=2)
            tab(followup.quality(followup_df), "controlled_timing_rank_followup_response_quality", "Controlled timing-rank follow-up: response quality")
            tab(followup.gee_followup(followup_df, [r.name for r in followup_runs]), "controlled_timing_rank_followup_gee", "Controlled timing-rank follow-up: GEE top3 ~ content x position (+ LLM)")
            tab(followup.exclusions(inputs), "controlled_timing_rank_followup_excluded_sources", "Controlled timing-rank follow-up: source windows excluded by the plausibility rule")
            followup_cells = F.followup_cells
            followup_cells(desc, followup_runs, "top3", fig_dir, "controlled_timing_rank_followup_top3_inclusion", "Focal event in top three (%)")
            fig("controlled_timing_rank_followup_top3_inclusion")
            followup_cells(desc, followup_runs, "top1", fig_dir, "controlled_timing_rank_followup_top1_selection", "Focal event ranked first (%)")
            fig("controlled_timing_rank_followup_top1_selection")
            F.followup_effects(eff, followup_runs, fig_dir)
            fig("timing_rank_vs_serial_position_effects")
            F.followup_crossover(desc, followup_runs, fig_dir)
            fig("timing_rank_vs_serial_position_crossover")
            m = eff[(eff.outcome == "top3") & eff.effect.str.endswith("| mean")]
            summary["rank_followup"] = {"responses": int(len(followup_df)), "mean_effects_pp": {f"{r.group}|{r.effect}": round(float(r.pp), 2) for r in m.itertuples()}}

    summary["seconds"] = round(time.time() - t0, 1)
    (out / f"{tag}__REPORT_MANIFEST.json").write_text(json.dumps({**summary, **made}, indent=2), encoding="utf-8")
    lines = [f"# Report: {', '.join(order)}", "", f"Integrity checks passed: {summary['integrity_all_ok']}", "", "## Tables", ""]
    lines += [f"- {tag}__{n}.csv: {titles[n]}" for n in made["tables"]]
    lines += ["", "## Figures", ""] + [f"- {tag}__{n}.png, .pdf, .eps, _gray.png" for n in made["figures"]]
    (out / f"{tag}__report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report written to {out} in {summary['seconds']} s")
    return summary
