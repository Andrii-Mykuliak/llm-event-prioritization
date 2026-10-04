IMPLICIT EVENT PRIORITIZATION IN LARGE LANGUAGE MODELS: REPRODUCIBLE RELEASE

One study of which event a large language model (LLM) picks as most important in a short stream of
Formula 1 timing events, with no importance rule in the prompt (implicit event prioritization; the measured quantity is implicit event saliency):
  - main experiment: 1,040 presentations in three experimental conditions (event type, numerical value, candidate-set size);
    the focal event is shown at five serial positions in every condition;
  - controlled timing-rank follow-up: the same real source windows with the focal timing made the fastest or the
    slowest of its comparison set and the focal event first or last (42 source windows x 6 variants = 252 presentations).

WHAT YOU CAN DO
  See all figures and tables from the shipped responses:   python scripts/analyze.py --reference
  Run ONE LLM and compare it with the reference:           python scripts/run_llm.py --llm configs/llm_<your-llm>.yaml
                                                            python scripts/analyze.py --with-reference --tag my_run
  Do everything in a notebook:                             open llm_saliency_pipeline.ipynb
  Rebuild the input data:                                  python scripts/make_inputs.py --out inputs_rebuilt --download --cache .fastf1_cache --compare inputs
                                                            (or --from-primitive inputs to rebuild offline)
  Check the shipped inputs:                                python scripts/verify_inputs.py
  Check run files against the inputs:                      python scripts/integrity.py [--with-reference]
  Run the tests:                                           python -m pytest

INSTALL
  python -m venv .venv ; activate it ; pip install -r requirements.txt   (Python 3.11 or newer)
  An LLM server is needed only to run inference: Ollama (ollama pull qwen3:8b) or any OpenAI-compatible server.

LAYOUT (every folder holds files only; related files share a name prefix)
  llm_saliency_pipeline.ipynb  the notebook that orchestrates everything (saved without outputs; run it to produce them)
  configs/             extractor.yaml, main_experiment.yaml, controlled_timing_rank_followup.yaml: frozen, byte-identical to the original package
                       llm_<name>.yaml: one file per LLM (qwen3-4b-q4_K_M, qwen3-8b, qwen3-14b, TEMPLATE_ollama, TEMPLATE_openai_compatible, mock)
  inputs/              primitive_events, seed_windows, dataset_*, main_experiment_trials (main experiment), controlled_timing_rank_followup_trials and
                       controlled_timing_rank_followup_exclusions (follow-up), MANIFEST.sha256 (verify_inputs.py writes VERIFY_*.txt)
  reference_results/   <llm>__main_experiment_responses.jsonl and <llm>__controlled_timing_rank_followup_responses.jsonl of the three reference LLMs;
                       controlled_timing_rank_followup_frozen_* = frozen follow-up tables used as a check
  results/             YOUR runs: <llm>__main_experiment_responses.jsonl, <llm>__controlled_timing_rank_followup_responses.jsonl,
                       <llm>__run_meta.json, <llm>__errors.jsonl
  outputs/             <tag>__<name>.<ext>: figures (png, pdf, eps, _gray.png) and tables (csv); python scripts/analyze.py --reference creates reference__*
  src/                 si_*.py creates the input data; run.py, cli.py, llm_*.py is the one-LLM pipeline; an_*.py is the analysis
  scripts/             make_inputs.py, verify_inputs.py, run_llm.py, analyze.py, integrity.py, run_reference_qwen3_models.sh / .ps1
  tests/               pytest suite

USING YOUR OWN LLM
  1. Copy configs/llm_TEMPLATE_ollama.yaml (or llm_TEMPLATE_openai_compatible.yaml) to configs/llm_my-llm.yaml and set name,
     short_label and model. name may contain letters, digits . _ - only; result files become results/<name>__*.
  2. python scripts/run_llm.py --llm configs/llm_my-llm.yaml --limit 20     (smoke test, kept apart as <name>__smoke)
     (--parts main_experiment controlled_timing_rank_followup selects which experiment part to run; default both)
  3. python scripts/run_llm.py --llm configs/llm_my-llm.yaml                (full run, 1,292 requests, resumable)
  4. python scripts/analyze.py --with-reference --tag my_run
  Run the pipeline once per LLM. Frozen inference settings (temperature 0, seed 42, context 4096, 48 output tokens, thinking off)
  are the defaults; any deviation is recorded in <name>__run_meta.json. Thinking must be off for a comparable run; a model that
  cannot answer with thinking off will give many invalid answers, which the quality table reports. configs/llm_mock.yaml tests the
  pipeline offline (mock runs are ignored by the analysis unless --include-mock).

OUTPUT NAMES (descriptive; <tag>__report.txt lists every file of an analysis with its title)
  Figures:  study_framework, example_presentation_event_type, timing_direction_top1_selection, serial_position_top3_inclusion,
            pooled_gee_position_effects_forest, candidate_set_size_top3_inclusion, candidate_set_size_chance_adjusted_lift,
            event_type_pit_event_selection, response_format_quality_by_model, controlled_timing_rank_followup_top3_inclusion,
            controlled_timing_rank_followup_top1_selection, timing_rank_vs_serial_position_effects,
            timing_rank_vs_serial_position_crossover; extra: candidate_set_size_lift_complete_seeds, lowest_displayed_value_preference,
            estimator_comparison, serial_position_effects_by_model, design_balance. Each figure also exists as _gray.png.
  Tables:   dataset_composition, focal_selection_by_condition_and_model, timing_direction_by_model, pooled_gee_effects(_full),
            serial_position_effects_by_model, serial_position_by_model_interaction, controlled_timing_rank_followup_descriptive,
            controlled_timing_rank_followup_marginal_effects_pp, controlled_timing_rank_followup_gee,
            controlled_timing_rank_followup_response_quality, controlled_timing_rank_followup_excluded_sources;
            extra: serial_position_top3_curve, candidate_set_size_top3_and_lift, candidate_set_size_complete_seeds(_log_odds),
            event_type_pit_events, response_quality_and_top1_agreement, valid_only_versus_all_trial_rates, parser_audit,
            lowest_displayed_value_preference, top1_agreement_by_condition_and_model, estimator_comparison_long / _wide;
            design diagnostics design_*; integrity_checks and frozen_config_hashes.
  Follow-up bootstrap: reference__* uses the frozen random stream (tables equal the frozen tables to 1e-14); any other set of
  runs uses a fixed seed per cell, so intervals do not depend on which LLMs are analysed together.

FROZEN DATA KEYS
  The input files and the three frozen configurations are byte-identical to the original package, because trial files embed the hash of the
  configuration and the reference responses refer to trial ids. Their internal keys are therefore unchanged: the experimental condition is stored in the field
  "family" (values type_outlier = event type, numerical_outlier = numerical value, density = candidate-set size), controlled follow-up trial ids start with
  "p2rR", and the follow-up trial records contain a "phase" field; the follow-up configuration keeps the key "phase1_seed_windows_sha256". The frozen follow-up configuration also keeps its original comments. The analysis code translates the condition keys to the descriptive condition names when it loads the data.
  Renaming the files changed only the paths in inputs/MANIFEST.sha256, not any content hash.

REPRODUCIBILITY
  1. verify_inputs.py: 16 main-experiment and 28 follow-up structural checks.
  2. inputs/MANIFEST.sha256 holds hashes of all input files and frozen configurations; run_llm.py refuses to run if they differ.
  3. make_inputs.py --out inputs_rebuilt --from-primitive inputs --compare inputs gives byte-identical files.
  4. python -m pytest: inputs verify, reference numbers reproduce the manuscript, the mock pipeline runs end to end.
  5. LLM answers depend on model, quantization, server version and hardware, so a new LLM gives numbers close to, not equal to,
     the reference. Compare patterns, not third decimals.
  Frozen configuration SHA-256:
    ba33d35589b6d96ba0f9b251e4f4a7c2d72af53a1fc62bfe97e8d092becf43b0  configs/extractor.yaml
    26061798626efc920fa135d42e88bbfce249c6768daf2597aa7939e39d2f968a  configs/main_experiment.yaml
    28d17492bc84c86fbda535b60e76122058a09b66b31e9ca7b9d36009936a604a  configs/controlled_timing_rank_followup.yaml
  The first hash is stored in every trial and the third in every follow-up trial, so these files must stay byte-identical.
  controlled_timing_rank_followup_exclusions.json has English reason text (sha 057593e5...); the excluded windows and numbers equal the original.
