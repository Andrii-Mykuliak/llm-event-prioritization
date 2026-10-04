"""Locations inside the release folder (the release root is the parent of src/). Every folder of the release is flat: files only."""

from __future__ import annotations

from pathlib import Path

__version__ = "1.0.0"

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
FROZEN = CONFIGS
LLM_CONFIGS = CONFIGS
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"
REFERENCE = ROOT / "reference_results"
OUTPUTS = ROOT / "outputs"

MAIN_EXPERIMENT_CONFIG = FROZEN / "main_experiment.yaml"
EXTRACTOR = FROZEN / "extractor.yaml"
FOLLOWUP_CONFIG = FROZEN / "controlled_timing_rank_followup.yaml"

# file names inside the inputs folder
PRIMITIVE_EVENTS = "primitive_events.jsonl"
SEED_WINDOWS = "seed_windows.jsonl"
DATASET_MANIFEST = "dataset_manifest.json"
DATASET_SUMMARY = "dataset_summary.csv"
MAIN_EXPERIMENT_TRIALS = "main_experiment_trials.jsonl"
FOLLOWUP_TRIALS = "controlled_timing_rank_followup_trials.jsonl"
FOLLOWUP_EXCLUSIONS = "controlled_timing_rank_followup_exclusions.json"

# part name -> (trials file inside the inputs folder, response file suffix).
# The responses of one LLM run live in results/<run name>__<suffix>; run_meta and errors use the same prefix.
EXPERIMENT_PARTS = {
    "main_experiment": (MAIN_EXPERIMENT_TRIALS, "main_experiment_responses.jsonl"),
    "controlled_timing_rank_followup": (FOLLOWUP_TRIALS, "controlled_timing_rank_followup_responses.jsonl"),
}
RUN_META = "run_meta.json"
RUN_ERRORS = "errors.jsonl"


def run_file(root, run_name: str, suffix: str) -> Path:
    return Path(root) / f"{run_name}__{suffix}"
