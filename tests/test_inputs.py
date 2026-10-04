from si_verify import verify_main_experiment, verify_followup
from an_integrity import check_runs
from paths import FROZEN, INPUTS


def test_shipped_inputs_pass_all_checks():
    r1 = verify_main_experiment(INPUTS / "main_experiment_trials.jsonl", INPUTS)
    r2 = verify_followup(FROZEN / "controlled_timing_rank_followup.yaml", INPUTS / "controlled_timing_rank_followup_trials.jsonl", INPUTS / "main_experiment_trials.jsonl",
                            INPUTS, FROZEN / "main_experiment.yaml")
    assert r1.ok, r1.summary()
    assert r2.ok, r2.summary()


def test_manifest_matches_files():
    assert check_runs([]).ok.all()
