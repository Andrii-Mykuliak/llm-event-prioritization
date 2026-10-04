"""The shipped reference responses must give the numbers reported in the manuscript."""
import pytest

import an_data as data
import an_controlled_timing_rank_followup as followup
import an_stats as stats
from paths import REFERENCE


@pytest.fixture(scope="module")
def ctx():
    runs = data.discover_runs(REFERENCE, reference_roots=(REFERENCE,))
    return runs, data.load_main_experiment(runs), [r.name for r in runs]


def test_counts(ctx):
    runs, df, _ = ctx
    assert [r.label for r in runs] == ["4B", "8B", "14B"]
    assert len(df) == 3120 and int(df.format_ok.sum()) == 3062


def test_direction_top1(ctx):
    _, df, _ = ctx
    d = stats.direction_table(df).set_index(["label", "direction"])
    assert d.loc[("14B", "faster"), "top1"] == pytest.approx(0.927, abs=1e-3)
    assert d.loc[("14B", "slower"), "top1"] == pytest.approx(0.043, abs=1e-3)


def test_pooled_gee_position_ors(ctx):
    _, df, order = ctx
    g = stats.gee_models(df, order)
    def orr(fam, spec, term):
        return float(g[(g.condition == fam) & (g.spec == spec) & (g.term == term)].OR.iloc[0])
    assert orr("event_type", "canonical", "pos[T.first]") == pytest.approx(0.229, abs=2e-3)
    assert orr("event_type", "canonical", "pos[T.last]") == pytest.approx(4.034, abs=2e-3)
    assert orr("candidate_set_size", "canonical", "pos[T.last]") == pytest.approx(3.538, abs=2e-3)


def test_followup_equals_frozen_tables(ctx):
    runs, _, order = ctx
    followup_df = data.load_followup(runs)
    desc = followup.descriptive(followup_df, frozen_random_stream=True)
    eff = followup.all_effects(followup_df, order, frozen_random_stream=True)
    frozen = REFERENCE
    if not frozen.exists():
        pytest.skip("frozen follow-up tables are not shipped")
    chk = followup.frozen_reference_check(desc, eff, frozen)
    assert chk["descriptive_max_abs_diff_pp"] < 1e-9 and chk["effect_max_abs_diff_pp"] < 1e-9
