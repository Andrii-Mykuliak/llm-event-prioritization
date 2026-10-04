"""End to end with the offline mock backend: run -> integrity -> analysis (single LLM)."""
import an_data as data
import an_integrity as integrity
from an_report import build_report
from llm_config import load_llm_config
from parsing import parse_labels
from paths import LLM_CONFIGS
from run import run_llm


def test_parser_rules():
    assert parse_labels("ABC,DEF,GHI", {"ABC", "DEF", "GHI"}) == ["ABC", "DEF", "GHI"]


def test_config_defaults_are_frozen():
    cfg = load_llm_config(LLM_CONFIGS / "llm_qwen3-8b.yaml")
    assert cfg.options["temperature"] == 0 and cfg.options["seed"] == 42 and cfg.think is False and not cfg.deviations_from_frozen


def test_mock_run_and_report(tmp_path):
    run_llm(LLM_CONFIGS / "llm_mock.yaml", results_root=tmp_path / "res", progress=lambda *_: None)
    runs = data.discover_runs(tmp_path / "res", include_mock=True)
    assert len(runs) == 1
    assert integrity.check_runs(runs).ok.all()
    s = build_report(runs, tmp_path / "out", tag="x", extras=False, log=lambda *_: None)
    assert s["integrity_all_ok"] and (tmp_path / "out" / "x__serial_position_top3_inclusion.png").exists()
