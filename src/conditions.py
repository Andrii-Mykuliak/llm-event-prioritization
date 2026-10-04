"""The three main experimental conditions and their labels; the frozen input files store them under older data keys."""

from __future__ import annotations

EVENT_TYPE = "event_type"
NUMERICAL_VALUE = "numerical_value"
CANDIDATE_SET_SIZE = "candidate_set_size"
CONDITIONS = (EVENT_TYPE, NUMERICAL_VALUE, CANDIDATE_SET_SIZE)
CONDITION_LABEL = {EVENT_TYPE: "Event type", NUMERICAL_VALUE: "Numerical value", CANDIDATE_SET_SIZE: "Candidate-set size"}
CONDITION_OF_DATA_KEY = {"type_outlier": EVENT_TYPE, "numerical_outlier": NUMERICAL_VALUE, "density": CANDIDATE_SET_SIZE}
DATA_KEY_OF_CONDITION = {v: k for k, v in CONDITION_OF_DATA_KEY.items()}


def condition_of(record: dict) -> str:
    return CONDITION_OF_DATA_KEY[record["family"]]
