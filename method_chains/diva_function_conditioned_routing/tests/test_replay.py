import csv
from pathlib import Path

from diva_highway_env.data.generate_anchor_bank import FUNCTIONAL_MODES
from diva_highway_env.data.response_bank import ResponseBank
from method_chains.diva_function_conditioned_routing.replay import (
    METHOD,
    select_replay_cases,
)


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = (
    PACKAGE.parents[1]
    / "results"
    / "method_chains"
    / PACKAGE.name
    / "functional_shift_benchmark"
)


def test_replays_are_dangerous_cases_selected_by_the_proposed_method() -> None:
    bank = ResponseBank.load(RESULTS / "response_bank.npz")
    with (RESULTS / "mining_results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    cases = select_replay_cases(bank, rows)
    assert tuple(case.mode for case in cases) == FUNCTIONAL_MODES
    selected = {
        (row["target_sut"], int(index))
        for row in rows
        if row["method"] == METHOD
        for index in row["queried_indices"].split(";")
    }
    assert all((case.target_sut, case.anchor_index) in selected for case in cases)
    assert all(case.collision or case.near_miss for case in cases)
