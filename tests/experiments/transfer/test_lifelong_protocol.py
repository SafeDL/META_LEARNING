import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.accumulation import lifelong
from experiments.accumulation.lifelong import _blocks, sequences
from experiments.lifelong.report import CONTINUAL_METRICS, _continual_effect
from experiments.lifelong_transfer import CROSS_ARCHITECTURE, TARGETS
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.schema import Scenario


@pytest.mark.parametrize("study,options,expected", [
    ("in_family", ["--workers", "1"], 1),
    ("in_family", [], lifelong.METHOD_WORKERS),
    ("cold_start", [], min(12, lifelong.COLLECTION_WORKERS)),
])
def test_cli_applies_branch_concurrency(study, options, expected, tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    pool = MagicMock()
    monkeypatch.setattr(sys, "argv", ["lifelong", study, *options])
    monkeypatch.setattr(lifelong, "ROOT", tmp_path)
    monkeypatch.setattr(lifelong, "verify_stage_two", lambda: None)
    monkeypatch.setattr(lifelong, "verify_study", lambda study: 0)
    monkeypatch.setattr(lifelong, "_models", lambda kind, study: (None, None))
    monkeypatch.setattr(lifelong, "prepare_processes", lambda: None)
    monkeypatch.setattr(lifelong, "initialize_worker", lambda: None)
    monkeypatch.setattr(lifelong, "ProcessPoolExecutor", pool)
    lifelong.main()
    assert pool.call_args.kwargs["max_workers"] == expected


def test_cold_start_sequences_use_all_twelve_targets():
    branches = sequences("cold_start")
    assert len(branches) == 3
    assert all(len(branch.target_ids) == 4 and tuple(range(4)) == branch.block_ids for branch in branches)
    assert sorted(target for branch in branches for target in branch.target_ids) == sorted(TARGETS)


def test_relay_holds_out_the_last_100_points_of_each_registered_block():
    scenes = [Scenario(f"scene:{i:04d}", "test", {"coordinate": i}) for i in range(2000)]
    banks = [SimpleNamespace(candidates=scenes), SimpleNamespace(candidates=scenes[::-1])]
    relay, terminal = _blocks(banks, "cross_architecture")[:4], _blocks(banks, "cross_architecture")[-1]
    order = np.random.default_rng(76023).permutation(2000)
    expected = [[scenes[index].scenario_id for index in position] for position in np.split(order, 4)]
    for block, registered in zip(relay, expected, strict=True):
        assert block.tolist() == registered[:400]
    assert terminal.tolist() == [name for position in expected for name in position[400:]]
    assert len({name for block in (*relay, terminal) for name in block}) == 2000
    forward, reverse = sequences("cross_architecture")
    assert forward.target_ids == (*CROSS_ARCHITECTURE, "target_ppo")
    assert reverse.target_ids == (*CROSS_ARCHITECTURE[::-1], "target_ppo")
    assert forward.block_ids == (0, 1, 2, 3, 4)
    assert reverse.block_ids == (3, 2, 1, 0, 4)


def test_in_family_reverse_reuses_the_registered_fixed_candidate_blocks():
    branches = {branch.name: branch for branch in sequences("in_family")}
    for group in ("group_a", "group_b"):
        forward, reverse = (branches[f"{group}_{order}"] for order in ("forward", "reverse"))
        assert forward.target_ids == reverse.target_ids[::-1]
        assert dict(zip(forward.target_ids, forward.block_ids, strict=True)) == dict(
            zip(reverse.target_ids, reverse.block_ids, strict=True)
        )


def test_continual_effect_preserves_signed_differences_and_undefined_recall():
    values = dict.fromkeys(CONTINUAL_METRICS, 0)
    rows = [
        dict(values, target=TARGETS[0], scenario=FORMAL_SCENARIOS[0], k0_nll=-0.25, c50=2, k0_recall_50=None),
        dict(values, target=TARGETS[1], scenario=FORMAL_SCENARIOS[0], k0_nll=-0.5, c50=0, k0_recall_50=0.25),
    ]
    effects = _continual_effect(rows)
    assert effects["k0_nll"]["mean"] == -0.375
    assert effects["k0_nll"]["wins"] == 2
    assert effects["c50"]["mean"] == 1
    assert effects["k0_recall_50"]["ads"] == effects["k0_recall_50"]["pairs"] == 1
    assert effects["k0_recall_50"]["ci_low"] is None
