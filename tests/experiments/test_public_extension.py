"""Public replay preserves feedback permissions and the fixed method matrix."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.budget import protocol
from experiments.formal_scenarios import map_scene
from experiments.public_ads import extend
from experiments.public_ads.extension_report import descriptive
from experiments.regression.fixed_inputs import methods_for


def test_public_target_only_receives_no_historical_models(monkeypatch):
    candidates = [map_scene("car_following_braking_onset", np.full(4, i / 200), str(i))[0]
                  for i in range(200)]
    bank = SimpleNamespace(candidates=candidates, snapshot_fingerprint="snapshot",
                           manifest={"sut_id": "online_mc_ppo", "contract": {}, "environment_seed": 81001})
    monkeypatch.setattr(extend, "_CONTEXT", {"models": (object(), object()), "sources": object()})
    case, public = extend.budget_inputs(bank, "target_only")
    assert public["pretrained"] is None and public["sources"] is None
    assert case.specification["models"] is None and case.specification["source_snapshot"] is None
    assert case.specification["budget"] == 200 and len(public["initial_indices"]) == 20
    assert case.specification["device"] == "cpu"


def test_public_path_binding_restores_sealed_readers_after_error():
    original = protocol.INPUT, protocol.OUTPUT
    with pytest.raises(RuntimeError):
        with extend.public_paths():
            assert protocol.INPUT == extend.OUTPUT and protocol.OUTPUT == extend.BUDGET
            raise RuntimeError("failed read")
    assert (protocol.INPUT, protocol.OUTPUT) == original


def test_public_summary_preserves_negative_and_tied_pairs():
    rows = [dict(target="a", scenario=str(s), method=m, C200=(10 + s if m == "full" else 12),
                 AUC=(5 + s if m == "full" else 7))
            for s in range(4) for m in protocol.METHODS]
    summary = descriptive(rows)
    assert summary["ours_minus"]["target_only"] == dict(
        gain=[-.5, -.5], wins=[1, 1], ties=[1, 1], losses=[2, 2])
    assert "spectre" in methods_for("dqn_continual")
    assert len(methods_for("dqn_continual")) == len(methods_for("ppo")) == 5
