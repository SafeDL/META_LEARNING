"""Endpoint evaluation is independent of the order of discoveries."""

import json

import numpy as np

from methods.baselines.dart.evaluation import plots, report
from methods.baselines.dart.evaluation.prepare import independent_scenes
from benchmark.evaluation import metrics
from benchmark.pools import scene_pool


def test_independent_pool_is_disjoint_and_preserves_physics():
    scenes = independent_scenes()
    assert len(scenes) == 2048
    assert len({s["scenario_id"] for s in scenes}) == 2048
    x = np.array([s["numeric_input"] for s in scenes])
    assert np.sum(x[:, 4] == 0) == 1024 and np.sum(x[:, 4] == 1) == 1024
    assert np.all((x[:, :4] >= 0) & (x[:, :4] <= 1))
    for pool in (0, 1):
        original = scene_pool(pool)
        assert scenes[0]["fixed_context"] == original[0]["fixed_context"]
        assert not set(map(tuple, x)).intersection(
            tuple(s["numeric_input"]) for s in original
        )


def test_same_discovered_set_has_same_count_and_coverage():
    labels = np.arange(210) % 3 == 0
    selected = np.arange(200)
    forward = metrics(labels, selected)
    backward = metrics(labels, selected[::-1])
    assert forward["F200"] == backward["F200"]
    assert forward["recall_200"] == backward["recall_200"]
    assert labels[selected[0]] != labels[selected[-1]]
    assert not any("area" in key for key in forward)


def test_acceptance_uses_only_terminal_count_and_coverage(tmp_path, monkeypatch):
    bank = tmp_path / "bank" / "pool_0" / "demo"
    bank.mkdir(parents=True)
    risk = np.r_[np.full(200, 0.8), np.zeros(10)]
    np.savez(bank / "responses.npz", risk=risk, collision=np.zeros(210, bool))
    output = tmp_path / "report"
    output.mkdir()
    paths = {}
    for policy, selected in (
        ("cg_brt", list(range(200))),
        ("full", [*range(200, 210), *range(190)]),
    ):
        paths[policy] = tmp_path / f"{policy}.json"
        paths[policy].write_text(
            json.dumps(
                {
                    "selected_indices": selected,
                    "queries": [{"index": i, "risk": float(risk[i])} for i in selected],
                    "decisions": [
                        {"reference_index": i, "selected_index": i} for i in selected
                    ],
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(report, "ROOT", output)
    monkeypatch.setattr(report, "SUT_IDS", ("demo",))
    monkeypatch.setattr(report, "SEEDS", (11,))
    monkeypatch.setattr(report, "pools", lambda stage: (0,))
    monkeypatch.setattr(report, "variants", lambda stage: ("cg_brt", "full"))
    monkeypatch.setattr(report, "bank_root", lambda stage: tmp_path / "bank")
    monkeypatch.setattr(
        report, "trace_path", lambda stage, pool, sut, policy, seed: paths[policy]
    )
    monkeypatch.setattr(plots, "plot_report", lambda result: None)
    result = report.main()
    assert result["effect_accepted"]
    assert result["retention"] == {"H200": 0.95, "high_risk_recall_pct": 0.95}
    assert "Area" not in (output / "development_comparison.md").read_text(
        encoding="utf-8"
    )
