"""Both historical methods must receive the same response-independent records."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import history_transfer_experiment as runner
from experiments.history import build_source, uniform_record_ids
from experiments.rss import measurement_contract
from experiments.scenarios import sample
from experiments.storage import read_json, write_json
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


def archives(n=8, sources=2):
    scenes = sample("car_following", n, "history", replicate=401)
    return [
        ResponseArchive(
            source_id=f"source_{s}",
            family="rule",
            lineage="idm",
            role="history",
            scenarios=scenes,
            collision=np.arange(n) % (s + 2) == 0,
            safety=np.ones((n, 2)),
            rss_margin=np.linspace(-0.9, 2, n),
            rss_contract=measurement_contract(),
            provenance={"record_ids": [f"source_{s}:{i:04d}" for i in range(n)]},
        )
        for s in range(sources)
    ]


def test_uniform_selection_is_order_independent_and_uses_the_registered_rng():
    ids = [f"id:{i:04d}" for i in range(768)]
    expected = np.sort(np.random.default_rng(1301).choice(768, 512, replace=False))
    chosen = uniform_record_ids(ids)
    assert chosen == [ids[i] for i in expected]
    assert chosen == uniform_record_ids(ids[::-1])
    assert chosen != uniform_record_ids(ids, seed=1302)
    assert uniform_record_ids(ids[:512]) == ids[:512]
    assert uniform_record_ids(ids[:2]) == ids[:2]


@pytest.mark.parametrize("ids,limit", [([], 2), (["a", "a"], 2), (["a"], 0)])
def test_uniform_selection_rejects_ambiguous_inputs(ids, limit):
    with pytest.raises(ValueError, match="unique nonempty"):
        uniform_record_ids(ids, limit=limit)


@pytest.mark.parametrize(
    "setting,value", [("HISTORY_VIEW_SEED", 1302), ("HISTORY_VIEW_LIMIT", 256)]
)
def test_pre_registered_view_protocol_rejects_setting_drift(tmp_path, monkeypatch, setting, value):
    runner.register_history_view_protocol(tmp_path)
    monkeypatch.setattr(runner, setting, value)
    with pytest.raises(ValueError, match="protocol changed"):
        runner.registered_history_views(tmp_path, "car_following", archives())


def test_all_eight_sources_use_512_of_768_without_response_stratification(tmp_path):
    history = archives(768, 8)
    views, spec = runner.registered_history_views(tmp_path, "car_following", history)
    assert len(views) == 8 and sum(len(a.scenarios) for a in views) == 4096
    changed = [replace(a, collision=~a.collision, rss_margin=a.rss_margin[::-1]) for a in history]
    _, other = runner.registered_history_views(tmp_path / "other", "car_following", changed)
    assert spec["selected_record_ids"] == other["selected_record_ids"]
    assert spec["input_record_ids"] == {a.source_id: a.provenance["record_ids"] for a in history}
    with pytest.raises(ValueError, match="protocol changed"):
        runner.registered_history_views(tmp_path, "car_following", changed)
    path = tmp_path / "reports/history_views/car_following.json"
    tampered = read_json(path)
    tampered["selected_record_ids"]["source_0"].reverse()
    write_json(path, tampered)
    with pytest.raises(ValueError, match="protocol changed"):
        runner.registered_history_views(tmp_path, "car_following", history)


def test_source_posterior_and_pooled_training_use_identical_ids_and_responses(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "HISTORY_VIEW_LIMIT", 4)
    history = archives()
    views, spec = runner.registered_history_views(tmp_path, "car_following", history)
    pooled = runner.pooled_history(views, spec)
    model = SharedPrivateKernel(history[0].x.shape[1]).double()
    candidates = sample("car_following", 3, "new", replicate=402)
    source = build_source(
        model,
        history,
        candidates,
        view_record_ids=spec["selected_record_ids"],
        cache_path=tmp_path / "priors.npz",
    )
    selected = source["specification"]["view_record_ids"]
    assert selected == spec["selected_record_ids"]
    assert pooled["record_ids"] == [r for a in history for r in selected[a.source_id]]
    for i, a in enumerate(history):
        positions = [a.provenance["record_ids"].index(r) for r in selected[a.source_id]]
        assert np.array_equal(pooled["collision"][4 * i : 4 * (i + 1)], a.collision[positions])
        assert np.array_equal(pooled["rss_margin"][4 * i : 4 * (i + 1)], a.rss_margin[positions])
    changed = {name: ids[::-1] for name, ids in selected.items()}
    with pytest.raises(ValueError, match="snapshot"):
        build_source(
            model, history, candidates, view_record_ids=changed, cache_path=tmp_path / "priors.npz"
        )


def test_formal_prediction_and_budget_entries_pass_the_registered_ids(tmp_path, monkeypatch):
    from experiments import transfer_batches as batches

    history = archives()
    model = SharedPrivateKernel(history[0].x.shape[1]).double()
    sut = SimpleNamespace(sut_id="development")
    bank = SimpleNamespace(
        candidates=sample("car_following", 101, "new", replicate=402),
        snapshot_fingerprint="fixture",
        manifest={"sut_id": sut.sut_id, "contract": {}, "environment_seed": 81001},
    )
    monkeypatch.setattr(runner, "HISTORY_VIEW_LIMIT", 4)
    monkeypatch.setattr(batches, "FORMAL_TARGET_SUTS", (sut,))
    monkeypatch.setattr(batches, "TARGET_COUNT", len(bank.candidates))
    monkeypatch.setattr(batches, "models_and_history", lambda *_: ((model, model), history))
    monkeypatch.setattr(batches, "ExecutionBank", lambda _: bank)
    monkeypatch.setattr(batches, "execution_contract", lambda *_, **__: {})
    monkeypatch.setattr(
        batches, "registered_populations", lambda _: {("target", "car_following"): bank.candidates}
    )
    monkeypatch.setattr(batches, "classifier_input", lambda *_: {})
    prediction, prediction_banks = batches.prepare_context(tmp_path, "car_following", "prediction")
    budget, budget_banks = batches.prepare_context(tmp_path, "car_following", "budget")
    mapping = prediction["sources"]["specification"]["view_record_ids"]
    assert all(len(ids) == 4 for ids in mapping.values())
    assert budget["sources"]["specification"]["view_record_ids"] == mapping
    assert prediction["pooled"]["record_ids"] == budget["pooled"]["record_ids"]
    assert prediction["source_resource_receipt"] == budget["source_resource_receipt"]
    assert prediction_banks == budget_banks == {sut.sut_id: bank}


def test_prediction_resume_rejects_a_different_history_view(tmp_path):
    history = archives()
    model = SharedPrivateKernel(history[0].x.shape[1]).double()
    views, spec = runner.registered_history_views(tmp_path, "car_following", history)
    pooled = runner.pooled_history(views, spec)
    bank = SimpleNamespace(
        manifest={"sut_id": "development", "contract": {}},
        candidates=history[0].scenarios,
        snapshot_fingerprint="fixture",
    )
    from experiments import transfer_execution as execution
    from experiments.k_shot import split_candidates
    from experiments.transfer_settings import native_settings

    settings = native_settings()
    case = execution.prediction_case(
        bank, split_candidates(bank.candidates, 11, budget=2), 11, "pooled_history",
        (model, None), pooled, None, settings=settings, checkpoints=(0, 2),
        limits={"wall_s": 120, "child_memory_bytes": 2**30},
    )
    pooled = {**pooled, "view_specification": {"records": "different"}}
    with pytest.raises(ValueError, match="pre-query task declaration"):
        execution.run_prediction(
            tmp_path, case, bank, (model, None), pooled, None,
            settings=settings, paid_execution=None,
        )
