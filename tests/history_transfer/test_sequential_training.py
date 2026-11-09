from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.history import model_version, update_shared_model
from experiments.rss import measurement_contract
from experiments.scenarios import sample
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer import training
from testing_methods.history_transfer.inference import sequential_probability
from testing_methods.history_transfer.model import JointResponseKernel


def test_reviewed_cleanup_reuses_checkpoint_and_rejects_other_scope(tmp_path, monkeypatch):
    from experiments import source_registration
    from experiments.storage import write_json

    registration = tmp_path / "source_changes.json"
    write_json(
        registration,
        {"source_changes": {"training.py": {"before_sha256": "old", "after_sha256": "new"}}},
    )
    monkeypatch.setattr(source_registration, "CODE_MAINTENANCE", registration)
    monkeypatch.setattr(
        training,
        "training_contract",
        lambda kernel_type=JointResponseKernel: {"training.py": "new"},
    )
    spec = {
        "training_contract": {"training.py": "old"},
        "continuous_response": measurement_contract(),
        "config": {"steps": 1},
        "scenario": "car_following",
    }
    model = JointResponseKernel(3).double()
    path = tmp_path / "target_kernel.pt"
    training.save_model(path, model, "target", spec, 1)
    report = {"status": "complete", "specification": spec}
    write_json(path.with_suffix(".json"), report)
    original = path.read_bytes(), path.with_suffix(".json").read_bytes()
    current = {**spec, "training_contract": training.training_contract()}
    assert training.completed_report(path, current) == report
    loaded, _ = training.load_model(path)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    assert (path.read_bytes(), path.with_suffix(".json").read_bytes()) == original
    with pytest.raises(ValueError, match="another source scope"):
        training.completed_report(path, {**current, "scenario": "front_brake"})


def archive(name, n=8, *, offset=0):
    scenes = sample("car_following", n, "history", replicate=offset)
    labels = np.arange(n) % 3 == 0
    return ResponseArchive(
        name,
        "rule",
        name,
        "history",
        scenes,
        labels,
        np.ones((n, 2)),
        rss_margin=np.where(labels, -0.9, 0.2),
        rss_contract=measurement_contract(),
    )


def test_continuous_episode_length_and_windows_cover_all_historical_records():
    history = archive("a", 768)
    config = training.Config()
    assert training.episode_layout(history, config) == (20, 200)
    windows = [training.selected_indices(history, config, config.seed, i) for i in range(4)]
    assert set(np.concatenate(windows)) == set(range(768))
    assert all(len(set(window)) == 200 for window in windows)
    np.testing.assert_array_equal(
        windows[0], training.selected_indices(history, config, config.seed)
    )
    completed = replace(
        archive("b", 200),
        role="target",
        provenance={"completed_task": True, "original_role": "target"},
    )
    for segment in (0, 1, 5):
        np.testing.assert_array_equal(
            training.selected_indices(completed, config, 99, segment), np.arange(200)
        )
    for n in (2, 8, 32, 128):
        assert training.episode_layout(archive("short", n), config) == (min(20, n - 1), n)
    with pytest.raises(ValueError, match="at least two"):
        training.episode_layout(archive("singleton", 1), config)


def test_historical_rss_ablation_ignores_margins_in_training_sources_and_view(tmp_path):
    from experiments.history import build_source, calculation_view

    historical = [archive("a", 12), archive("b", 12, offset=1)]
    changed = [replace(a, rss_margin=-a.rss_margin + 0.1) for a in historical]
    config = training.Config(
        steps=2,
        initial_count=1,
        budget=4,
        historical_context=3,
        diagnostic_interval=1,
        historical_rss=False,
    )
    models = []
    for name, data in (("original", historical), ("changed", changed)):
        training.train(data, tmp_path / name, config=config)
        pair = [
            training.load_model(tmp_path / name / file)[0]
            for file in (
                "target_kernel.pt",
                "historical_encoder.pt",
            )
        ]
        models.append(pair)
    for original, altered in zip(*models):
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, altered.state_dict()[key], rtol=0, atol=0)
    candidates = sample("car_following", 8, "new")
    priors = [
        build_source(
            pair[1],
            data,
            candidates,
            limit=5,
            use_rss=False,
        )
        for pair, data in zip(models, (historical, changed))
    ]
    for key in ("means", "blocks", "weights"):
        np.testing.assert_array_equal(priors[0][key], priors[1][key])
    for first, second in zip(historical, changed):
        assert (
            calculation_view(first, limit=5, use_rss=False).provenance["calculation_indices"]
            == (calculation_view(second, limit=5, use_rss=False).provenance["calculation_indices"])
        )
    with pytest.raises(ValueError, match="Source feedback"):
        build_source(models[0][1], historical, candidates)


def test_next_response_loss_differentiates_through_prequery_source_weight_updates():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    config = training.Config(initial_count=2, budget=5, historical_context=4)
    historical = [archive("a"), archive("b", 6, offset=1)]
    base = JointResponseKernel(3).double()
    episode = training.cache_episode(base, historical[0], config, config.seed)
    training.ensure_states(episode, 3)
    means = torch.tensor(
        [[[-1.0, 0.2], [0.7, -0.3], [-0.4, 0.5], [0.2, -0.2], [-0.1, 0.1]]],
        requires_grad=True,
        dtype=torch.float64,
    )
    blocks = torch.eye(2, dtype=torch.float64).expand(1, 5, 2, 2).clone()
    sources = {"names": ["b"], "weights": np.ones(1), "means": means, "blocks": blocks}
    probability = sequential_probability(
        sources,
        episode["states"],
        episode["reference"],
        episode["x"],
        episode["labels"],
        episode["margins"],
        3,
    )
    (gradient,) = torch.autograd.grad(probability, means)
    assert gradient[0, :3].abs().max() > 1e-8
    assert gradient[0, 4].abs().max() == 0
    assert torch.isfinite(gradient).all()
    contexts = training.contexts_for_episode(historical, episode, np.random.default_rng(1), config)
    assert [a.source_id for a in contexts] == ["b"]


def test_final_checkpoint_is_saved_even_when_later_training_loss_is_worse(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = training.Config(
        steps=2, initial_count=1, budget=3, diagnostic_interval=1, historical_context=2
    )
    historical = [archive("a"), archive("b", offset=1)]
    actual = training.episode_probability
    calls = []

    def supervised(encoder, contexts, episode, count):
        calls.append(count)
        p = actual(encoder, contexts, episode, count)
        truth = bool(episode["labels"][count])
        ideal = 0.99 if truth else 0.01
        if len(calls) == 2:
            ideal = 1 - ideal
        return p * 0.001 + p.new_tensor(ideal * 0.999)

    monkeypatch.setattr(training, "episode_probability", supervised)
    report = training.train(historical, tmp_path, config=config)
    _, saved = training.load_model(tmp_path / "historical_encoder.pt")
    assert (
        report["encoder"]["curve"][1]["training_nll"]
        > report["encoder"]["curve"][0]["training_nll"]
    )
    assert saved["step"] == 2
    assert "development_fingerprints" not in saved["specification"]
    assert "support_counts" not in saved["specification"]["config"]
    assert training.train(historical, tmp_path, config=config) == report


def test_equal_old_new_task_schedule_and_numeric_rejection_preserve_parent(tmp_path, monkeypatch):
    history = [archive(f"old{i}") for i in range(4)] + [archive("new")]
    chosen = [training.scheduled_item(history, ["new"], step).source_id for step in range(1, 9)]
    assert chosen[::2] == ["new"] * 4
    assert set(chosen[1::2]) == {f"old{i}" for i in range(4)}
    models = tuple(JointResponseKernel(3).double() for _ in range(2))
    before = [model_version(m) for m in models]
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise FloatingPointError("nonfinite fixture")

    monkeypatch.setattr(training, "train", fail)
    for _ in range(2):
        active, publication = update_shared_model(
            tmp_path / "candidate", history, models, new_source_ids=["new"]
        )
        assert active is models and not publication["accepted"]
        assert publication["reason"] == "numerical_failure"
    assert len(calls) == 1
    assert [model_version(m) for m in models] == before
    assert not (tmp_path / "active.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_both_training_stages_support_cuda_without_development_archives(tmp_path):
    history = [archive("a", 3), archive("b", 4, offset=1)]
    config = training.Config(
        steps=1, initial_count=1, budget=3, historical_context=2, device="cuda"
    )
    report = training.train(history, tmp_path, config=config)
    for name in ("target_kernel.pt", "historical_encoder.pt"):
        model, saved = training.load_model(tmp_path / name, device="cuda")
        assert model.mean.is_cuda
        assert saved["specification"]["training_device"] == "cuda"
    assert report["encoder"]["status"] == "complete"


def test_four_scenario_parallel_construction_matches_serial_and_records_costs(tmp_path):
    from experiments.history import build_source, load_training_archives
    from experiments.scenarios import TEST_SCENARIOS, candidate_pool
    from experiments.storage import read_json, save_archive, write_json
    from testing_methods.common.response import archive_fingerprints

    entries = []
    for kind in TEST_SCENARIOS:
        for i in range(2):
            item = archive(f"history_{i}", 3, offset=i)
            item.scenarios = candidate_pool(kind, 3, replicate=90 + i)
            path = tmp_path / "data" / kind / f"{item.source_id}.npz"
            save_archive(path, item, provenance=item.provenance)
            entries.append(
                {
                    "scenario": kind,
                    "source_id": item.source_id,
                    "path": path.relative_to(tmp_path / "data").as_posix(),
                    "fingerprint": archive_fingerprints([item])[item.source_id],
                }
            )
    manifest = tmp_path / "data/manifest.json"
    write_json(manifest, {"archives": entries})
    config = training.Config(
        steps=1, initial_count=1, budget=3, diagnostic_interval=1, historical_context=2
    )
    root = tmp_path / "parallel"
    reports = training.main(config=config, dataset_manifest=manifest, model_root=root)
    assert [r["scenario"] for r in reports] == list(TEST_SCENARIOS)
    saved = read_json(root / "construction.json")
    assert saved["status"] == "complete" and saved["parallel_workers"] == 4
    assert saved["new_physical_calls"] == 0
    assert all(r["resources"]["sampled_peak_cpu_bytes"] > 0 for r in reports)
    for report in reports:
        assert report["target"]["specification"]["scenario"] == report["scenario"]
        assert report["encoder"]["specification"]["scenario"] == report["scenario"]
    training.build_scope(
        "front_brake", config=config, dataset_manifest=manifest, model_root=tmp_path / "serial"
    )
    for name in ("target_kernel.pt", "historical_encoder.pt"):
        parallel, _ = training.load_model(root / "front_brake/all_sources" / name)
        serial, _ = training.load_model(tmp_path / "serial/front_brake/all_sources" / name)
        assert model_version(parallel) == model_version(serial)
    cut_out = load_training_archives(manifest, "cut_out")
    candidates = candidate_pool("cut_out", 8, replicate=94)
    with pytest.raises(ValueError, match="another functional scenario"):
        build_source(serial, cut_out, candidates)
    with pytest.raises(ValueError, match="one functional scenario"):
        build_source(
            JointResponseKernel(6).double(),
            load_training_archives(manifest, "front_brake"),
            candidates,
        )
    with pytest.raises(ValueError, match="another functional scenario"):
        training.train(
            cut_out, tmp_path / "invalid", config=config, initial_models=(serial, serial)
        )
    assert not (tmp_path / "invalid").exists()
