import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import expit

from methods.common.prior import TransferPrior, physical_kernel
from methods.gcrt.controller import (
    StopController,
    block_scores,
    fit_tilt,
    threshold,
)
from methods.gcrt.model import (
    RemainingDiscoveries,
    log_pmf,
    snapshot,
    tilted_log_pmf,
    weighted_loss,
)
from methods.gcrt.posterior import JointPosterior
from methods.gcrt.selection import acquisition_values, select, shortlist


def prior(n=18):
    x = np.random.default_rng(11).random((n, 5))
    x[:, 4] = 0
    k = physical_kernel(x)
    archive = k * 0.2
    means = np.column_stack((np.linspace(0.1, 0.9, n), np.linspace(0.8, 0.2, n)))
    return TransferPrior(
        x, means, np.array([0.5, 0.5]), ("a", "b"), k, k, (archive, archive)
    )


def test_virtual_real_updates_match_without_mutation():
    model = JointPosterior(prior())
    model.observe(1, 0.75)
    before = copy.deepcopy(model)
    indices = np.array([3, 6, 9])
    predicted = model.conditional_probabilities(2, np.array([0.2, 0.8]), indices)
    for row, risk in enumerate((0.2, 0.8)):
        clone = copy.deepcopy(model)
        clone.observe(2, risk)
        np.testing.assert_allclose(
            predicted[row], clone.probabilities()[indices], atol=1e-12
        )
    np.testing.assert_array_equal(model.probabilities(), before.probabilities())
    assert np.isfinite(
        model.conditional_probabilities(2, np.array([-2.0, 3.0]), indices)
    ).all()


def test_shortlist_last_budget_and_target_independence():
    model = JointPosterior(prior(30))
    remaining = np.ones(30, bool)
    candidates = shortlist(model, remaining)
    assert len(candidates) <= 24
    assert np.argmax(model.probabilities()) in candidates
    np.testing.assert_array_equal(
        acquisition_values(model, candidates, remaining, 1, 11),
        model.probabilities()[candidates],
    )
    changed = prior(30)
    changed.means[:] = 10
    np.testing.assert_array_equal(
        model.component_probabilities()[:, -1],
        JointPosterior(changed).component_probabilities()[:, -1],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_and_cpu_conditional_lookahead_agree():
    model = JointPosterior(prior(30))
    model.observe(1, 0.3)
    remaining = np.ones(30, bool)
    remaining[1] = False
    candidates = shortlist(model, remaining)
    cpu = acquisition_values(model, candidates, remaining, 10, 11, gpu=False)
    gpu = acquisition_values(model, candidates, remaining, 10, 11, gpu=True)
    np.testing.assert_allclose(cpu, gpu, atol=1e-10)


def test_invariant_encoder_gradients_and_count_normalization():
    model = JointPosterior(prior())
    state = snapshot(model, np.ones(18, bool), [], [], 10, [0, 1], 0)
    network = RemainingDiscoveries()
    output = network.predict(state, [1, 5, 10])
    changed = copy.deepcopy(state)
    changed["tokens"] = changed["tokens"][:, [2, 0, 1]]
    torch.testing.assert_close(output, network.predict(changed, [1, 5, 10]))
    loss = weighted_loss(output, [1, 5, 10], [0, 2, 4])
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in network.parameters()
    )
    for parameters, b in zip(output, [1, 5, 10]):
        p = tilted_log_pmf(log_pmf(parameters, b), 1.7).exp()
        assert abs(float(p.sum()) - 1) < 1e-12


def test_tilt_is_fitted_from_base_probabilities_and_has_unique_root():
    logits, labels = np.array([-2.0, -1.0, 0.0, 1.0]), np.array([1, 1, 1, 0])
    u = fit_tilt(logits, labels, 0.1)
    assert u > 0
    assert abs((expit(logits + u) - labels).sum() + 0.1 * u) < 1e-9
    assert fit_tilt(logits, labels, 0.1) == u


def test_calibration_units_strict_ties_and_integer_retention():
    rows = [
        dict(sut="a", full_found=20, forecast=[dict(found=18, score=0.8)]),
        dict(sut="a", full_found=20, forecast=[dict(found=19, score=0.99)]),
        dict(sut="b", full_found=0, forecast=[dict(found=0, score=0.99)]),
    ]
    assert block_scores(rows) == {"a": 0.8, "b": 0.0}
    assert threshold([0.5, 0.8]) == 0.8
    assert threshold([0.5] * 18, "strict") == 1
    assert threshold([0.5] * 19, "strict") == 0.5
    assert 20 * 19 == 19 * 20


def test_online_stopping_is_full_prefix_with_initial_correction():
    torch.manual_seed(11)
    network = RemainingDiscoveries().eval()
    oracle = SimpleNamespace(query=lambda i: SimpleNamespace(risk=float(i % 2)))
    fixed = select(prior(24), oracle, budget=20, initial_indices=list(range(10)))
    stopper = StopController(network, -1, 1.0)
    adaptive = select(
        prior(24), oracle, budget=20, initial_indices=list(range(10)), stopper=stopper
    )
    assert adaptive["selected_indices"] == fixed["selected_indices"][:10]
    assert len(stopper.labels) == 2
    assert adaptive["stopping"][-1]["tilt"] == stopper.u
    assert adaptive["stop_reason"] == "quality_retained"
    assert len(set(fixed["selected_indices"])) == 20
    strict = select(
        prior(24),
        oracle,
        budget=20,
        initial_indices=list(range(10)),
        stopper=StopController(network, 1.0, 1.0),
    )
    assert strict["selected_indices"] == fixed["selected_indices"]
    with pytest.raises(ValueError):
        select(prior(), oracle, budget=10, initial_indices=[0, 0])


def test_selective_label_weighting_expectation():
    parameters = torch.tensor([[0.0, 1.2, 2.0], [0.1, 2.0, 1.0]], dtype=torch.float64)
    full = weighted_loss(parameters, [1, 5], [1, 3])
    ordinary = weighted_loss(parameters, [1, 5], [1, 0], [True, False], [1, 0.5])
    complete = weighted_loss(parameters, [1, 5], [1, 3], [True, True], [1, 0.5])
    torch.testing.assert_close((ordinary + complete) / 2, full)


def test_information_value_changes_selection_in_enumerable_example():
    class BinaryModel:
        def probabilities(self):
            return np.array([0.5, 0.5, 0.6])

        def marginal_draws(self, indices, count, generator):
            # Exact integration of A=B~Bernoulli(.5), independent C~Bernoulli(.6).
            return np.tile(np.r_[np.zeros(8), np.ones(8)][:, None], (1, len(indices)))

        def conditional_probabilities(self, index, values, indices):
            output = np.tile(self.probabilities()[indices], (len(values), 1))
            if index in (0, 1):
                output[:, np.asarray(indices) < 2] = values[:, None]
            return output

    values = acquisition_values(
        BinaryModel(), np.arange(3), np.ones(3, bool), 2, 11, gpu=False
    )
    np.testing.assert_allclose(values, [1.3, 1.3, 1.1])
    assert np.argmax(values) != np.argmax(BinaryModel().probabilities())


def test_task_level_rank_calibration_with_discrete_ties():
    from itertools import product

    good = 0
    for values in product((0.1, 0.4, 0.7, 0.9), repeat=4):
        gamma = threshold(values[:3], "strict", alpha=0.25)
        good += values[-1] <= gamma
    assert good / 4**4 >= 0.75


def test_missing_labels_do_not_change_weighted_training_loss():
    parameters = torch.tensor([[0.0, 1.2, 2.0], [0.1, 2.0, 1.0]], requires_grad=True)
    first = weighted_loss(parameters, [1, 5], [1, 0], [True, False], [1, 0.5])
    other = weighted_loss(parameters, [1, 5], [1, 5], [True, False], [1, 0.5])
    torch.testing.assert_close(first, other)
    first.backward()
    assert torch.equal(parameters.grad[1], torch.zeros(3))


def test_complete_build_repairs_missing_artifacts_without_value_training(
    tmp_path, monkeypatch
):
    from methods.gcrt import build
    from methods.common.workspace import HistoricalWorkspace
    from methods.baselines.hgt.io import write_json

    calls = []
    monkeypatch.setattr(build, "ARCHIVE_ROOT", tmp_path / "absent_archive")
    workspace = HistoricalWorkspace(
        tmp_path / "cutin", 0, {"source": {"x": np.zeros((1, 5)), "risk": np.zeros(1)}}
    )

    def base(workspace, kind):
        calls.append(kind)
        path = (
            workspace.root
            / kind
            / "models"
            / ("seed_11.pt" if kind == "kernel" else "decision_11.pt")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"base")

    def fit(workspace):
        calls.append("prior")
        write_json(workspace.root / "prior.json", {})

    def generate(workspace):
        calls.append("historical")
        for condition in workspace.archives:
            path = workspace.root / "historical" / f"source_{condition}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")
        write_json(workspace.root / "policy.json", {"selected": "segment"})

    def train(workspace):
        calls.append("count")
        (workspace.root / "model.pt").write_bytes(b"model")
        write_json(workspace.root / "training.json", {})

    def measure(workspace):
        write_json(workspace.root / "historical_cost.json", {})

    for name, action in (
        ("ensure_base", base),
        ("fit", fit),
        ("generate", generate),
        ("train", train),
        ("construction_components", measure),
    ):
        monkeypatch.setattr(build, name, action)
    build.build_family(workspace)
    assert calls == ["kernel", "encoder", "prior", "historical", "count"]
    build.build_family(workspace)
    assert len(calls) == 5
    (workspace.root / "model.pt").unlink()
    build.build_family(workspace)
    assert calls[-1] == "count" and len(calls) == 6


def test_source_reordering_preserves_joint_updates():
    original = prior()
    reordered = copy.deepcopy(original)
    reordered.means = reordered.means[:, ::-1]
    reordered.weights = reordered.weights[::-1]
    reordered.names = reordered.names[::-1]
    reordered.archive_covariances = reordered.archive_covariances[::-1]
    left, right = JointPosterior(original), JointPosterior(reordered)
    for index, risk in ((0, 0.8), (2, 0.1), (5, 0.9)):
        left.observe(index, risk)
        right.observe(index, risk)
    np.testing.assert_allclose(left.probabilities(), right.probabilities(), atol=1e-12)


def test_stop_prefix_cost_includes_terminal_state_and_excludes_future():
    from methods.gcrt.evaluation import stopping_row

    trace = {
        "found": list(range(1, 201)),
        "positives": 500,
        "sut": "test",
        "pool": 0,
        "timing": {"task_initialization_s": 1, "query_compute_s": [1] * 200},
        "state_recording_s": [3] * 200,
        "execution_s": [2] * 200,
        "selected_indices": list(range(200)),
    }
    prediction = {
        "prequery_s": [0.5] * 200,
        "correction_s": [0.2] * 200,
        "forecast": [
            dict(
                query=t,
                found=t,
                deterministic=(200 - t <= t // 19),
                score=0,
                remaining_mean=200 - t,
                compute_s=0.1,
            )
            for t in range(10, 200)
        ],
    }
    row = stopping_row(trace, prediction, 1)
    assert row["queries"] == 190 and not row["quality_failed"]
    assert row["terminal_decision_s"] == 0.1
    assert row["online_s"] == pytest.approx(459.1)
    assert row["execution_s"] == 380
    prediction["prequery_s"][190:] = [1000] * 10
    assert stopping_row(trace, prediction, 1)["online_s"] == row["online_s"]


def test_isolated_new_selector_discloses_only_200_unique_paid_risks():
    import multiprocessing as mp
    from benchmark.risk_channel import disclose_risk
    from methods.gcrt.evaluation import selector_worker

    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=selector_worker, args=(child,))
    process.start()
    child.close()
    try:
        truth = (np.arange(210) % 2).astype(float)
        result = disclose_risk(
            parent,
            {
                "prior": prior(210),
                "initial_indices": list(range(10)),
                "parameters": {"kappa": 1.0, "target_probability": 0.25},
            },
            truth,
            process,
        )
        assert len(set(result["selected_indices"])) == 200
        assert result["selected_indices"][:10] == list(range(10))
        assert len(result["records"]) == 200
        assert all(r["collision"] is None for r in result["disclosures"])
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()


def test_mean_projection_preserves_rate_shift_at_every_budget():
    from methods.gcrt.model import corrected_log_pmf

    for b in (1, 5, 50, 190):
        base = torch.full((b + 1,), -np.log(b + 1), dtype=torch.float64)
        u = -0.236
        corrected = corrected_log_pmf(base, u)
        mean = float(corrected.exp() @ torch.arange(b + 1, dtype=torch.float64))
        assert mean / b == pytest.approx(expit(u), abs=1e-10)
        assert abs(float(corrected.exp().sum()) - 1) < 1e-12
        if b == 1:
            torch.testing.assert_close(corrected, tilted_log_pmf(base, u))
        plus = corrected_log_pmf(base, u + 1e-5)
        difference = (
            float(
                (plus.exp() - corrected.exp())
                @ torch.arange(b + 1, dtype=torch.float64)
            )
            / b
            / 1e-5
        )
        assert 0 < difference < 0.250001


def test_segment_calibration_uses_only_matured_nonoverlapping_records():
    network = RemainingDiscoveries().eval()
    model = JointPosterior(prior(30))
    state = snapshot(model, np.ones(30, bool), [], [], 20, list(range(10)), 0)
    controller = StopController(network, 1)
    controller.before_state(state)
    assert not controller.labels
    for t, risk in enumerate((1, 0, 1, 0)):
        controller.observe(risk, t + 1)
    assert not controller.labels and controller.u == 0
    controller.observe(1, 5)
    assert controller.labels == [0.6] and controller.pending is None


def test_segment_horizon_and_analytic_lower_bound():
    oracle = SimpleNamespace(query=lambda i: SimpleNamespace(risk=float(i % 2)))
    result = select(prior(30), oracle, budget=20, initial_indices=list(range(10)))
    assert [d["horizon"] for d in result["decisions"][10:]] == [5, 4, 3, 2, 1] * 2
    assert all(d["value_over_static"] >= -1e-10 for d in result["decisions"][10:])


def test_mismatched_calibration_rejected_before_query():
    stopper = StopController(
        RemainingDiscoveries(),
        1,
        contract=dict(
            pool_size=30, budget=200, initial=10, policy="segment", segment=5
        ),
    )
    oracle = SimpleNamespace(
        query=lambda i: pytest.fail("mismatch must precede feedback")
    )
    with pytest.raises(ValueError, match="budget"):
        select(
            prior(30),
            oracle,
            budget=20,
            initial_indices=list(range(10)),
            stopper=stopper,
        )


def test_calibration_is_bound_to_predictor_weights(tmp_path, monkeypatch):
    import hashlib
    from methods.gcrt import train
    from methods.baselines.hgt.io import write_json

    folder = tmp_path / "cutin"
    folder.mkdir()
    weights = folder / "model.pt"
    weights.write_bytes(b"checkpoint")
    contract = dict(
        family="cutin",
        network="initial",
        correction="segment_rate_KL_projection",
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
    )
    write_json(
        folder / "calibration.json",
        {"initial": {"contract": contract, "empirical": 0.9}},
    )
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "load_network", lambda *args: RemainingDiscoveries())
    assert train.load_stopper("cutin", updated=False).gamma == 0.9
    weights.write_bytes(b"another checkpoint")
    with pytest.raises(ValueError, match="Recalibrate"):
        train.load_stopper("cutin", updated=False)


def test_bootstrap_is_charged_once_before_operating_cycles():
    from methods.gcrt.report import campaign_total, campaign_quality

    bootstrap = [dict(adaptive_s=10), dict(adaptive_s=20)]
    ongoing = [dict(adaptive_s=1), dict(adaptive_s=2)]
    assert campaign_total(bootstrap, ongoing, 1, "adaptive_s") == 10
    assert campaign_total(bootstrap, ongoing, 5, "adaptive_s") == 34
    empty = dict(
        sut="empty",
        found=0,
        full_found=0,
        positives=0,
        valid_recall=False,
        recall=None,
        full_recall=None,
    )
    quality = campaign_quality([empty], [empty], 100)
    assert quality["passed"] and quality["macro_recall"] is None


def test_validation_scores_are_native_json_scalars():
    import json
    from methods.gcrt.train import validate

    example = prior(24)
    oracle = SimpleNamespace(query=lambda i: SimpleNamespace(risk=float(i % 2)))
    trace = select(example, oracle, budget=20, initial_indices=list(range(10)))
    # Build a short causal task; the saved state budget defines its true horizon.
    model = JointPosterior(example)
    remaining = np.ones(24, bool)
    queries = []
    evidence = []
    records = []
    for row in trace["queries"]:
        records.append(
            snapshot(
                model, remaining, queries, evidence, 20, list(range(10)), row["index"]
            )
        )
        evidence.append(
            {"prequery_probability": float(model.probabilities()[row["index"]])}
        )
        model.observe(row["index"], row["risk"])
        remaining[row["index"]] = False
        queries.append(row)
    task = dict(
        records=records,
        labels=np.array([q["high_risk"] for q in trace["queries"]]),
        budget=20,
        initial=10,
        group="test",
    )
    scores = validate(RemainingDiscoveries().eval(), [task])
    assert all(type(v) is float for v in scores.values())
    json.dumps({"accepted": scores["score"] < 10, "scores": scores}, allow_nan=False)


def test_campaign_recall_keeps_controllers_equally_weighted():
    from methods.gcrt.report import campaign_quality

    rows = []
    for sut, found in (("a", 1), ("a", 0), ("b", 1)):
        rows.append(
            dict(
                sut=sut,
                found=found,
                full_found=1,
                positives=1,
                valid_recall=True,
                recall=float(found),
                full_recall=1.0,
            )
        )
    result = campaign_quality(rows, rows, 3)
    assert result["macro_recall"] == 0.75
    assert result["pooled_recall"] == pytest.approx(2 / 3)
