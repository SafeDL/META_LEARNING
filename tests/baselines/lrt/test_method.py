"""Local joint semantics, honest continuation labels and protected queries."""

import copy
import multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import stdtr
from scipy.stats import t as student_t
import torch

from methods.baselines.dart.selection import select as previous_select
from methods.baselines.lrt.fantasy import (
    acquisition_values,
    conditional_probabilities,
)
from methods.baselines.lrt.posterior import (
    LocalPosterior,
    model_set,
)
from methods.common.prior import (
    build_prior,
    physical_kernel,
    archive_covariance,
)
from methods.baselines.lrt.selection import choose, select
from methods.baselines.lrt.teacher import (
    branch_return,
    replay_reference,
)
from methods.baselines.lrt.value import (
    ContinuationValue,
    make_features,
    value_loss,
)
from methods.baselines.lrt.evaluation.evaluate import (
    selector_worker,
)
from benchmark.risk_channel import disclose_risk

torch.set_num_threads(1)


def problem(n=24):
    rng = np.random.default_rng(91)
    x = np.column_stack((rng.random((n, 4)), np.arange(n) % 2))
    history = {
        name: {"x": x[:16].copy(), "risk": 0.2 + 0.6 * x[:16, offset]}
        for offset, name in enumerate(("idm_reactive", "idm_delayed"))
    }
    covariance = physical_kernel(x) * 0.3
    prior = build_prior(x, history, covariance)
    common = dict(
        x=x,
        covariance=covariance,
        priors=prior.means,
        probability=prior.weights,
        fixed_prior=prior.means.mean(1),
        seed=11,
    )
    truth = 0.1 + 0.8 * x[:, 0]
    return common, prior, truth


class Oracle:
    def __init__(self, risk):
        self.risk = risk
        self.indices = []

    def query(self, index):
        assert index not in self.indices
        self.indices.append(index)
        return SimpleNamespace(risk=float(self.risk[index]))


def test_archive_kernel_projectivity_psd_and_scene_isolation():
    common, prior, _ = problem()
    x = common["x"]
    actual = archive_covariance(x[:8], x[:16])
    np.testing.assert_allclose(actual, prior.archive_covariances[0][:8, :8], atol=1e-14)
    for covariance in (*prior.archive_covariances, prior.physical_covariance):
        assert np.linalg.eigvalsh(covariance).min() >= -1e-12
        assert np.count_nonzero(covariance[x[:, 4, None] != x[None, :, 4]]) == 0
    sparse = archive_covariance(x, x[:4])
    difference = sparse - prior.archive_covariances[0]
    assert np.linalg.eigvalsh(difference).min() >= -1e-12
    assert np.trace(difference) > 0.1
    model = LocalPosterior(prior, kappa=2.0)
    for rows, block in model.blocks.values():
        expected = prior.enhanced_covariance + 2 * prior.archive_covariances[0]
        np.testing.assert_allclose(
            block.groups[0][1].covariance, expected[np.ix_(rows, rows)]
        )


def test_other_family_is_exactly_unchanged_and_global_control_is_not():
    _, prior, _ = problem()
    for local in (True, False):
        model = LocalPosterior(prior, local=local)
        before = model.probabilities().copy()
        weights = copy.deepcopy(model.weights())
        model.observe(0, 0.95)
        if local:
            np.testing.assert_array_equal(before[1::2], model.probabilities()[1::2])
            assert weights["1"] == model.weights()["1"]
        else:
            assert not np.allclose(before[1::2], model.probabilities()[1::2])


@pytest.mark.parametrize("local", [False, True])
def test_virtual_condition_matches_actual_update_without_mutation(local):
    _, prior, _ = problem()
    prior.archive_covariances = (
        prior.archive_covariances[0],
        1.7 * prior.archive_covariances[1],
    )
    model = LocalPosterior(prior, local=local)
    model.observe(0, 0.6)
    indices = np.arange(2, len(prior.x))
    before = model.probabilities().copy()
    predicted = conditional_probabilities(model, 1, [0.8], indices)[0]
    updated = copy.deepcopy(model)
    updated.observe(1, 0.8)
    np.testing.assert_allclose(predicted, updated.probabilities()[indices], atol=1e-13)
    np.testing.assert_array_equal(model.probabilities(), before)
    assert np.isfinite(conditional_probabilities(model, 1, [-3, 4], indices)).all()
    with pytest.raises(ValueError):
        model.observe(1, -3)


def test_target_component_does_not_use_historical_means_or_covariance():
    _, prior, _ = problem()
    changed = copy.deepcopy(prior)
    changed.means[:] = 100
    changed.enhanced_covariance *= 20
    first, second = LocalPosterior(prior), LocalPosterior(changed)
    for model in (first, second):
        model.observe(0, 0.8)
    for family in first.blocks:
        a = first.blocks[family][1].groups[-1][1]
        b = second.blocks[family][1].groups[-1][1]
        np.testing.assert_array_equal(a.source_mean, b.source_mean)
        np.testing.assert_array_equal(a.variance, b.variance)
        np.testing.assert_array_equal(a.beta, b.beta)


def test_joint_loss_matches_draws_and_cross_family_factorization():
    _, prior, _ = problem()
    model = LocalPosterior(prior)
    model.observe(0, 0.6)
    indices = np.array([2, 3, 4])
    losses = model.losses(indices, 4)
    q = model.probabilities()
    assert losses[1] == (1 - q[3]) * q[4]
    assert losses[2] == 0
    samples = model.draws(indices, 100000, np.random.default_rng(814))
    empirical = ((samples[:, 0] <= 0.5) & (samples[:, 2] > 0.5)).mean()
    assert abs(empirical - losses[0]) < 0.007


def test_same_family_replacement_matches_adaptive_student_integration():
    _, prior, _ = problem()
    model = LocalPosterior(prior)
    model.observe(0, 0.6)
    a, b = 2, 4
    block = model.blocks[0][1]
    exact = 0.0
    for columns, component in block.groups:
        indices = model.positions[[a, b]]
        covariance = component.covariance_block(indices)
        for offset, column in enumerate(columns):
            mean = component.source_mean[indices, offset]
            scale = covariance * component.beta[offset] / component.alpha
            df = 2 * component.alpha

            def integrand(r):
                standardized = (r - mean[1]) / np.sqrt(scale[1, 1])
                conditional_mean = mean[0] + scale[0, 1] / scale[1, 1] * (r - mean[1])
                conditional_scale = np.sqrt(
                    (scale[0, 0] - scale[0, 1] ** 2 / scale[1, 1])
                    * (df + standardized**2)
                    / (df + 1)
                )
                return (
                    student_t.pdf(standardized, df)
                    / np.sqrt(scale[1, 1])
                    * student_t.cdf(
                        (0.5 - conditional_mean) / conditional_scale, df + 1
                    )
                )

            exact += (
                np.exp(block.log_weight[column])
                * quad(integrand, 0.5, np.inf, epsabs=1e-10)[0]
            )
    assert abs(model.losses(np.array([a]), b)[0] - exact) < 5e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_integer_student_cdf_matches_scipy_across_online_degrees():
    import cupy as cp
    from methods.common.student import student_cdf

    grid = np.unique(
        np.r_[
            np.linspace(-25, 25, 2001),
            np.logspace(-12, 12, 300),
            -np.logspace(-12, 12, 300),
            -np.inf,
            np.inf,
            np.nan,
        ]
    )
    values = cp.asarray(grid)
    for df in (*range(1, 206), 5.5):
        np.testing.assert_allclose(
            cp.asnumpy(student_cdf(df, values)),
            stdtr(df, grid),
            atol=2e-14,
            rtol=0,
            equal_nan=True,
        )


def test_gpu_continuation_matches_cpu_and_last_step():
    _, prior, _ = problem()
    model = LocalPosterior(prior)
    model.observe(0, 0.6)
    remaining = np.ones(len(prior.x), bool)
    remaining[0] = False
    candidates = np.array([1, 2, 3])
    cpu = acquisition_values(model, candidates, remaining, 8, 10, gpu=False)
    if torch.cuda.is_available():
        gpu = acquisition_values(model, candidates, remaining, 8, 10, gpu=True)
        np.testing.assert_allclose(cpu, gpu, atol=1e-11)
    np.testing.assert_array_equal(
        acquisition_values(model, candidates, remaining, 1, 10),
        model.probabilities()[candidates],
    )


def test_branch_labels_preserve_prefix_and_return_correct_paid_suffix():
    common, _, truth = problem()
    reference, remaining = replay_reference(common, [], 10, 2)
    q = []
    for step in range(3):
        index, _ = reference.recommend(remaining, step)
        reference.observe(index, truth[index], truth[index] > 0.5)
        remaining[index] = False
        q.append({"index": int(index), "risk": float(truth[index])})
    candidate = int(np.flatnonzero(remaining)[2])
    count, selected = branch_return(reference, remaining, candidate, truth)
    assert len(set(selected)) == 7
    assert selected[0] == candidate
    assert count == np.count_nonzero(truth[selected] > 0.5)
    assert reference.model.indices == [r["index"] for r in q]
    reconstructed, _ = replay_reference(common, q, 10, 2)
    assert branch_return(reconstructed, remaining, candidate, truth) == (
        count,
        selected,
    )


def test_value_source_permutation_and_finite_gradients():
    common, prior, _ = problem()
    models = model_set(prior)
    remaining = np.ones(len(prior.x), bool)
    candidates = np.array([0, 1, 2])
    losses = np.stack([m.losses(candidates, 0) for m in models])
    features = make_features(
        models, remaining, candidates, models[1].probabilities(), 0.5, 8, 0, losses, 11
    )
    network = ContinuationValue()
    a = network(features)
    permuted = {**features, "source": features["source"][:, :, ::-1].copy()}
    torch.testing.assert_close(a, network(permuted))
    order = np.arange(len(features["x"]))[::-1].copy()
    inverse = np.argsort(order)
    shuffled_pool = {
        **features,
        "source": features["source"][:, order],
        "x": features["x"][order],
        "q": features["q"][:, order],
        "reference": features["reference"][order],
        "candidates": inverse[features["candidates"]],
        "top": inverse[features["top"]],
        "frontier": inverse[features["frontier"]],
    }
    torch.testing.assert_close(a, network(shuffled_pool))
    loss = value_loss(a, [3, 4, 6], 8)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in network.parameters()
    )
    assert sum(float(p.grad.abs().sum()) for p in network.parameters()) > 0


def test_disabled_path_reproduces_previous_and_zero_budget_uses_reference():
    common, prior, truth = problem()
    a = previous_select(
        **common,
        oracle=Oracle(truth),
        proposal_covariance=prior.enhanced_covariance,
        budget=12,
        initial_queries=3,
    )
    b = select(
        **common,
        oracle=Oracle(truth),
        transfer_prior=prior,
        value_model=None,
        budget=12,
        initial_queries=3,
        enabled=False,
    )
    assert a["queries"] == b["queries"]
    oracle = Oracle(truth)
    result = select(
        **common,
        oracle=oracle,
        transfer_prior=prior,
        value_model=ContinuationValue().eval(),
        budget=12,
        initial_queries=3,
        loss_fraction=0,
    )
    assert result["selected_indices"] == oracle.indices
    assert len(set(oracle.indices)) == 12
    assert all(d["selected_index"] == d["reference_index"] for d in result["decisions"])


def test_all_deviations_charge_maximum_model_loss():
    common, prior, truth = problem()
    result = select(
        **common,
        oracle=Oracle(truth),
        transfer_prior=prior,
        value_model=None,
        variant="no_value_learning",
        budget=12,
        initial_queries=3,
    )
    for row in result["decisions"]:
        if row["selected_index"] != row["reference_index"]:
            position = row["candidate_indices"].index(row["selected_index"])
            assert row["loss_charged"] == max(
                m[position] for m in row["candidate_losses"]
            )
    assert sum(d["loss_charged"] for d in result["decisions"]) <= 1.2 + 1e-12


def test_finite_horizon_advantage_telescopes():
    # Fully enumerable deterministic problem; pi0 takes the lowest remaining index.
    reward = np.array([0, 1, 0, 1])

    def baseline_value(remaining, h):
        return reward[list(remaining)[:h]].sum()

    remaining, horizon = [0, 1, 2, 3], 3
    baseline = baseline_value(remaining, horizon)
    actual, advantage = 0, 0
    for chosen in (3, 1, 0):
        before = baseline_value(remaining, horizon)
        future = [x for x in remaining if x != chosen]
        advantage += reward[chosen] + baseline_value(future, horizon - 1) - before
        actual += reward[chosen]
        remaining, horizon = future, horizon - 1
    assert actual - baseline == advantage


def test_continuation_identity_with_enumerated_correlated_feedback():
    worlds = np.array([[0, 0, 1, 1], [0, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 1]])
    probability = np.array([0.15, 0.25, 0.35, 0.25])

    def belief(history):
        mask = np.ones(len(worlds), bool)
        for index, value in history.items():
            mask &= worlds[:, index] == value
        weight = probability * mask
        return weight / weight.sum()

    def reference_action(history):
        order = (0, 2, 1, 3) if history.get(0, 0) else (0, 1, 3, 2)
        return next(a for a in order if a not in history)

    def reference_value(history, horizon):
        if horizon == 0:
            return 0.0
        return continuation(history, reference_action(history), horizon)

    def continuation(history, action, horizon):
        weight = belief(history)
        value = 0.0
        for observed in (0, 1):
            chance = weight[worlds[:, action] == observed].sum()
            if chance:
                value += chance * (
                    observed
                    + reference_value({**history, action: observed}, horizon - 1)
                )
        return value

    actual_return, expected_advantage = 0.0, 0.0
    for world, chance in zip(worlds, probability):
        history = {}
        for horizon in (3, 2, 1):
            order = (1, 3, 0, 2) if history.get(1, 0) else (1, 0, 2, 3)
            action = next(a for a in order if a not in history)
            expected_advantage += chance * (
                continuation(history, action, horizon)
                - reference_value(history, horizon)
            )
            actual_return += chance * world[action]
            history[action] = world[action]
    assert actual_return - reference_value({}, 3) == pytest.approx(expected_advantage)


def test_global_source_weights_reverse_unobserved_family_ranking():
    _, prior, _ = problem()
    prior.means[:] = 0.5
    prior.means[0] = [0.95, 0.05]
    prior.means[1] = [0.85, 0.15]
    prior.means[3] = [0.15, 0.85]
    prior.weights[:] = [0.4, 0.6]
    prior.enhanced_covariance[:] = np.eye(len(prior.x)) * 0.0025
    local = LocalPosterior(prior, archive_uncertainty=False, target_component=False)
    global_model = LocalPosterior(
        prior, local=False, archive_uncertainty=False, target_component=False
    )
    before = local.probabilities()
    assert before[1] < before[3]
    local.observe(0, 0.95)
    global_model.observe(0, 0.95)
    np.testing.assert_array_equal(local.probabilities()[[1, 3]], before[[1, 3]])
    assert global_model.probabilities()[1] > global_model.probabilities()[3]


def test_final_choice_uses_exact_marginals_without_value_network():
    common, prior, _ = problem()
    anchor, remaining = replay_reference(common, [], 1, 0)
    reference, _ = anchor.recommend(remaining, 0)
    models = model_set(prior)
    _, _, decision = choose(models, remaining, anchor, reference, 1, 20, 0, 11, None)
    candidates = np.asarray(decision["candidate_indices"])
    np.testing.assert_array_equal(
        decision["candidate_values"],
        np.stack([m.probabilities()[candidates] for m in models]),
    )


def test_isolated_selector_makes_exactly_200_unique_risk_only_queries():
    common, prior, truth = problem(210)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=selector_worker, args=(child,))
    process.start()
    child.close()
    task = dict(
        **common,
        policy="full",
        transfer_prior=prior,
        value_model=ContinuationValue().eval(),
        loss_fraction=0,
    )
    # Neither the response array nor collision labels are sent to the child.
    assert "risk" not in task and "collision" not in task
    try:
        trace = disclose_risk(parent, task, truth, process)
        assert len(set(trace["selected_indices"])) == 200
        assert len(trace["disclosures"]) == 200
        assert all(d["collision"] is None for d in trace["disclosures"])
        assert all(
            d["selected_index"] == d["reference_index"] for d in trace["decisions"]
        )
        assert all(
            d["reason"] == "common_initialization" for d in trace["decisions"][:10]
        )
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0


def test_training_cache_rebuild_reuses_saved_trace_and_branch_labels(
    tmp_path, monkeypatch
):
    from methods.baselines.lrt import data, value
    from methods.baselines.hgt.io import write_json

    common, prior, truth = problem()
    trace = select(
        **common,
        transfer_prior=prior,
        value_model=None,
        oracle=Oracle(truth),
        variant="no_value_learning",
        budget=12,
        initial_queries=2,
    )
    path = tmp_path / "historical/11/train/idm_reactive_complete_on_policy_full.json"
    write_json(path, trace)
    label_path = path.with_name(path.stem + "_returns.json")
    expected = data.records_for_trace(
        common, prior, truth, trace, label_path, "full", (2, 5, 9), 12, 2
    )
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(data, "TRAIN_BUDGET", 12)
    monkeypatch.setattr(data, "TRAIN_INITIAL", 2)
    monkeypatch.setattr(data, "TRAIN_STATES", (2, 5, 9))
    monkeypatch.setattr(
        data, "historical_case", lambda *args, **kwargs: (common, prior, truth)
    )

    def unnecessary(*args, **kwargs):
        pytest.fail("Cache rebuild must reuse the trace and paid branch labels")

    monkeypatch.setattr(data, "branch_return", unnecessary)
    monkeypatch.setattr(value, "ContinuationValue", unnecessary)
    data.generate_on_policy((11, "full", "idm_reactive", "complete", None))
    cache = data.feature_path(
        11, "full", "train", "idm_reactive", "complete", "on_policy"
    )
    actual = torch.load(cache, map_location="cpu", weights_only=False)["records"]
    assert len(actual) == len(expected)
    for first, second in zip(actual, expected):
        for key in ("returns", "reference_position", "candidate_indices", "step"):
            assert first[key] == second[key]
        for key in first["features"]:
            np.testing.assert_allclose(
                first["features"][key], second["features"][key], atol=1e-11
            )


def test_completed_training_does_not_rebuild_feature_cache(tmp_path, monkeypatch):
    from methods.baselines.lrt import train
    from methods.baselines.hgt.io import write_json

    write_json(tmp_path / "models/training_full_11.json", {"status": "complete"})
    torch.save({"state_dict": {}}, tmp_path / "models/full_11.pt")
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(
        train, "generate", lambda *args: pytest.fail("Training already complete")
    )
    train.train_seed(11, "full")
    assert not (tmp_path / "feature_cache").exists()


def test_missing_value_weights_rebuild_even_with_completed_report(
    tmp_path, monkeypatch
):
    from methods.baselines.lrt import train
    from methods.baselines.hgt.io import write_json

    write_json(tmp_path / "models/training_full_11.json", {"status": "complete"})
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "load_history", lambda: {"idm_reactive": {}})

    class RebuildStarted(Exception):
        pass

    def generate(seed, modes, *, workspace=None):
        assert workspace is None
        assert seed == 11 and modes == ("full",)
        raise RebuildStarted

    monkeypatch.setattr(train, "generate", generate)
    with pytest.raises(RebuildStarted):
        train.train_seed(11, "full")


def test_label_generation_builds_only_requested_training_variant(tmp_path, monkeypatch):
    from methods.baselines.lrt import data

    common, prior, truth = problem()
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(
        data, "historical_case", lambda *args, **kwargs: (common, prior, truth)
    )
    monkeypatch.setattr(data, "task_settings", lambda split: (12, 2, (2, 5, 9)))
    data.generate_case((11, "idm_reactive", "complete", "train", ("full",), None))
    assert {p.name for p in (tmp_path / "feature_cache/11").iterdir()} == {"full"}
    for behavior in ("cg_brt", "previous"):
        path = data.feature_path(
            11, "full", "train", "idm_reactive", "complete", behavior
        )
        records = torch.load(path, map_location="cpu", weights_only=False)["records"]
        assert [r["step"] for r in records] == [2, 5, 9]
        assert all(
            0 <= value <= 12 - r["step"] for r in records for value in r["returns"]
        )
