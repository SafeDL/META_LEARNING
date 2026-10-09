import numpy as np
import pytest
import torch

from .model import joint_prior
from .session import AdaptiveTestingSession


def example():
    x = np.array([[0.1, 0.2, 0.3, 0.4, 0], [0.2, 0.3, 0.4, 0.5, 0],
                  [0.1, 0.2, 0.3, 0.4, 1]])
    risks = np.array([[0.2, 0.7, 0.4], [0.3, 0.9, 0.5], [0.4, 0.8, 0.6]])
    collisions = np.array([[0.1, 0.8, 0.2], [0.2, 0.95, 0.3], [0.1, 0.9, 0.5]])
    options = {
        "transform": "raw",
        "source_scale": 1,
        "noise": 0.01,
        "risk_scale": 0.2,
        "collision_scale": 0.3,
        "length": 0.1,
        "correlation": 0.8,
        "lookahead": False
    }
    return x, risks, collisions, options


def test_risk_conditioning_matches_batch_gaussian_conditioning():
    x, risks, collisions, options = example()
    mr, mc, rr, cr, cc, _ = joint_prior(x, risks, collisions, options, "cpu")
    session = AdaptiveTestingSession(x,
                                     risks,
                                     collisions,
                                     options,
                                     device="cpu")
    observed = []
    values = []
    for index, value in [(0, 0.6), (1, 0.8)]:
        session.pending = index
        session.records.append({})
        session.observe(value)
        observed.append(index)
        values.append(value)
    kernel = rr[observed][:, observed] + options["noise"] * torch.eye(2)
    residual = torch.tensor(values, dtype=torch.float64) - mr[observed]
    update = torch.linalg.solve(kernel, residual)
    assert torch.allclose(session.mean_r, mr + rr[:, observed] @ update)
    assert torch.allclose(session.mean_c, mc + cr[:, observed] @ update)
    expected_variance = cc - (cr[:, observed] * torch.linalg.solve(
        kernel, cr[:, observed].T).T).sum(1)
    assert torch.allclose(session.cc, expected_variance)
    assert torch.equal(session.mean_c[2:], mc[2:])


def test_last_query_is_direct_discovery():
    x, risks, collisions, options = example()
    options["lookahead"] = True
    session = AdaptiveTestingSession(x,
                                     risks,
                                     collisions,
                                     options,
                                     budget=1,
                                     device="cpu")
    assert session.next_index() == int(session.probabilities().argmax())


def test_risk_marginal_cannot_identify_collision_only_transfer_parameters():
    x, risks, collisions, options = example()
    first = {
        **options, "residual_mode": "learned_sensitivity",
        "positive_loading": True,
        "loading": [[0, 0, 0, 0, 0, 1]] * 2,
        "collision_offset": [-0.1, -0.2]
    }
    second = {
        **first, "collision_scale": 4,
        "loading": [[2, 1, 0, 1, 0, 0]] * 2,
        "collision_offset": [2, 3]
    }
    left = joint_prior(x, risks, collisions, first, "cpu")
    right = joint_prior(x, risks, collisions, second, "cpu")
    torch.testing.assert_close(left[0], right[0], atol=0, rtol=0)
    torch.testing.assert_close(left[2], right[2], atol=0, rtol=0)
    assert not torch.equal(left[1], right[1])
    assert not torch.equal(left[3], right[3])
    assert not torch.equal(left[4], right[4])


def test_endpoint_and_area_reward_identity():
    labels = np.array([1, 0, 1, 0, 0])
    budget = len(labels)
    score = 0.5 * labels.sum() + 0.5 * np.cumsum(labels).mean()
    weights = 0.5 + 0.5 * np.arange(budget, 0, -1) / budget
    assert score == pytest.approx(labels @ weights)


@pytest.mark.parametrize("residual_mode",
                         ["sensitivity", "learned_sensitivity"])
def test_shared_residual_and_decision_difference_covariance(residual_mode):
    x, risks, collisions, options = example()
    options.update({
        "residual_mode": residual_mode,
        "collision_transform": "probit",
        "ranking_information": True,
        "lookahead": True
    })
    if residual_mode == "learned_sensitivity":
        options.update({
            "loading": [[-1, 0.1, 0.2, 0.1, -0.3, 0.2]] * 2,
            "collision_offset": [0, 0],
            "positive_loading": True
        })
    mr, mc, rr, cr, cc, _ = joint_prior(x,
                                        risks,
                                        collisions,
                                        options,
                                        "cpu",
                                        full_collision=True)
    joint = torch.cat((torch.cat((rr, cr.T), 1), torch.cat((cr, cc), 1)), 0)
    assert torch.linalg.eigvalsh(joint).min() > -1e-10
    session = AdaptiveTestingSession(x,
                                     risks,
                                     collisions,
                                     options,
                                     budget=2,
                                     device="cpu")
    query = session.next_index()
    before = session.cc_matrix.clone()
    cross = session.cr[:, query].clone()
    denominator = session.rr[query, query].clone() + options["noise"]
    session.observe(0.65)
    first, second = 0, 1
    initial_difference = before[first, first] + before[
        second, second] - 2 * before[first, second]
    final_difference = (session.cc_matrix[first, first] +
                        session.cc_matrix[second, second] -
                        2 * session.cc_matrix[first, second])
    reduction = (cross[first] - cross[second]).square() / denominator
    assert torch.allclose(initial_difference - final_difference, reduction)


def test_unobservable_collision_difference_has_no_diagnostic_value():
    x, risks, collisions, options = example()
    options.update({
        "correlation": 0,
        "collision_transform": "probit",
        "ranking_information": True
    })
    collisions[1] = collisions[0]
    session = AdaptiveTestingSession(x[:2],
                                     risks[:2],
                                     collisions[:2],
                                     options,
                                     budget=1,
                                     device="cpu")
    assert torch.equal(session.cr[0], session.cr[1])
    assert torch.equal(session.ranking_information(session.probabilities()),
                       torch.zeros(2, dtype=torch.float64))


def test_probit_readout_matches_numerical_integration():
    from scipy.integrate import quad
    from scipy.special import ndtr
    from scipy.stats import norm

    x, risks, collisions, options = example()
    options["collision_transform"] = "probit"
    session = AdaptiveTestingSession(x,
                                     risks,
                                     collisions,
                                     options,
                                     device="cpu")
    for mean, deviation in [(0.5, 2), (-1.5, 0.3), (3, 1.5)]:
        exact = quad(lambda z: ndtr(mean + deviation * z) * norm.pdf(z),
                     -10,
                     10,
                     epsabs=1e-12)[0]
        value = session.readout(torch.tensor(mean), torch.tensor(deviation**2))
        assert float(value) == pytest.approx(exact, abs=1e-7)


def test_empirical_predictive_mean_preserves_prior_probabilities():
    x, risks, collisions, options = example()
    options.update({
        "collision_transform": "probit",
        "residual_mode": "sensitivity",
        "predictive_mean": "empirical_probability"
    })
    session = AdaptiveTestingSession(x,
                                     risks,
                                     collisions,
                                     options,
                                     device="cpu")
    assert np.allclose(session.probabilities(), collisions.mean(1))


def test_exact_confirmation_test_handles_positive_effect_and_zero_effect():
    from .evaluate_confirmation import sign_flip_p

    differences = np.array([[1, 1], [1, -1], [1, 0], [1, 0]], dtype=float)
    assert np.allclose(sign_flip_p(differences), [2 / 16, 1])


def test_complete_risk_observation_retains_source_discrepancy_uncertainty():
    from scipy.special import ndtri

    x, risks, collisions, options = example()
    options.update({
        "residual_mode": "sensitivity",
        "collision_transform": "probit"
    })
    _, _, rr, cr, cc, ur = joint_prior(x,
                                       risks,
                                       collisions,
                                       options,
                                       "cpu",
                                       full_collision=True)
    values = ndtri(collisions)
    bc = (values - values.mean(1)[:, None]) / np.sqrt(values.shape[1] - 1)
    uc = torch.as_tensor(np.concatenate(
        (bc * (x[:, 4:5] == 0), bc * (x[:, 4:5] == 1)), axis=1).T,
                         dtype=torch.float64) * np.sqrt(
                             options["source_scale"])
    local_r = rr - ur.T @ ur
    loading = (cr - uc.T @ ur).diag() / local_r.diag()
    posterior_u = torch.eye(
        len(ur), dtype=rr.dtype) - ur @ torch.linalg.solve(rr, ur.T)
    source_difference = uc.T - loading[:, None] * ur.T
    independent_c = local_r * options["collision_scale"]**2 / options[
        "risk_scale"]**2
    reconstructed = independent_c + source_difference @ posterior_u @ source_difference.T
    conditioned = cc - cr @ torch.linalg.solve(rr, cr.T)
    assert torch.allclose(conditioned, reconstructed, atol=1e-12)
    assert torch.linalg.eigvalsh(conditioned - independent_c).min() > -1e-12
    assert float((conditioned - independent_c).diag().sum()) > 1e-6


@pytest.mark.parametrize("method", ["candidate", "ras_frt_uq"])
def test_sealed_selector_requests_only_unique_budgeted_observations(method):
    import multiprocessing as mp
    from methods.history_guided_testing.config import ROOT as BASELINE
    from methods.history_guided_testing.io import read_json
    from research.history_response_testing.history_model import predict
    from research.history_response_testing.scenarios import ras_predictions
    from .config import BUDGET, OUTPUT
    from .confirm import selector_worker

    torch.set_num_threads(1)
    bank = np.load(BASELINE / "target/responses.npz")
    options = read_json(OUTPUT /
                        "models/monotone_transfer/logit_600.json")["options"]
    x = bank["x"]
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=selector_worker,
                                              args=(child, ))
    process.start()
    child.close()
    disclosed = []
    try:
        parent.send(
            ("task", (method, 11, x, predict(x, 11), ras_predictions(x, 11),
                      np.zeros(len(x)), options)))
        while True:
            message, value = parent.recv()
            if message == "query":
                index = int(value)
                assert index not in disclosed and 0 <= index < len(x) and len(
                    disclosed) < BUDGET
                disclosed.append(index)
                parent.send((None, bool(bank["collision"][index])) if method ==
                            "ras_frt_uq" else (float(bank["risk"][index]),
                                               None))
            else:
                assert message == "result", value
                assert value["selected_indices"] == disclosed and len(
                    disclosed) == BUDGET
                break
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
        process.join()
    assert process.exitcode == 0
