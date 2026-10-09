"""Small invariant checks for the risk-conditioned sequential decoder."""
import numpy as np
import torch

from .risk_session import RiskConditionedTestingSession


def calibrators():
    return {
        str(family): {
            "risk_residual_mean": 0.0,
            "risk_residual_scale": 0.1,
            "intercept": -1.0,
            "collision_logit_scale": 1.0,
            "standardized_risk_residual_scale": 1.5,
            "collision_only": {
                "intercept": -1.0,
                "collision_logit_scale": 1.0,
            },
        }
        for family in (0, 1)
    }


def test_query_feedback_updates_risk_posterior_and_collision_decode():
    x = np.array([
        [0.1, 0.1, 0.1, 0.1, 0],
        [0.2, 0.2, 0.2, 0.2, 0],
        [0.3, 0.3, 0.3, 0.3, 1],
        [0.4, 0.4, 0.4, 0.4, 1],
    ])
    risk = np.array([
        [0.2, 0.2, 0.2, 0.2],
        [0.4, 0.4, 0.4, 0.4],
    ])
    logits = np.zeros_like(risk)
    discrepancy = {
        "gp_variance": [0.04, 0.04],
        "noise_variance": [0.002, 0.002],
        "length": [0.2, 0.2],
    }
    session = RiskConditionedTestingSession(
        x, risk, logits, discrepancy, calibrators(), budget=2, device="cpu")
    initial = session.probabilities().clone()
    query = session.next_index()
    session.observe(0.9)
    updated = session.probabilities()
    assert session.count == 1
    assert not session.remaining[query]
    assert not torch.allclose(initial, updated)
    assert torch.all((updated >= 0) & (updated <= 1))
    assert session.next_index() != query
    assert np.isfinite(float(updated.max()))
    ablation = RiskConditionedTestingSession(
        x, risk, logits, discrepancy, calibrators(), collision_only=True,
        budget=2, device="cpu")
    np.testing.assert_allclose(ablation.probabilities().numpy(), 1 / (1 + np.e))


if __name__ == "__main__":
    test_query_feedback_updates_risk_posterior_and_collision_decode()
    print("RISK CONDITIONED SESSION TEST PASSED")
