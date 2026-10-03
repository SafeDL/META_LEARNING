import numpy as np
import torch

from methods.srd_tnp_bqd.data import SourceContext, risk_logit
from methods.srd_tnp_bqd.mean_readout import local_mean, tail_loss
from methods.srd_tnp_bqd.matched_history import source_spec
from methods.srd_tnp_bqd.s01 import build_spec


def test_local_mean_preserves_constant_response_and_permutation():
    rng = np.random.default_rng(11)
    x = rng.random((20, 4))
    context = SourceContext("source", x, risk_logit(np.full((20, 1), .8)), np.ones(20, bool))
    query = rng.random((5, 4))
    expected = np.full((5, 1), np.log(4.))
    np.testing.assert_allclose(local_mean(context, query), expected, atol=1e-12)
    shuffled = SourceContext("source", x[::-1].copy(), context.z[::-1].copy(), context.valid)
    np.testing.assert_allclose(local_mean(shuffled, query), local_mean(context, query), atol=1e-12)


def test_tail_loss_moves_dangerous_scene_above_safe_scene():
    target = torch.tensor([2., 1., -2., -3.])
    reversed_scores = (-target).clone().requires_grad_(True)
    loss = tail_loss(reversed_scores, target)
    assert tail_loss(target, target) < loss
    loss.backward()
    assert reversed_scores.grad[0] < 0 and reversed_scores.grad[-1] > 0


def test_supplemental_source_changes_speed_without_changing_control_law():
    source = source_spec()
    original = build_spec("idm_source_strong_brake")
    assert source.profile["controller"] == "IDM"
    assert source.profile["target_speed"] == 23.
    for field in original.profile:
        if field not in ("name", "target_speed"):
            assert source.profile[field] == original.profile[field]
