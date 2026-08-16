from __future__ import annotations

import unittest
from dataclasses import replace
import torch

from pearl_learning.src.context_encoder import kl_diag_normal, product_of_gaussians_with_prior
from pearl_learning.src.io import read_config
from pearl_learning.src.scenario_encoder import build_task_descriptor
from pearl_learning.src.taskbook import build_taskbook


class ScenarioPriorTests(unittest.TestCase):
    def test_prior_only_product_is_exact_prior(self) -> None:
        prior_mu = torch.tensor([[0.3, -0.2]])
        prior_log_var = torch.tensor([[0.4, -0.7]])
        # A deliberately almost-uninformative evidence factor makes the
        # expected precision-sum calculation directly checkable.
        evidence_mu = torch.zeros((1, 1, 2))
        evidence_log_var = torch.full((1, 1, 2), 20.0)
        mu, log_var = product_of_gaussians_with_prior(evidence_mu, evidence_log_var, prior_mu, prior_log_var)
        self.assertTrue(torch.allclose(mu, prior_mu, atol=1e-6))
        self.assertTrue(torch.allclose(log_var, prior_log_var, atol=1e-6))

    def test_matching_normals_have_zero_kl(self) -> None:
        mu = torch.tensor([[0.3, -0.2]])
        log_var = torch.tensor([[0.4, -0.7]])
        self.assertAlmostEqual(float(kl_diag_normal(mu, log_var, mu, log_var)), 0.0, places=7)

    def test_prior_evidence_product_is_order_independent(self) -> None:
        prior_mu = torch.zeros((1, 2)); prior_log_var = torch.zeros((1, 2))
        mu = torch.tensor([[[0.0, 1.0], [2.0, -1.0]]])
        log_var = torch.zeros_like(mu)
        first = product_of_gaussians_with_prior(mu, log_var, prior_mu, prior_log_var)
        second = product_of_gaussians_with_prior(mu.flip(1), log_var.flip(1), prior_mu, prior_log_var)
        self.assertTrue(torch.allclose(first[0], second[0]))
        self.assertTrue(torch.allclose(first[1], second[1]))

    def test_static_descriptor_excludes_case_seed_and_hidden_rule(self) -> None:
        task = build_taskbook(read_config("pearl_learning/configs/merge_method_flow_pilot.yaml"))["meta_train"][0]
        changed = replace(
            task,
            case_seed=task.case_seed + 123,
            priority_spec={**task.priority_spec, "target_contact_entry_order": "adversary_first"},
        )
        self.assertTrue(torch.equal(torch.from_numpy(build_task_descriptor(task)), torch.from_numpy(build_task_descriptor(changed))))
