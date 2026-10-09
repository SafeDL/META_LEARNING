"""Scientific checks for learned likelihoods and risk-only conditioning."""
import math
import unittest

import numpy as np
import torch

from .config import KERNEL_NUGGET
from .model import ResponsePrior, failure_log_probabilities, risk_log_likelihood
from .posterior import (UnitFieldState, condition_field, correlation,
                        gaussian_condition, point_condition)


class InferenceTests(unittest.TestCase):

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(18)
        self.x = torch.rand(12, 5, dtype=torch.float64)
        self.x[:, 4] = torch.arange(12) % 2
        self.mean = 0.3 + 0.4 * torch.rand(4, 12, dtype=torch.float64)
        self.amplitude = 0.1 + 0.15 * torch.rand_like(self.mean)
        self.noise = 0.03 + 0.04 * torch.rand_like(self.mean)
        self.lengths = [0.4, 0.6]
        self.covariance = correlation(self.x, self.lengths)

    def test_heterogeneous_covariances_are_positive(self):
        covariance = (self.amplitude[:, :, None] * self.covariance
                      * self.amplitude[:, None, :]
                      + torch.diag_embed(self.noise.square()))
        torch.testing.assert_close(covariance, covariance.transpose(-2, -1))
        self.assertTrue(bool((torch.linalg.eigvalsh(covariance) > 0).all()))

    def test_batch_matches_sequential_interior(self):
        support, values = [1, 3, 5, 6], [0.4, 0.65, 0.55, 0.5]
        weights, mean, variance = gaussian_condition(
            self.covariance, self.mean, self.amplitude, self.noise, support, values)
        state = UnitFieldState(self.covariance, self.mean, self.amplitude, self.noise)
        for index, risk in zip(support, values):
            state.observe(index, risk)
        for a, b in ((weights, state.log_weights), (mean, state.mean),
                     (variance, state.variance)):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)

    def test_evidence_includes_variance_normalizer(self):
        support, values = [0, 2], [0.5, 0.7]
        weights = gaussian_condition(self.covariance, self.mean, self.amplitude,
                                     self.noise, support, values)[0]
        expected = []
        for index in range(4):
            amplitude = self.amplitude[index, support]
            covariance = (self.covariance[support][:, support]
                          * amplitude[:, None] * amplitude[None, :]
                          + torch.diag(self.noise[index, support].square()))
            law = torch.distributions.MultivariateNormal(self.mean[index, support],
                                                        covariance)
            expected.append(law.log_prob(torch.tensor(values, dtype=torch.float64)))
        expected = torch.stack(expected)
        expected -= torch.logsumexp(expected, 0)
        torch.testing.assert_close(weights, expected)

    def test_censored_support_projection_matches_full_online_state(self):
        support, values = [1, 3, 6], [1.0, 0.55, 0.0]
        weights, mean, variance = condition_field(
            self.x, self.mean, self.amplitude, self.noise, support, values, self.lengths)
        state = UnitFieldState(self.covariance, self.mean, self.amplitude, self.noise)
        for index, risk in zip(support, values):
            state.observe(index, risk)
        for a, b in ((weights, state.log_weights), (mean, state.mean),
                     (variance, state.variance)):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)

    def test_gradient_through_interior_and_censored_updates(self):
        for values in ([0.45, 0.55], [1.0, 0.55]):
            mean = self.mean[:2].clone().requires_grad_()
            amplitude = self.amplitude[:2].clone().requires_grad_()
            noise = self.noise[:2].clone().requires_grad_()

            def objective(mu, amp, nu):
                weights, posterior_mean, posterior_variance = condition_field(
                    self.x, mu, amp, nu, [1, 4], values, self.lengths)
                return (weights.exp()[:, None] *
                        (posterior_mean[:, 7:].square() + posterior_variance[:, 7:])).sum()

            self.assertTrue(torch.autograd.gradcheck(objective, (mean, amplitude, noise),
                                                    eps=1e-6, atol=1e-4, rtol=1e-3))

    def test_query_likelihood_fields_do_not_condition_the_posterior(self):
        support, values = [1, 4], [1.0, 0.55]
        baseline = condition_field(self.x, self.mean, self.amplitude, self.noise,
                                   support, values, self.lengths)
        mean, amplitude, noise = self.mean.clone(), self.amplitude.clone(), self.noise.clone()
        query = [i for i in range(12) if i not in support]
        mean[:, query] += 100
        amplitude[:, query] *= 10
        noise[:, query] *= 20
        changed = condition_field(self.x, mean, amplitude, noise, support, values,
                                  self.lengths)
        for a, b in zip(baseline, changed):
            torch.testing.assert_close(a, b)

    def test_clipped_likelihood_normalizes(self):
        means = torch.tensor([-0.2, 0.4, 1.2], dtype=torch.float64)
        variance = torch.tensor([0.04, 0.09, 0.04], dtype=torch.float64)
        points = torch.linspace(0.000001, 0.999999, 10000, dtype=torch.float64)
        density = risk_log_likelihood(means[:, None], variance[:, None], points).exp()
        atoms = risk_log_likelihood(means, variance, means.new_zeros(3)).exp()
        atoms += risk_log_likelihood(means, variance, means.new_ones(3)).exp()
        mass = torch.trapezoid(density, points, dim=1) + atoms
        torch.testing.assert_close(mass, torch.ones_like(mass), atol=1e-5, rtol=0)

    def test_probit_gaussian_mean_matches_monte_carlo(self):
        logits = self.mean[:, :3]
        latent_mean, latent_variance = logits - 0.5, self.amplitude[:, :3].square()
        family = torch.tensor([0, 1, 0])
        slope = torch.tensor([1.3, -0.8], dtype=torch.float64)
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        collision, safe = failure_log_probabilities(
            logits, family, weights.log(), latent_mean, latent_variance, slope)
        torch.testing.assert_close(collision.exp() + safe.exp(), torch.ones(3).double())
        rng = np.random.default_rng(24)
        samples = torch.from_numpy(rng.standard_normal((4, 3, 300000)))
        field = latent_mean[..., None] + latent_variance.sqrt()[..., None] * samples
        conditional = torch.special.ndtr(logits[..., None] + slope[family][None, :, None] * field)
        estimate = (weights[:, None] * conditional.mean(-1)).sum(0)
        torch.testing.assert_close(collision.exp(), estimate, atol=0.001, rtol=0)

    def test_point_condition_matches_full_filter(self):
        values = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
        covariance = torch.eye(3, dtype=torch.float64) * (1 + KERNEL_NUGGET)
        state = UnitFieldState(covariance, self.mean[:, :3],
                               self.amplitude[:, :3], self.noise[:, :3])
        expected = point_condition(self.mean[:, :3], self.amplitude[:, :3],
                                   self.noise[:, :3], values, 1 + KERNEL_NUGGET)
        for index, risk in enumerate(values):
            state.observe(index, float(risk))
        torch.testing.assert_close(state.mean, expected[0])
        torch.testing.assert_close(state.variance, expected[1])

    def test_neural_risk_correction_is_unbounded_and_differentiable(self):
        weight, bias = torch.rand(2, 128), torch.rand(2)
        model = ResponsePrior(weight, bias, torch.zeros(2, 128), torch.ones(2, 128),
                              {"gp_variance": [0.01, 0.02], "noise_variance": [0.001, 0.002]})
        for head in model.risk_corrections:
            with torch.no_grad():
                head[-1].bias.fill_(1.2)
        features = torch.rand(4, 12, 128)
        family = self.x[:, 4].long()
        frozen = torch.full((4, 12), 0.8)
        mean, amplitude, noise = model.risk_fields(features, family, frozen)
        self.assertTrue(bool((mean > 1).all()))
        loss = -risk_log_likelihood(mean, amplitude.square() + noise.square(),
                                   mean.new_ones(12)).mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.risk_corrections.parameters()))


if __name__ == "__main__":
    unittest.main()
