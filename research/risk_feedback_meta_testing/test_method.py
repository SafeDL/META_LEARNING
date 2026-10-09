"""Check deployment equivalence, stable probabilities and useful gradients."""
import unittest

import numpy as np
import torch

from research.behavior_response_testing.model import BehaviorResponseModel
from research.behavior_response_testing.session import BehaviorTestingSession

from .model import FailureDecoder, FrozenResponseBackbone
from .planner import BudgetLookaheadSession, continuation_gains
from .posterior import condition_risk
from .risk_state import CensoredRiskState
from .session import MetaTestingSession
from .train import collision_loss, meta_inputs


class MethodTests(unittest.TestCase):

    def setUp(self):
        torch.set_num_threads(1)
        generator = np.random.default_rng(17)
        self.x = np.column_stack((generator.random((8, 4)), np.arange(8) % 2))
        self.risk = generator.uniform(0.2, 0.8, (3, 8))
        self.discrepancy = {"gp_variance": [0.04, 0.02],
                            "noise_variance": [0.003, 0.005],
                            "length": [0.3, 0.2]}

    def test_batch_matches_sequential_conditioning(self):
        support, feedback = [1, 6, 5], [0.8, 0.3, 0.7]
        session = BehaviorTestingSession(
            self.x, self.risk, np.zeros_like(self.risk), self.discrepancy,
            device="cpu")
        for index, risk in zip(support, feedback):
            session.pending = index
            session.records.append({"index": index})
            session.observe(risk)
        weights, mean, variance = condition_risk(
            torch.as_tensor(self.x), torch.as_tensor(self.risk),
            support, feedback, self.discrepancy)
        torch.testing.assert_close(weights, session.log_weights, atol=1e-12, rtol=0)
        torch.testing.assert_close(mean, session.risk_mean - torch.as_tensor(self.risk),
                                   atol=1e-12, rtol=0)
        torch.testing.assert_close(variance, session.rr.diagonal()[None, :].expand_as(mean),
                                   atol=1e-12, rtol=0)

    def test_training_and_online_decode_match(self):
        torch.manual_seed(23)
        family = torch.as_tensor(self.x[:, 4]).long()
        features = torch.randn(3, 8, 4, dtype=torch.float32)
        decoder = FailureDecoder(torch.randn(2, 4, dtype=torch.float32),
                                 torch.zeros(2, dtype=torch.float32),
                                 [0.02, -0.01], [0.1, 0.2])
        with torch.no_grad():
            decoder.slope.copy_(torch.tensor([0.5, -0.4]))
        support, feedback = [1, 6], [1.0, 0.0]
        weights, mean, variance = condition_risk(
            torch.as_tensor(self.x), torch.as_tensor(self.risk),
            support, feedback, self.discrepancy)
        train_log = decoder.log_probabilities(features, family, weights, mean, variance)
        session = MetaTestingSession(
            self.x, self.risk, decoder.logits(features, family).detach(),
            self.discrepancy,
            {key: getattr(decoder, key).detach().tolist()
             for key in ("slope", "center", "scale")}, device="cpu")
        for index, risk in zip(support, feedback):
            session.pending = index
            session.records.append({"index": index})
            session.observe(risk)
        for train, online in zip(train_log, session.log_probabilities()):
            torch.testing.assert_close(train, online, atol=1e-12, rtol=0)
        torch.testing.assert_close(train_log[0].exp() + train_log[1].exp(),
                                   torch.ones(8, dtype=torch.float64))
        labels = torch.as_tensor(np.arange(8) % 2)
        loss = -(labels * train_log[0] + (1 - labels) * train_log[1]).mean()
        loss.backward()
        for parameter in decoder.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_ranking_retains_differences_near_probability_one(self):
        session = MetaTestingSession(
            self.x[:2], self.risk[:, :2], np.array([[50., 80.]] * 3),
            self.discrepancy, {"slope": [0, 0], "center": [0, 0],
                               "scale": [1, 1]}, budget=2, device="cpu")
        self.assertEqual(session.next_index(), 1)
        session.observe(0.7)
        self.assertEqual(session.next_index(), 0)
        low_probability = MetaTestingSession(
            self.x[:2], self.risk[:, :2], np.array([[-80., -50.]] * 3),
            self.discrepancy, {"slope": [0, 0], "center": [0, 0],
                               "scale": [1, 1]}, budget=2, device="cpu")
        probability = low_probability.probabilities()
        self.assertTrue(torch.all((probability > 0) & (probability <= 1)))
        self.assertGreater(float(probability[1]), float(probability[0]))
        self.assertEqual(low_probability.next_index(), 1)

    def test_backbone_tables_preserve_frozen_risk(self):
        torch.manual_seed(11)
        originals = [BehaviorResponseModel().eval() for _ in (0, 1)]
        backbone = FrozenResponseBackbone(
            {str(i): model.state_dict() for i, model in enumerate(originals)})
        x = torch.as_tensor(self.x, dtype=torch.float32)
        behavior = torch.rand(3, 4)
        risk, hidden = backbone.features(x, behavior)
        weight, bias = backbone.initial_head()
        decoder = FailureDecoder(weight, bias, [0, 0], [1, 1])
        table_risk, table_logits = backbone.tables(x, behavior, decoder)
        torch.testing.assert_close(table_risk, risk)
        torch.testing.assert_close(table_logits, decoder.logits(hidden, x[:, 4].long()))
        for family, original in enumerate(originals):
            indices = torch.nonzero(x[:, 4] == family, as_tuple=True)[0]
            for hypothesis in range(len(behavior)):
                inputs = torch.cat((x[indices, :4],
                                    behavior[hypothesis].expand(len(indices), -1)), 1)
                expected_risk, expected_logits = original(inputs)
                torch.testing.assert_close(risk[hypothesis, indices], expected_risk)
                torch.testing.assert_close(table_logits[hypothesis, indices],
                                           expected_logits)
        self.assertFalse(any(p.requires_grad for p in backbone.parameters()))

    def test_censored_update_matches_conditional_gaussian_samples(self):
        coordinates = torch.zeros(1, 5, dtype=torch.float64)
        predictions = torch.tensor([[0.9]], dtype=torch.float64)
        state = CensoredRiskState(coordinates, predictions, self.discrepancy, 1)
        state.observe(0, 1.0)
        rng = np.random.default_rng(271)
        latent = 0.9 + rng.normal(0, np.sqrt(0.04), 400000)
        response = latent + rng.normal(0, np.sqrt(0.003), len(latent))
        selected = latent[response >= 1]
        self.assertAlmostEqual(float(state.mean[0, 0]), selected.mean(), delta=0.002)
        self.assertAlmostEqual(float(state.variance[0, 0]), selected.var(), delta=0.002)
        self.assertGreater(float(state.mean[0, 0]), 1)

    def test_predictive_integration_preserves_first_two_moments(self):
        coordinates = torch.zeros(1, 5, dtype=torch.float64)
        predictions = torch.tensor([[0.3], [0.9]], dtype=torch.float64)
        state = CensoredRiskState(coordinates, predictions, self.discrepancy, 1)
        nodes, probabilities = state.predictive_nodes(0, 32)
        self.assertTrue(torch.all((nodes >= 0) & (nodes <= 1)))
        self.assertAlmostEqual(float(probabilities.sum()), 1, places=12)
        prior_mean = (state.log_weights.exp() * state.mean[:, 0]).sum()
        prior_second = (state.log_weights.exp() *
                        (state.variance[:, 0] + state.mean[:, 0].square())).sum()
        means, seconds = [], []
        for node in nodes:
            weights, mean, variance, _ = state.conditional_moments(0, float(node))
            means.append((weights.exp() * mean[:, 0]).sum())
            seconds.append((weights.exp() *
                            (variance[:, 0] + mean[:, 0].square())).sum())
        self.assertAlmostEqual(float(probabilities @ torch.stack(means)),
                               float(prior_mean), delta=0.0005)
        self.assertAlmostEqual(float(probabilities @ torch.stack(seconds)),
                               float(prior_second), delta=0.0005)
        self.assertEqual(state.count, 0)

    def test_batched_hypotheses_match_scalar_censored_updates(self):
        state = CensoredRiskState(torch.as_tensor(self.x),
                                  torch.as_tensor(self.risk), self.discrepancy, 3)
        state.observe(1, 1.0)
        before = state.mean.clone()
        nodes, _ = state.predictive_nodes(6, 8)
        batched = state.conditional_candidates(6, nodes)
        for position, node in enumerate(nodes):
            scalar = state.conditional_moments(6, float(node))
            for actual, expected in zip(batched, scalar[:3]):
                torch.testing.assert_close(actual[position], expected,
                                           atol=1e-12, rtol=0)
        torch.testing.assert_close(state.mean, before, atol=0, rtol=0)
        self.assertEqual(state.count, 1)

    def test_planner_does_not_disclose_fantasies_and_enforces_reference(self):
        generator = np.random.default_rng(29)
        session = BudgetLookaheadSession(
            self.x, self.risk, generator.normal(size=self.risk.shape),
            self.discrepancy,
            {"slope": [1.2, -0.4], "center": [0, 0], "scale": [0.1, 0.2]},
            budget=3, device="cpu")
        before = session.state.mean.clone()
        chosen = session.next_index()
        record = session.records[-1]
        self.assertGreaterEqual(record["predicted_terminal_count"],
                                record["reference_terminal_count"])
        self.assertGreaterEqual(record["predicted_cumulative_count"],
                                record["reference_cumulative_count"])
        self.assertEqual(session.count, 0)
        self.assertEqual(session.state.count, 0)
        self.assertEqual(len(session.records), 1)
        torch.testing.assert_close(session.state.mean, before, atol=0, rtol=0)
        session.observe(1.0)
        self.assertFalse(session.remaining[chosen])
        next_chosen = session.next_index()
        self.assertNotEqual(next_chosen, chosen)
        session.observe(0.4)
        final = session.next_index()
        self.assertNotIn(final, (chosen, next_chosen))
        session.observe(0.8)
        self.assertIsNone(session.next_index())
        self.assertEqual(session.state.count, 3)

    def test_information_value_is_zero_when_choices_and_order_do_not_change(self):
        current = torch.tensor([0.6, 0.5, 0.4, 0.3], dtype=torch.float64)
        future = (current + 0.1)[None, :]
        values = continuation_gains(current, future, torch.ones(4, dtype=torch.bool), 2)
        self.assertEqual(float(values[2][0]), 0)
        self.assertEqual(float(values[3][0]), 0)
        self.assertGreater(float(values[4][0]), float(values[0]))

    def test_choice_gain_equals_raw_lookahead_when_probabilities_are_consistent(self):
        current = torch.tensor([0.8, 0.7, 0.3], dtype=torch.float64)
        future = torch.tensor([[0.9, 0.6, 0.3], [0.7, 0.8, 0.3]], dtype=torch.float64)
        values = continuation_gains(current, future, torch.ones(3, dtype=torch.bool), 2)
        torch.testing.assert_close(values[0] + values[2].mean(), values[4].mean())
        torch.testing.assert_close(values[1] + values[3].mean(), values[5].mean())

    def test_meta_training_slices_query_variance_on_the_scene_axis(self):
        models = [BehaviorResponseModel().eval() for _ in (0, 1)]
        backbone = FrozenResponseBackbone(
            {str(i): model.state_dict() for i, model in enumerate(models)})
        support, query = np.array([1, 6]), np.array([0, 2, 3, 4, 5, 7])
        risk = np.full(8, 0.5)
        risk[support] = [1.0, 0.0]
        banks = {("historical", 0): {"x": self.x, "risk": risk}}
        task = ("historical", 0, support, query)
        inputs = meta_inputs(task, banks, backbone, torch.rand(3, 4), self.discrepancy)
        self.assertEqual(tuple(inputs[-1].shape), (3, 6))
        weight, bias = backbone.initial_head()
        decoder = FailureDecoder(weight, bias, [0, 0], [0.1, 0.1])
        loss = collision_loss(decoder, inputs, torch.zeros(6, dtype=torch.float64))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(decoder.slope.grad).all())


if __name__ == "__main__":
    unittest.main()
