"""Use paid mixed-observation evidence to select or average two event priors."""
import numpy as np
from scipy.special import log_ndtr, logsumexp

from .censored_margin_response import CensoredMarginResponse, neutral_margin_prior


def mixed_log_likelihood(means, variances, risk, collision):
    if collision:
        return log_ndtr(-means / np.sqrt(variances))
    return -.5 * (np.log(2 * np.pi * variances) + (1 - risk - means) ** 2 / variances)


def event_log_likelihood(means, variances, collision):
    sign = -1 if collision else 1
    return log_ndtr(sign * means / np.sqrt(variances))


class AdaptiveMarginPrior:
    def __init__(self, reference, family, prior, *, device="cuda", model_class=CensoredMarginResponse, evidence_score="mixed", alternative_model=None):
        self.reference = reference
        self.family = np.asarray(family, dtype=int)
        self.models = [model_class(reference, family, prior, continuous_safe=True, device=device),
                       alternative_model if alternative_model is not None else
                       model_class(reference, family, neutral_margin_prior(prior), continuous_safe=True, device=device)]
        self.margin_audit_anchor = 0 if alternative_model is not None else 1
        self.evidence_score = evidence_score
        self.logweights = np.full(2, -np.log(2))
        self.indices, self.evidence_records, self.selection_checks = [], [], []

    def selection_weights(self):
        return np.exp(self.logweights)

    def probabilities(self):
        return self.selection_weights() @ np.asarray([m.probabilities() for m in self.models])

    def observe(self, index, risk, collision):
        means = np.asarray([float(m.mean[index]) for m in self.models])
        variances = np.asarray([float(m.covariance[index, index] + m.noise[index]) for m in self.models])
        likelihoods = (event_log_likelihood(means, variances, collision) if self.evidence_score == "event"
                       else mixed_log_likelihood(means, variances, risk, collision))
        before = self.selection_weights()
        self.logweights += likelihoods
        self.logweights -= logsumexp(self.logweights)
        after = self.selection_weights()
        for model in self.models:
            model.observe(index, risk, collision)
        self.indices.append(int(index))
        self.evidence_records.append({"index": int(index), "risk": float(risk), "collision": bool(collision),
                                      "weights_before": before.tolist(), "weights_after": after.tolist(),
                                      "selection_weights_before": before.tolist(),
                                      "log_predictive_likelihoods": likelihoods.tolist()})

    def rank(self, remaining, *, tie_break):
        head = int(self.reference.head())
        indices = np.flatnonzero(remaining & (self.family == self.family[head]))
        weights = self.selection_weights()
        probability = weights @ np.asarray([m.probabilities() for m in self.models])
        scores = np.asarray([(m.mean / (m.covariance.diag() + m.noise).sqrt()).cpu().numpy() for m in self.models])
        score = weights @ scores
        order = np.lexsort((tie_break[indices], score[indices], -np.round(probability[indices], 12)))
        proposed = int(indices[order[0]])
        index = proposed if probability[proposed] > probability[head] + 1e-12 else head
        check = {"query": len(self.indices) + 1, "chosen_index": index, "reference_index": head,
                 "family": int(self.family[head]), "chosen_probability": float(probability[index]),
                 "reference_probability": float(probability[head]), "selection_weights": weights.tolist()}
        component_probabilities = np.asarray([m.probabilities() for m in self.models])
        check.update(component_chosen_probabilities=component_probabilities[:, index].tolist(),
                     component_reference_probabilities=component_probabilities[:, check["reference_index"]].tolist())
        self.selection_checks.append(check)
        return np.asarray([index])

    def diagnostics(self):
        anchor = self.margin_audit_anchor
        return {"margin_records": self.models[anchor].records, "margin_observations": len(self.indices),
                "component_margin_records": [m.records for m in self.models], "margin_audit_anchor": int(anchor),
                "component_inference_updates": [getattr(m, "ep_updates", []) for m in self.models],
                "prior_evidence_records": self.evidence_records, "regime_selection_checks": self.selection_checks,
                "evidence_score": self.evidence_score,
                "inference": "two working Gaussian posteriors; before-query scoring updates prior credibility"}
