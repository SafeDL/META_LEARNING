"""Audit paid feedback, causal model weights and actual discovery protection."""
import numpy as np
from scipy.special import log_ndtr, logsumexp

from .adaptive_margin_prior import event_log_likelihood, mixed_log_likelihood
from .config import BUDGET, INITIAL_QUERIES
from .decision_support import SUPPORT_THRESHOLD, bivariate_normal_probability
from .evaluation import metrics


def audit_run(run, method, bank, baseline):
    order = run["selected_indices"]
    if len(order) != BUDGET or len(set(order)) != BUDGET:
        raise ValueError("Invalid mixed-feedback paid budget")
    actual = metrics(bank["collision"], order)
    if any(run[k] != actual[k] for k in ("area_200", "recall_200", "curve", "F200")):
        raise ValueError("Margin metrics differ from measured events")
    expected = [{"index": i, "risk": float(bank["risk"][i]), "collision": bool(bank["collision"][i])} for i in order]
    if run["disclosures"] != expected or len(run["margin_records"]) != BUDGET or run["margin_observations"] != BUDGET:
        raise ValueError("Margin feedback differs from actual paid disclosures")
    for observation, record in zip(expected, run["margin_records"]):
        if any(record[k] != observation[k] for k in observation):
            raise ValueError("Margin inference uses unrequested feedback")
        continuous = not record["collision"]
        likelihood = "positive_value" if continuous else "negative_censor" if record["collision"] else "positive_censor"
        proxy = 1 - record["risk"] if continuous else None
        if record["likelihood"] != likelihood or record["observed_proxy"] != proxy:
            raise ValueError("Mixed likelihood does not match the declared observation type")
        mean, variance = record["predictive_mean_before"], record["predictive_variance_before"]
        if variance <= 0:
            raise ValueError("Invalid margin predictive variance")
        if continuous:
            innovation, strength = (proxy - mean) / np.sqrt(variance), 1.
        else:
            sign = -1 if record["collision"] else 1
            z = sign * mean / np.sqrt(variance)
            ratio = np.exp(-.5 * z ** 2 - .5 * np.log(2 * np.pi) - log_ndtr(z))
            innovation, strength = sign * ratio, np.clip(ratio * (ratio + z), 0, 1)
        if not np.allclose([innovation, strength], [record["standardized_mean_innovation"],
                            record["conditional_variance_factor"]], rtol=1e-8, atol=1e-8):
            raise ValueError("Margin moment update differs from its observation likelihood")
    if method == "frozen_reference" and order != baseline["selected_indices"]:
        raise ValueError("Frozen reference no longer matches retained baseline")
    if order[:10] != baseline["selected_indices"][:10]:
        raise ValueError("Margin controls have different initialization")
    prefix = run["reference_prefix_indices"]
    if prefix != baseline["selected_indices"][:len(prefix)] or len(run["coupling_checks"]) != BUDGET:
        raise ValueError("Margin coupling changed the independent reference")
    gains = np.asarray(run["curve"]) - baseline["curve"]
    bounds = []
    for t, check in enumerate(run["coupling_checks"], 1):
        k = check["reference_prefix_length"]
        coverage = min(t, INITIAL_QUERIES)
        scheduled = t > 10 and (t - 11) % 2 == 0
        if not coverage <= k <= t or check["scheduled_reference"] != scheduled:
            raise ValueError("Margin policy violates reference coverage")
        extra = [i for i in order[:t] if i not in prefix[:k]]
        negative = [i for i in extra if not bank["collision"][i]]
        if (extra != check["extra_indices"] or negative != check["negative_extra_indices"]
                or check["negative_extra_count"] != len(negative) or check["discovery_gain_lower"] != -len(negative)):
            raise ValueError("Margin reference ledger differs from the paid prefix")
        if gains[t - 1] < -len(negative):
            raise ValueError("Margin discovery curve violates the reference count bound")
        if method != "frozen_reference":
            found = run["curve"][t - 1]
            if len(negative) > 1 + found / np.sqrt(BUDGET) + 1e-12:
                raise ValueError("Margin policy exceeds discovery credit")
            if found < (baseline["curve"][t - 1] - 1) / (1 + 1 / np.sqrt(BUDGET)) - 1e-12:
                raise ValueError("Margin policy violates the full relative discovery bound")
        bounds.append(-len(negative))
    if sum(bounds) != run["area_gain_lower"] or bounds[-1] != run["terminal_gain_lower"] or gains.sum() < sum(bounds):
        raise ValueError("Margin area or terminal accounting is inconsistent")
    for check in run["regime_selection_checks"]:
        i, j = check["chosen_index"], check["reference_index"]
        if order[check["query"] - 1] != i or bank["x"][i, 4] != bank["x"][j, 4]:
            raise ValueError("Margin candidate changed the declared functional regime")
        if i != j and check["chosen_probability"] <= check["reference_probability"] + 1e-12:
            raise ValueError("Margin correction lacks its declared probability advantage")


def audit_adaptive(run, method, bank, baseline, *, evidence_score="event", event_report=False):
    audit_run(run, method, bank, baseline)
    if len(run["component_margin_records"]) != 2 or len(run["prior_evidence_records"]) != BUDGET:
        raise ValueError("Missing adaptive prior component or evidence records")
    for index, component in enumerate(run["component_margin_records"]):
        if index != 1 or not event_report:
            audit_run({**run, "margin_records": component}, method, bank, baseline)
        else:
            if len(component) != BUDGET or len(run["component_inference_updates"][1]) != BUDGET:
                raise ValueError("Missing report constraints or feedback records")
            for t, (record, actual) in enumerate(zip(component, run["disclosures"]), 1):
                if any(record[k] != actual[k] for k in ("index", "risk", "collision")):
                    raise ValueError("Report component used unrequested target feedback")
                if (record["observed_proxy"] is not None
                        or not np.isfinite(record["predictive_mean_before"]) or record["predictive_variance_before"] <= 0):
                    raise ValueError("Report component invented a numeric safe value")
                update = run["component_inference_updates"][1][t - 1]
                if (update["observations"] != t or update["constraints"] != t
                        or update["relative_site_change"] >= 1e-7):
                    raise ValueError("Report inference differs from actual paid-prefix constraints")
    logweights = np.full(2, -np.log(2))
    component_loss, mixture_loss = np.zeros(2), 0.
    for t, evidence in enumerate(run["prior_evidence_records"], 1):
        records = [c[t - 1] for c in run["component_margin_records"]]
        if any(evidence[k] != records[0][k] for k in ("index", "risk", "collision")):
            raise ValueError("Adaptive prior evidence differs from real observations")
        means = np.asarray([r["predictive_mean_before"] for r in records])
        variances = np.asarray([r["predictive_variance_before"] for r in records])
        likelihood = (event_log_likelihood(means, variances, evidence["collision"]) if evidence_score == "event"
                      else mixed_log_likelihood(means, variances, evidence["risk"], evidence["collision"]))
        weights = np.exp(logweights)
        used = weights
        if (not np.allclose(evidence["weights_before"], weights, rtol=0, atol=1e-10)
                or not np.allclose(evidence["selection_weights_before"], used, rtol=0, atol=1e-10)
                or not np.allclose(evidence["log_predictive_likelihoods"], likelihood, rtol=0, atol=1e-8)):
            raise ValueError("Adaptive prior uses post-query or inconsistent evidence")
        if evidence_score == "event":
            component_loss -= likelihood
            mixture_loss -= logsumexp(logweights + likelihood)
            if (abs(mixture_loss + logsumexp(-component_loss) - np.log(2)) > 1e-8
                    or mixture_loss > component_loss.min() + np.log(2) + 1e-8):
                raise ValueError("Before-query event scoring violated its log-loss mixture identity")
        logweights += likelihood
        logweights -= logsumexp(logweights)
        after = np.exp(logweights)
        if not np.allclose(evidence["weights_after"], after, rtol=0, atol=1e-10):
            raise ValueError("Adaptive prior weights differ from mixed-likelihood update")
    for check in run["regime_selection_checks"]:
        used = run["prior_evidence_records"][check["query"] - 1]["selection_weights_before"]
        if not np.allclose(check["selection_weights"], used, atol=1e-10, rtol=0):
            raise ValueError("Prior selection weights changed before the recorded action")
        predictions = np.asarray([used @ np.asarray(check["component_chosen_probabilities"]),
                                  used @ np.asarray(check["component_reference_probabilities"])])
        if not np.allclose(predictions, [check["chosen_probability"], check["reference_probability"]], atol=1e-10, rtol=0):
            raise ValueError("Action probabilities do not match adaptive prior weights")


def audit_support(run):
    checks = run["discovery_support_checks"]
    if len(checks) != BUDGET:
        raise ValueError("Missing per-query discovery support records")
    for query, check in enumerate(checks, 1):
        if run["selected_indices"][query - 1] != check["chosen_index"]:
            raise ValueError("Supported action differs from actual paid query")
        support = check["binary_support"]
        if support is not None:
            weights = run["prior_evidence_records"][query - 1]["selection_weights_before"]
            if not np.allclose(weights, support["model_weights"], rtol=0, atol=1e-10):
                raise ValueError("Binary support uses future model evidence")
            loss, gain = [], []
            for component in support["components"]:
                first, second = component["report_standardized_means"]
                correlation = component["report_correlation"]
                loss.append(bivariate_normal_probability(first, -second, -correlation))
                gain.append(bivariate_normal_probability(-first, second, -correlation))
            loss, gain = float(np.asarray(weights) @ loss), float(np.asarray(weights) @ gain)
            if not np.allclose([loss, gain], [support["loss_probability"], support["gain_probability"]], atol=1e-9, rtol=0):
                raise ValueError("Binary support differs from the stored joint report posterior")
            eligible = loss <= 1 - SUPPORT_THRESHOLD and gain - loss > 1e-12
            if (support["supported"] != eligible or abs(support["loss_allowance"] - (1 - SUPPORT_THRESHOLD)) > 1e-12):
                raise ValueError("Binary discovery support differs from predeclared budget rule")
        if check["reference_observation_skipped"]:
            if not check["probe_credit_available"] or support is None or not support["supported"]:
                raise ValueError("Reference observation skipped without both model support and actual loss credit")
        elif check["scheduled_reference"] and check["chosen_index"] != check["reference_index"]:
            raise ValueError("Unresolved scheduled reference observation was not executed")
    if run["reference_observations_skipped"] != sum(c["reference_observation_skipped"] for c in checks):
        raise ValueError("Reference skip count is inconsistent")
