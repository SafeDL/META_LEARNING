"""Check Laplace stationarity separately from reproducing ten Newton steps."""

import argparse
import copy
import hashlib
import time

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import expit
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.classification import logistic_moments
from methods.common.gp_classifier import ClassificationKernel, fit_target
from methods.gcrt.residual_scale import cached_covariance

from .discovery_config import OUTPUT, read_json, write_json
from .discovery_counterfactual_adaptation import REPORT as INPUT, protocol as input_protocol
from .discovery_physical_mixture import checkpoint_path
from .discovery_residual_scale import cache_path
from .discovery_response_locality import history_inputs
from .discovery_scaml_audit import mode_numpy
from .protocol import ROOT


REPORT = INPUT/'mode_audit'
TOLERANCE = 1e-6


def protocol():
    return {'purpose': 'stationarity audit of existing historical virtual predictions; no new search',
        'input_protocol': input_protocol(), 'whitened_gradient_tolerance': TOLERANCE,
        'legacy_newton_iterations': 10, 'jitter': 1e-5,
        'cached_scope': 'all 16 prefixes; fixed-target branches whose candidate is among the cached 160 points; all physical branches',
        'full_scope': 'all 16 prefixes and all 64 virtual branches, fixed and refitted target/physical models',
        'full_device': 'cuda', 'inference_device': 'cpu',
        'full_stage_prerequisite': 'complete 80-run RGPE audit and report to avoid broadcast-sampler GPU contention',
        'input_evidence': {f: hashlib.sha256((INPUT/f).read_bytes()).hexdigest()
                           for f in ('protocol.json', 'report.json', 'summary_audit.json')},
        'source_files': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in (
            'benchmark/discovery_counterfactual_mode_audit.py', 'benchmark/discovery_scaml_audit.py',
            'methods/common/gp_classifier.py')},
        'new_selector_queries': 0, 'new_physical_calls': 0}


def prepare():
    original, reviewed = read_json(INPUT/'report.json'), read_json(INPUT/'summary_audit.json')
    if (original['protocol'] != input_protocol() or reviewed['status'] != 'passed'
            or reviewed['scoring_rows_checked'] != 192 or reviewed['historical_tasks_checked'] != 16):
        raise ValueError('Complete the historical prediction matrix and saved-array review first')
    specification = protocol()
    path = REPORT/'protocol.json'
    if path.exists() and read_json(path) != specification:
        raise ValueError('The mode audit is already registered differently')
    write_json(path, specification)
    return specification


def legacy_prediction(mean, covariance, observed, labels):
    """Independent ten-step reconstruction plus a whitened mode residual."""
    observed = np.asarray(observed, dtype=int)
    y = np.asarray(labels, dtype=float)
    center = np.asarray(mean)[observed]
    gram = covariance[np.ix_(observed, observed)]+1e-5*np.eye(len(observed))
    prior_factor = cholesky(gram, lower=True)
    latent = center.copy()
    for _ in range(10):
        probability = expit(latent)
        weight = np.maximum(probability*(1-probability), 1e-9)
        curvature = np.sqrt(weight)
        factor = cholesky(np.eye(len(y))+curvature[:, None]*gram*curvature[None], lower=True)
        b = weight*(latent-center)+y-probability
        alpha = b-curvature*cho_solve((factor, True), curvature*(gram @ b))
        latent = center+gram @ alpha
    probability = expit(latent)
    curvature = np.sqrt(np.maximum(probability*(1-probability), 1e-9))
    factor = cholesky(np.eye(len(y))+curvature[:, None]*gram*curvature[None], lower=True)
    alpha = cho_solve((prior_factor, True), latent-center)
    whitened = solve_triangular(prior_factor, latent-center, lower=True)
    gradient = whitened+prior_factor.T @ (probability-y)
    maximum = float(np.max(np.abs(gradient)))

    def predict(a, root, posterior_factor):
        cross = covariance[:, observed]
        predicted_mean = mean+cross @ a
        solved = solve_triangular(posterior_factor, root[:, None]*cross.T, lower=True)
        variance = np.maximum(np.diag(covariance)-np.square(solved).sum(0), 1e-10)
        return logistic_moments(predicted_mean, variance)[0]

    old = predict(alpha, curvature, factor)
    if maximum > TOLERANCE:
        _, converged_alpha, root, stable_factor, converged_residual = mode_numpy(center, covariance[np.ix_(observed, observed)], y)
        stable = predict(converged_alpha, root, stable_factor)
    else:
        stable, converged_residual = old, None
    return old, stable, {'whitened_gradient_max': maximum,
        'converged_within_audit_tolerance': maximum <= TOLERANCE,
        'stable_reference_gradient_max': converged_residual,
        'stable_probability_max_difference': float(np.max(np.abs(stable-old)))}


def priors(target_mean, covariance, source_mean, source_covariance, reference, beta):
    count = len(covariance)
    means = np.concatenate((source_mean+target_mean-reference, np.full((1, count), target_mean)))
    covariances = np.concatenate((source_covariance+beta**2*covariance[None], covariance[None]))
    return means, covariances


def component_check(mean, covariance, observed, labels, names, metadata):
    old, stable, rows = [], [], []
    for m, c, name in zip(mean, covariance, names):
        p, corrected, row = legacy_prediction(m, c, observed, labels)
        old.append(p)
        stable.append(corrected)
        rows.append({**metadata, 'component': name, **row})
    return np.asarray(old), np.asarray(stable), rows


def after_weights(saved, x, index, label):
    probability = saved['prequery_components'][:, index]
    likelihood = np.log(np.maximum(probability if label else 1-probability, 1e-15))
    influence = np.exp(-np.square(x-x[index]).mean(-1)/(2*.3**2))
    score = saved['prequery_log_weights']+likelihood[:, None]*influence
    result = np.exp(score-score.max(0))
    return result/result.sum(0)


def inputs(kind, archive):
    with np.load(cache_path(kind, archive.source_id), allow_pickle=False) as cache:
        ordered = cache['ordered_indices'].copy()
        z = cache['encoded'][64].copy()
        parameters = cache['parameters'][64].copy()
        reference = float(cache['parameters'][0, 0])
        raw = cache['raw_probabilities'][64].copy()
    expected = np.random.default_rng(1301).permutation(512)[:160]
    np.testing.assert_array_equal(ordered, expected)
    path = INPUT/'histories'/kind/f'{archive.source_id}.npz'
    with np.load(path, allow_pickle=False) as stored:
        saved = {k: stored[k].copy() for k in stored.files}
    np.testing.assert_array_equal(saved['paid'], ordered[:64])
    return saved, ordered, z, parameters, reference, raw


def raw_predictions(model, x, observed, labels):
    with torch.no_grad():
        covariance = model.covariance(x).cpu().numpy()
        mean = np.full(len(x), float(model.mean.detach()))
    return legacy_prediction(mean, covariance, observed, labels)


def run(scope):
    specification = read_json(REPORT/'protocol.json')
    if specification != protocol() or (REPORT/f'{scope}.json').exists():
        raise ValueError('Run the unchanged and not already completed mode audit')
    if scope == 'full':
        audit, report = read_json(OUTPUT/'rgpe/audit/report.json'), read_json(OUTPUT/'rgpe/report.json')
        if (audit['status'] != 'passed' or audit['runs'] != 80
                or report['status'] != 'complete_audited_comparison'):
            raise ValueError('Defer the CUDA refits until the complete RGPE audit and report')
    started = time.perf_counter()
    torch.set_num_threads(1)
    rows, covered, unmatched = [], [], []
    prediction_checks = 0
    with threadpool_limits(1):
        for kind in ('cut_in', 'cut_out'):
            history, all_means, all_covariances = history_inputs(kind)
            beta = specification['input_protocol']['selected_beta'][kind]
            for archive in history:
                saved, ordered, encoded, parameters, reference, raw_cached = inputs(kind, archive)
                paid = ordered[:64].tolist()
                labels = archive.collision[paid].tolist()
                positions = [i for i, a in enumerate(history) if a.source_id != archive.source_id]
                names = [history[i].source_id for i in positions]+['target_fsbo']
                meta = {'scenario': kind, 'pseudo_target': archive.source_id, 'lineage': archive.lineage}
                raw_base = ClassificationKernel(archive.x.shape[1])
                raw_current = copy.deepcopy(raw_base)
                fit_target(raw_current, archive.x[paid], labels, steps=10, learning_rate=.01)
                old_raw, _, raw_mode = raw_predictions(raw_current, archive.x, paid, labels)
                np.testing.assert_allclose(old_raw[ordered], raw_cached, atol=2e-9, rtol=2e-9)
                rows.append({**meta, 'stage': 'prefix', 'component': 'physical', **raw_mode})
                if scope == 'full':
                    base, _ = load_fsbo(checkpoint_path(kind, 'physical_fsbo'))
                    base = base.to('cuda')
                    current = copy.deepcopy(base)
                    fit_target(current, archive.x[paid], labels, steps=10, learning_rate=.003)
                    with torch.no_grad():
                        np.testing.assert_allclose(current.features(archive.x)[ordered].cpu(), encoded, atol=2e-7, rtol=2e-6)
                        kernel = current.covariance(archive.x).cpu().numpy()
                        center = float(current.mean.detach())
                    sm, sc = all_means[positions], all_covariances[positions]
                    predict_positions, observed = np.arange(512), paid
                else:
                    kernel = cached_covariance(torch.tensor(encoded[:64]), torch.tensor(parameters[1]), torch.tensor(parameters[2])).numpy()
                    center = float(parameters[0])
                    sm = all_means[positions][:, paid]
                    sc = all_covariances[positions][:, paid, :][:, :, paid]
                    predict_positions, observed = np.asarray(paid), list(range(64))
                means, covariance = priors(center, kernel, sm, sc, reference, beta)
                before, _, checked = component_check(means, covariance, observed, labels, names, {**meta, 'stage': 'prefix'})
                rows += checked
                np.testing.assert_allclose(before, saved['prequery_components'][:, predict_positions], atol=2e-7, rtol=2e-6)
                for case, index in enumerate(saved['candidate_indices'].tolist()):
                    available = scope == 'full' or index in ordered
                    (covered if available else unmatched).append({**meta, 'candidate_index': index})
                    for label in (False, True):
                        extended, feedback = [*paid, index], [*labels, label]
                        prefix = {**meta, 'stage': 'virtual', 'candidate_index': index, 'virtual_label': label}
                        refit_raw = copy.deepcopy(raw_base)
                        fit_target(refit_raw, archive.x[extended], feedback, steps=10, learning_rate=.01)
                        raw_fixed, _, mode = raw_predictions(raw_current, archive.x, extended, feedback)
                        rows.append({**prefix, 'component': 'physical_fixed', **mode})
                        raw_refit, _, mode = raw_predictions(refit_raw, archive.x, extended, feedback)
                        rows.append({**prefix, 'component': 'physical_refit', **mode})
                        expected_delta = 10*(saved['predictions'][case, int(label), 2]-saved['predictions'][case, int(label), 1])
                        np.testing.assert_allclose(raw_refit-raw_fixed, expected_delta, atol=2e-8, rtol=2e-7)
                        if not available:
                            continue
                        if scope == 'full':
                            predict_positions, observed = np.arange(512), extended
                            options = [('fixed', current)]
                            adapted = copy.deepcopy(base)
                            fit_target(adapted, archive.x[extended], feedback, steps=10, learning_rate=.003)
                            options.append(('refit_target', adapted))
                            sm, sc = all_means[positions], all_covariances[positions]
                        else:
                            cached_positions = [int(np.flatnonzero(ordered == i)[0]) for i in extended]
                            z = torch.tensor(encoded[cached_positions])
                            local_kernel = cached_covariance(z, torch.tensor(parameters[1]), torch.tensor(parameters[2])).numpy()
                            predict_positions, observed = np.asarray(extended), list(range(65))
                            sm = all_means[positions][:, extended]
                            sc = all_covariances[positions][:, extended, :][:, :, extended]
                            options = [('fixed', None)]
                        weight = after_weights(saved, archive.x, index, label)[:, predict_positions]
                        for model_name, model in options:
                            if model is not None:
                                with torch.no_grad():
                                    local_kernel = model.covariance(archive.x).cpu().numpy()
                                    local_center = float(model.mean.detach())
                            else:
                                local_center = center
                            means, covariance = priors(local_center, local_kernel, sm, sc, reference, beta)
                            probabilities, _, checked = component_check(means, covariance, observed, feedback, names,
                                {**prefix, 'target_parameters': model_name})
                            rows += checked
                            mixture = .9*(weight*probabilities).sum(0)+.1*raw_fixed[predict_positions]
                            mi = 0 if model_name == 'fixed' else 1
                            np.testing.assert_allclose(mixture, saved['predictions'][case, int(label), mi, predict_positions], atol=2e-7, rtol=2e-6)
                            prediction_checks += 1
                            if model_name == 'refit_target':
                                mixture = .9*(weight*probabilities).sum(0)+.1*raw_refit[predict_positions]
                                np.testing.assert_allclose(mixture, saved['predictions'][case, int(label), 2, predict_positions], atol=2e-7, rtol=2e-6)
                                prediction_checks += 1
                print('counterfactual_mode', scope, kind, archive.source_id, flush=True)
    if scope == 'full' and (len(rows) != 1296 or prediction_checks != 192 or len(covered) != 32):
        raise ValueError('The full mode audit must cover every source, virtual branch and probability array')
    failures = [r for r in rows if not r['converged_within_audit_tolerance']]
    result = {'status': 'complete_modes_checked', 'scope': scope, 'protocol': specification,
        'mode_checks': len(rows), 'nonstationary_modes': len(failures),
        'maximum_whitened_gradient': max(r['whitened_gradient_max'] for r in rows),
        'maximum_stable_probability_difference': max(r['stable_probability_max_difference'] for r in rows),
        'covered_candidates': covered, 'candidates_requiring_full_cuda_refit': unmatched,
        'saved_prediction_comparisons': prediction_checks, 'rows': rows,
        'whole_virtual_matrix_audited': scope == 'full', 'wall_s': time.perf_counter()-started,
        'new_selector_queries': 0, 'new_physical_calls': 0, 'development_admission_claimed': False}
    write_json(REPORT/f'{scope}.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'cached', 'full'))
    stage = parser.parse_args().stage
    prepare() if stage == 'prepare' else run(stage)
