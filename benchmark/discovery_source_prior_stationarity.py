"""Audit the historical joint-source modes without changing their saved priors."""

import hashlib
import json
import time

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import expit
from threadpoolctl import threadpool_limits
import torch

from methods.common.response import archive_fingerprints
from methods.gcrt.joint_response import MARGIN_NOISE, logistic_laplace, margin_values
from methods.gcrt.joint_training import load_model

from .construction import load_construction
from .discovery_config import OUTPUT, write_json
from .discovery_joint_response import PRIORS, model_path


TOLERANCE = 1e-6
DESTINATION = OUTPUT / 'converged_response/source_prior_stationarity.json'


def check_source(model, archive, prediction_x, cached_mean, cached_covariance):
    """Rebuild the original CUDA source state and independently check its gradient."""
    started = time.perf_counter()
    count = len(archive.x)
    combined = np.concatenate((archive.x, prediction_x))
    size = len(combined)
    device = model.mean.device
    with torch.no_grad():
        mean = model.joint_mean(size)
        covariance = model.joint_covariance(combined)
        selected = torch.arange(count, device=device)
        margin_indices = size + selected
        eye = torch.eye(count, dtype=mean.dtype, device=device)

        # Match the frozen joint source construction: Gaussian margins first.
        cross = covariance[:, margin_indices]
        factor = torch.linalg.cholesky(
            covariance[margin_indices][:, margin_indices] + MARGIN_NOISE**2 * eye)
        values = torch.as_tensor(margin_values(archive.safety[:, 0]),
                                 dtype=mean.dtype, device=device)
        residual = values - mean[margin_indices]
        margin_alpha = torch.cholesky_solve(residual[:, None], factor)[:, 0]
        mean = mean + cross @ margin_alpha
        solved = torch.linalg.solve_triangular(factor, cross.T, upper=False)
        covariance = covariance - solved.T @ solved

        _, alpha, root, factor = logistic_laplace(
            mean[:count], covariance[:count, :count], archive.collision)
        gram = covariance[:count, :count] + 1e-5 * eye
        latent = mean[:count] + gram @ alpha

        # The mode uses the stabilized training Gram; prediction cross-covariances
        # exclude that numerical diagonal, as in the original implementation.
        predicted_mean = mean + covariance[:, selected] @ alpha
        solved = torch.linalg.solve_triangular(
            factor, root[:, None] * covariance[selected, :], upper=False)
        predicted_covariance = covariance - solved.T @ solved
        indices = np.r_[np.arange(count, size), np.arange(size + count, 2 * size)]
        actual_mean = predicted_mean.cpu().numpy()[indices]
        actual_covariance = predicted_covariance.cpu().numpy()[np.ix_(indices, indices)]
        actual_covariance = .5 * (actual_covariance + actual_covariance.T)
        np.testing.assert_allclose(actual_mean, cached_mean, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(actual_covariance, cached_covariance, atol=2e-7, rtol=2e-6)

        prior_factor = torch.linalg.cholesky(gram)
        whitened = torch.linalg.solve_triangular(
            prior_factor, (latent - mean[:count])[:, None], upper=False)[:, 0]
        labels = torch.as_tensor(archive.collision, dtype=mean.dtype, device=device)
        torch_gradient = whitened + prior_factor.T @ (latent.sigmoid() - labels)
        center_np, gram_np = mean[:count].cpu().numpy(), gram.cpu().numpy()
        latent_np = latent.cpu().numpy()

    # Independent SciPy factorization and NumPy gradient, using the same actual
    # CUDA kernel and old iterate rather than changing the trained representation.
    factor_np = cholesky(gram_np, lower=True)
    whitened_np = solve_triangular(factor_np, latent_np - center_np, lower=True)
    gradient_np = whitened_np + factor_np.T @ (expit(latent_np) - archive.collision)
    gradient_torch_np = torch_gradient.cpu().numpy()
    np.testing.assert_allclose(gradient_np, gradient_torch_np, atol=2e-7, rtol=2e-7)
    maximum = float(np.max(np.abs(gradient_np)))
    if not np.isfinite(maximum):
        raise FloatingPointError('Nonfinite historical source mode residual')
    return {
        'source_id': archive.source_id, 'role': archive.role,
        'training_count': count, 'prediction_candidates': len(prediction_x),
        'joint_cached_channels': 2,
        'whitened_gradient_max': maximum,
        'within_diagnostic_tolerance': maximum <= TOLERANCE,
        'independent_gradient_max_error': float(np.max(np.abs(gradient_np - gradient_torch_np))),
        'cached_mean_max_error': float(np.max(np.abs(actual_mean - cached_mean))),
        'cached_covariance_max_error': float(np.max(np.abs(actual_covariance - cached_covariance))),
        'wall_s': time.perf_counter() - started,
    }


def run():
    if DESTINATION.exists():
        raise ValueError('Preserve the completed source-prior stationarity audit')
    torch.set_num_threads(1)
    started = time.perf_counter()
    rows, inputs = [], {}
    with threadpool_limits(1):
        for kind in ('cut_in', 'cut_out'):
            history, _ = load_construction(kind)
            if len(history) != 8 or any(a.role != 'history' or len(a.x) != 512 for a in history):
                raise ValueError('Audit the eight original 512-point historical sources only')
            model, metadata = load_model(model_path(kind))
            checkpoint_digest = hashlib.sha256(model_path(kind).read_bytes()).hexdigest()
            if metadata['specification']['history_fingerprints'] != archive_fingerprints(history):
                raise ValueError('Historical teacher and source archive identities differ')
            model = model.to('cuda')
            path = PRIORS / f'{kind}.npz'
            original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as saved:
                specification = json.loads(str(saved['specification']))
                if (specification['model_sha256'] != checkpoint_digest
                        or specification['history_fingerprints'] != archive_fingerprints(history)
                        or set(saved['names'].tolist()) != {a.source_id for a in history}):
                    raise ValueError('Saved source priors differ from their original construction')
                for archive in history:
                    index = saved['names'].tolist().index(archive.source_id)
                    row = check_source(model, archive, saved['x'],
                                       saved['means'][index], saved['covariances'][index])
                    rows.append({'scenario': kind, **row})
                    print('source_prior_stationarity', kind, archive.source_id,
                          row['whitened_gradient_max'], flush=True)
            if hashlib.sha256(path.read_bytes()).hexdigest() != original_digest:
                raise AssertionError('Read-only source-prior audit changed its input')
            inputs[kind] = {'checkpoint_sha256': checkpoint_digest,
                            'source_prior_sha256': original_digest,
                            'history_fingerprints': archive_fingerprints(history)}
    result = {
        'status': 'passed' if all(r['within_diagnostic_tolerance'] for r in rows)
                  else 'complete_nonstationary_source_modes_found',
        'scope': 'all frozen historical source construction modes, not target conditional modes',
        'source_modes': len(rows), 'source_training_points_per_mode': 512,
        'legacy_iterations': 10, 'diagnostic_whitened_gradient_tolerance': TOLERANCE,
        'kernel_and_legacy_inference_device': 'cuda',
        'independent_gradient_device': 'cpu',
        'maximum_whitened_gradient': max(r['whitened_gradient_max'] for r in rows),
        'nonstationary_modes': sum(not r['within_diagnostic_tolerance'] for r in rows),
        'cached_joint_predictions_and_covariances_reproduced': True,
        'inputs_unchanged': True, 'inputs': inputs, 'records': rows,
        'new_training': False, 'new_selector_queries': 0, 'new_physical_calls': 0,
        'current_frozen_search_changed': False,
        'wall_s': time.perf_counter() - started,
        'cost_scope': 'diagnostic overhead sharing the machine with live selectors',
    }
    write_json(DESTINATION, result)
    return result


if __name__ == '__main__':
    run()
