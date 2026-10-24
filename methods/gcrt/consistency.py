"""Matched source-kernel/reliability meta-training on historical systems only."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from benchmark.discovery_config import read_json, write_json
from benchmark.protocol import ROOT
from methods.common.classification import torch_condition, torch_moments
from methods.common.response import HistoricalPredictor, ResponseEncoder, archive_fingerprints, source_weights

from .config import CONFIG
from .train import episode


MODES = {'shared_global': (False, None), 'source_global': (True, None),
         'source_local': (True, 0.30)}


def training_contract():
    paths = ('methods/gcrt/consistency.py', 'methods/gcrt/config.py', 'methods/gcrt/train.py',
             'methods/gcrt/discovery.py', 'methods/gcrt/posterior.py',
             'methods/common/response.py', 'methods/common/classification.py')
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def tensor_prior(encoder, x, contexts, mode, *, target_weight=0.25):
    independent, _ = MODES[mode]
    physical = encoder.physical_kernel(x).double()
    if independent:
        encoded = encoder.encode(x, contexts).double()
        covariance = physical[None] + encoder.log_response_scale.exp().double().square() * (
            encoded @ encoded.transpose(-1, -2))
    else:
        covariance = encoder.covariance(x, contexts)[None].expand(len(contexts), -1, -1)
    covariance = torch.cat((covariance, physical[None]))
    covariance = covariance + 1e-6 * torch.eye(len(x), dtype=torch.float64, device=physical.device)
    means = np.concatenate((HistoricalPredictor(contexts).latent_means(np.asarray(x)),
                            np.zeros((1, len(x)))))
    weights = np.r_[source_weights(contexts) * (1 - target_weight), target_weight]
    return (torch.as_tensor(means, dtype=torch.float64, device=physical.device), covariance,
            torch.as_tensor(weights, dtype=torch.float64, device=physical.device))


def query_probabilities(encoder, x, contexts, labels, support_count, mode, *, target_weight=0.25):
    mean, covariance, weights = tensor_prior(encoder, x, contexts, mode, target_weight=target_weight)
    log_weights = weights.log()[:, None].expand(-1, len(x))
    _, bandwidth = MODES[mode]
    if bandwidth is None:
        influence = torch.ones((support_count, len(x)), dtype=mean.dtype, device=mean.device)
    else:
        values = torch.as_tensor(x, dtype=mean.dtype, device=mean.device)
        distance = ((values[:support_count, None] - values[None]) ** 2).mean(-1)
        influence = (-distance / (2 * bandwidth ** 2)).exp()
    for i in range(support_count):
        mean, covariance, likelihood = torch_condition(mean, covariance, i, bool(labels[i]))
        log_weights = log_weights + likelihood[:, None] * influence[i]
    probability = torch_moments(mean[:, support_count:],
                                covariance.diagonal(dim1=1, dim2=2)[:, support_count:])[0]
    return (log_weights[:, support_count:].softmax(dim=0) * probability).sum(0)


def episode_loss(encoder, contexts, target, indices, support_count, mode, config=CONFIG):
    x, labels = target.x[indices], target.collision[indices]
    probability = query_probabilities(encoder, x, contexts, labels, support_count, mode,
                                      target_weight=config.target_weight)
    truth = torch.as_tensor(labels[support_count:], dtype=probability.dtype, device=probability.device)
    nll = F.binary_cross_entropy(probability.clamp(1e-8, 1 - 1e-8), truth)
    auxiliary_truth = encoder.tensor(target.auxiliary[indices[support_count:]])
    auxiliary = F.mse_loss(encoder.auxiliary_prediction(x[support_count:], contexts), auxiliary_truth)
    return nll + config.auxiliary_weight * auxiliary, {
        'collision_nll': float(nll.detach()), 'auxiliary_mse': float(auxiliary.detach()),
        'brier': float(((probability-truth)**2).mean().detach())}


def validate(encoder, history, development, mode, config=CONFIG):
    encoder.eval()
    rows = []
    with torch.no_grad():
        for target in development:
            contexts, indices = episode(history, target, np.random.default_rng(config.seed + 1),
                                         max(config.support_counts), config.query_count)
            _, metrics = episode_loss(encoder, contexts, target, indices,
                                       max(config.support_counts), mode, config)
            rows.append({'family': target.family, **metrics})
    encoder.train()
    families = sorted({r['family'] for r in rows})
    return {key: float(np.mean([np.mean([r[key] for r in rows if r['family'] == family])
                               for family in families]))
            for key in ('collision_nll', 'auxiliary_mse', 'brier')}


def train(history, development, destination, mode, config=CONFIG):
    if (not history or not development or any(a.role != 'history' for a in history)
            or any(a.role != 'development' for a in development)
            or len({a.source_id for a in (*history, *development)}) != len(history)+len(development)):
        raise ValueError('Consistency training requires disjoint historical and development roles')
    kinds = {a.scenarios[0].kind for a in (*history, *development)}
    if len(kinds) != 1 or mode not in MODES:
        raise ValueError('Train exactly one registered scenario and inference mode')
    destination = Path(destination)
    specification = {'mode': mode, 'kind': next(iter(kinds)), 'config': asdict(config),
                     'training_contract': training_contract(),
                     'history_fingerprints': archive_fingerprints(history),
                     'development_fingerprints': archive_fingerprints(development)}
    # JSON round trip makes immutable tuple-valued configuration comparable after reload.
    specification = json.loads(json.dumps(specification))
    report_path = destination.with_suffix('.json')
    if destination.exists() or report_path.exists():
        report = read_json(report_path)
        if report['status'] != 'complete' or report['specification'] != specification:
            raise ValueError('Existing consistency training is incomplete or belongs to another contract')
        load_encoder(destination)
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    generator = np.random.default_rng(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = ResponseEncoder(history[0].x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=config.learning_rate, weight_decay=0)
    best, curve = (float('inf'), float('inf')), []
    for step in range(1, config.steps + 1):
        target = history[(step-1) % len(history)]
        support = int(generator.choice(config.support_counts))
        contexts, indices = episode(history, target, generator, support, config.query_count)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = episode_loss(encoder, contexts, target, indices, support, mode, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite consistency loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
        optimizer.step()
        if step % config.validation_interval == 0 or step == config.steps:
            validation = validate(encoder, history, development, mode, config)
            score = validation['collision_nll'], validation['brier']
            curve.append({'step': step, 'training': metrics, 'validation': validation})
            if score < best:
                best = score
                temporary = destination.with_suffix('.tmp')
                torch.save({'format': 'collision-source-consistency', 'specification': specification,
                            'state_dict': encoder.state_dict(), 'dimension': encoder.dimension,
                            'step': step, 'parameter_count': sum(p.numel() for p in encoder.parameters()),
                            'history_ids': [a.source_id for a in history]}, temporary)
                temporary.replace(destination)
            write_json(report_path, {'status': 'complete' if step == config.steps else 'running',
                       'specification': specification, 'curve': curve,
                       'best_score': list(best), 'wall_s': time.perf_counter()-started})
            print(mode, next(iter(kinds)), step, '/', config.steps,
                  'validation_nll', round(score[0], 6), flush=True)
    return read_json(report_path)


def load_encoder(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('format') != 'collision-source-consistency'
            or payload['specification']['training_contract'] != training_contract()):
        raise ValueError('Consistency model format or training source contract differs')
    encoder = ResponseEncoder(payload['dimension'])
    encoder.load_state_dict(payload['state_dict'])
    return encoder.eval().requires_grad_(False), payload
