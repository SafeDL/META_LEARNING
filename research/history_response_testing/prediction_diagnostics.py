"""Matched-support prediction diagnostics, separate from discovery evaluation."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from methods.history_guided_testing.config import ROOT as BASELINE
from methods.history_guided_testing.gp import RiskGP
from methods.history_guided_testing.history import historical_risk, load_history, split_indices, subset
from methods.history_guided_testing.io import write_json
from methods.history_guided_testing.train import load_kernel

from .calibration import calibrators
from .config import OUTPUT, SEEDS, SOURCE_WEIGHTS
from .history_model import predict
from .kernel import covariance
from .session import RiskTestingSession


def main():
    torch.set_num_threads(1)
    history = load_history()
    (train, _) = split_indices(next(iter(history.values()))['x'])
    bank = np.load(BASELINE / 'target/responses.npz')
    weights = np.asarray(SOURCE_WEIGHTS)
    knn_prior = historical_risk(subset(history, train), bank['x'])
    calibration = calibrators()
    records = []
    for seed in SEEDS:
        (means, collision_means) = predict(bank['x'], seed)
        prior = means @ weights
        order = np.random.default_rng(seed).permutation(len(prior))
        kernel = covariance(bank['x'], means)
        candidate = RiskTestingSession(bank['x'], kernel, means,
                                       collision_means, weights, calibration)
        with torch.no_grad():
            old = RiskGP(
                load_kernel(seed,
                            'global_feedback').covariance(bank['x']).numpy(),
                knn_prior)
        for count in range(1, 101):
            index = int(order[count - 1])
            candidate.pending = index
            candidate.records.append({})
            candidate.observe(float(bank['risk'][index]))
            old.observe(index, float(bank['risk'][index]))
            if count not in (10, 20, 50, 100):
                continue
            unseen = order[count:]
            truth = bank['risk'][unseen]
            for (name, prediction) in (('candidate',
                                        candidate.mean[:, 0].cpu().numpy()),
                                       ('frozen', old.mean),
                                       ('emulator_prior', prior), ('knn_prior',
                                                                   knn_prior)):
                prediction = prediction[unseen].clip(0, 1)
                tail = truth > 0.5
                records.append({
                    'seed':
                    seed,
                    'support_count':
                    count,
                    'method':
                    name,
                    'rmse':
                    float(np.sqrt(np.mean((prediction - truth)**2))),
                    'tail_rmse':
                    float(np.sqrt(np.mean(
                        (prediction[tail] - truth[tail])**2))),
                    'tail_ap':
                    float(average_precision_score(tail, prediction))
                })
    aggregate = {
        name: {
            key:
            float(
                np.mean([row[key] for row in records
                         if row['method'] == name]))
            for key in ('rmse', 'tail_rmse', 'tail_ap')
        }
        for name in ('candidate', 'frozen', 'emulator_prior', 'knn_prior')
    }
    write_json(
        OUTPUT / 'prediction_diagnostics.json', {
            'records': records,
            'aggregate': aggregate,
            'scope':
            'already-inspected original target; diagnostic, not confirmation or model selection',
            'support_counts': [10, 20, 50, 100],
            'target_feedback': 'queried continuous risk only'
        })
    print(aggregate, flush=True)


if __name__ == '__main__':
    main()
