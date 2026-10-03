"""One archive-improvement query followed by two global-risk queries."""
import time

import numpy as np

from .data import risk_logit
from .discrepancy import PoolDiscrepancyGP
from .qd import (
    RiskArchive,
    expected_archive_improvement,
    posterior_risk_mean,
)


def mode_at(query_number):
    if query_number < 1:
        raise ValueError("query numbers start at one")
    return "eai" if query_number % 3 == 1 else "risk"


class Selector:
    def __init__(self, prior, kernel, noise=0.001, jitter=1e-8, cell_count=256,
                 risk_threshold=0.5, mean_calibration=None):
        self.prior = prior
        self.kernel = kernel
        self.noise = noise
        self.jitter = jitter
        self.cell_count = cell_count
        self.risk_threshold = risk_threshold
        self.mean_calibration = mean_calibration

    def run(self, x, cells, oracle, budget=200, callback=None):
        if not 0 < budget <= len(x):
            raise ValueError("invalid query budget")
        gp = PoolDiscrepancyGP(self.kernel, x, self.prior.h, self.prior.m, self.noise, self.jitter,
                               mean_calibration=self.mean_calibration)
        archive = RiskArchive(risk_threshold=self.risk_threshold, cell_count=self.cell_count)
        selected, records = [], []
        for query_number in range(1, budget + 1):
            started = time.perf_counter()
            posterior = gp.predict()
            bounds = archive.threshold + archive.elite_quality[cells]
            eai = expected_archive_improvement(
                posterior.mean[:, 0], posterior.latent_var[:, 0], bounds
            )
            risk = posterior_risk_mean(
                posterior.mean[:, 0], posterior.latent_var[:, 0]
            )
            mode = mode_at(query_number)
            scores = (eai if mode == "eai" else risk).copy()
            scores[selected] = -np.inf
            index = int(np.argmax(scores))
            cell = int(cells[index])
            before = float(archive.elite_quality[cell])
            calibration = {}
            mean_at_index = float(gp.m[index, 0])
            if gp.coefficient_mean is not None:
                mean_at_index = float(gp.basis[index] @ gp.coefficient_mean)
                calibration = {
                    "mean_offset": float(gp.coefficient_mean[0]),
                    "mean_scale": float(gp.coefficient_mean[1]),
                    "mean_offset_sd": float(np.sqrt(gp.coefficient_covariance[0, 0])),
                    "mean_scale_sd": float(np.sqrt(gp.coefficient_covariance[1, 1])),
                }
            observation = oracle.query(index)
            selected.append(index)
            z, residual = None, None
            if observation.valid_risk:
                z = float(risk_logit(observation.risk))
                residual = z - mean_at_index
                gp.observe(index, observation.risk)
                archive.observe(cell, observation.risk)

            available = np.ones(len(x), dtype=bool)
            available[selected[:-1]] = False

            def rank(values):
                ahead = (values > values[index]) | (
                    (values == values[index]) & (np.arange(len(x)) < index)
                )
                return 1 + int(np.sum(available & ahead))

            row = {
                "index": index,
                "scenario_id": observation.scenario_id,
                "query_number": query_number,
                "acquisition_mode": mode,
                "m": float(self.prior.m[index, 0]),
                "m0": mean_at_index,
                "mu": float(posterior.mean[index, 0]),
                "latent_var": float(posterior.latent_var[index, 0]),
                "EAI": float(eai[index]),
                "posterior_risk_mean": float(risk[index]),
                "risk_rank": rank(risk),
                "EAI_rank": rank(eai),
                "cell": cell,
                "elite_before": before,
                "elite_after": float(archive.elite_quality[cell]),
                "archive_gain": float(archive.elite_quality[cell]) - before,
                "label": observation.collision,
                "risk": observation.risk,
                "valid_risk": observation.valid_risk,
                "z": z,
                "e": residual,
                "noise_variance": self.noise,
                "jitter": self.jitter,
                "elapsed_s": time.perf_counter() - started,
                "risk_archive": archive.metrics(),
                **calibration,
            }
            records.append(row)
            if callback is not None:
                callback(row)
        posterior = gp.predict()
        result = {
            "selected_indices": selected,
            "queries": records,
            "final_mean": posterior.mean[:, 0].tolist(),
            "final_latent_var": posterior.latent_var[:, 0].tolist(),
            "final_risk_mean": posterior_risk_mean(
                posterior.mean[:, 0], posterior.latent_var[:, 0]
            ).tolist(),
            "risk_archive": archive.metrics(),
        }
        if gp.coefficient_mean is not None:
            result["mean_calibration"] = {
                "prior": self.mean_calibration,
                "coefficient_mean": gp.coefficient_mean.tolist(),
                "coefficient_covariance": gp.coefficient_covariance.tolist(),
                "model": "z = offset + scale * historical_m + discrepancy",
            }
        return result
