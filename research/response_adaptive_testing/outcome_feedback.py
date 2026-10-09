"""Development comparison using real observed failures and Gaussian moment filtering."""
import time

import numpy as np
from scipy.special import ndtri
import torch

from methods.history_guided_testing.config import ROOT as BASELINE
from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import predict

from .config import BUDGET, OUTPUT, RISK_EPSILON, SEEDS
from .confirmation import CONFIRMATION, verify_lock
from .develop import metrics
from .model import joint_prior, risk_coordinate

RESULTS = OUTPUT / "outcome_feedback"


def probit_factors(mean, variance, collision):
    denominator = 1 + variance
    sign = 1 if collision else -1
    standardized = sign * mean / denominator.sqrt()
    log_density = -0.5 * standardized.square() - 0.5 * np.log(2 * np.pi)
    ratio = (log_density - torch.special.log_ndtr(standardized)).exp()
    curvature = (ratio * (ratio + standardized)).clamp(0, 1)
    return sign * ratio / denominator.sqrt(), curvature / denominator


class OutcomeTestingSession:

    def __init__(self,
                 x,
                 prediction,
                 options,
                 learned=None,
                 use_risk=True,
                 budget=BUDGET,
                 device="cuda"):
        self.mean_r, self.mean_c, self.rr, self.cr, self.cc, _ = joint_prior(
            x, *prediction, options, device, full_collision=True)
        if learned is not None:
            family = torch.as_tensor(x[:, 4], dtype=torch.long, device=device)
            values = torch.as_tensor(ndtri(
                np.clip(prediction[1], RISK_EPSILON, 1 - RISK_EPSILON)),
                                     dtype=self.mean_c.dtype,
                                     device=device)
            logits = self.mean_c.new_tensor(learned["mean_logits"])[1]
            weights = logits[family].softmax(1)
            self.mean_c = (values * weights).sum(1) + self.mean_c.new_tensor(
                options["collision_offset"])[family]
        self.options, self.use_risk, self.budget = options, use_risk, budget
        self.remaining = torch.ones(len(x), dtype=torch.bool, device=device)
        self.pending, self.count, self.records = None, 0, []

    def probabilities(self):
        return torch.special.ndtr(self.mean_c /
                                  (1 + self.cc.diag().clamp(min=0)).sqrt())

    def next_index(self):
        if self.pending is not None:
            raise RuntimeError("Observe the pending query first")
        if self.count == self.budget:
            return None
        q = self.probabilities()
        self.pending = int(q.masked_fill(~self.remaining, -torch.inf).argmax())
        return self.pending

    def observe(self, risk, collision):
        if self.pending is None:
            raise RuntimeError("Request a query before feedback")
        if not np.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Invalid continuous risk")
        if not isinstance(collision, (bool, np.bool_)):
            raise ValueError("Failure feedback must be a measured boolean")
        index = self.pending
        if self.use_risk:
            cross_r, cross_c = self.rr[:,
                                       index].clone(), self.cr[:,
                                                               index].clone()
            denominator = self.rr[index, index] + self.options["noise"]
            residual = float(risk_coordinate(
                risk, self.options["transform"])) - self.mean_r[index]
            self.mean_r += cross_r / denominator * residual
            self.mean_c += cross_c / denominator * residual
            self.rr -= cross_r[:, None] * cross_r[None, :] / denominator
            self.cr -= cross_c[:, None] * cross_r[None, :] / denominator
            self.cc -= cross_c[:, None] * cross_c[None, :] / denominator
        cross_r, cross_c = self.cr[index].clone(), self.cc[:, index].clone()
        mean_factor, covariance_factor = probit_factors(
            self.mean_c[index], self.cc[index, index], collision)
        self.mean_r += mean_factor * cross_r
        self.mean_c += mean_factor * cross_c
        self.rr -= covariance_factor * cross_r[:, None] * cross_r[None, :]
        self.cr -= covariance_factor * cross_c[:, None] * cross_r[None, :]
        self.cc -= covariance_factor * cross_c[:, None] * cross_c[None, :]
        self.records.append({
            "index": index,
            "continuous_risk": float(risk),
            "collision": bool(collision)
        })
        self.remaining[index] = False
        self.count += 1
        self.pending = None


def main():
    torch.set_num_threads(1)
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    options = protocol["candidate_options"]
    learned = read_json(OUTPUT /
                        "initial_prior/models/collision_mean_600.json")
    assert learned["target_data_used"] is False
    pools = [("original", BASELINE / "target/responses.npz")]
    pools += [(path.parent.name, path)
              for path in sorted((OUTPUT /
                                  "development_pools").glob("*/responses.npz"))
              ]
    records = []
    for name, bank_path in pools:
        with np.load(bank_path) as bank:
            for seed in SEEDS:
                prediction = predict(bank["x"], seed)
                for policy, use_risk in (("risk_and_collision", True),
                                         ("collision_only", False)):
                    path = RESULTS / "development" / f"{name}_{policy}_{seed}.json"
                    if path.exists():
                        result = read_json(path)
                    else:
                        started = time.perf_counter()
                        session = OutcomeTestingSession(
                            bank["x"], prediction, options, learned, use_risk)
                        selected = []
                        while (index := session.next_index()) is not None:
                            session.observe(float(bank["risk"][index]),
                                            bool(bank["collision"][index]))
                            selected.append(index)
                        result = {
                            **metrics(bank["collision"], selected), "selected_indices":
                            selected,
                            "observations":
                            session.records,
                            "options":
                            options,
                            "mean_model":
                            learned,
                            "elapsed_s":
                            time.perf_counter() - started,
                            "role":
                            "Development with real failure feedback; not risk-only confirmation",
                            "inference":
                            "Sequential Gaussian moment approximation after probit observations"
                        }
                        write_json(path, result)
                    selected = result["selected_indices"]
                    assert len(selected) == len(set(selected)) == BUDGET
                    assert result["observations"] == [{
                        "index":
                        index,
                        "continuous_risk":
                        float(bank["risk"][index]),
                        "collision":
                        bool(bank["collision"][index])
                    } for index in selected]
                    measured = metrics(bank["collision"], selected)
                    for key, value in measured.items():
                        assert result[key] == value
                    records.append({
                        "pool": name,
                        "seed": seed,
                        "policy": policy,
                        **measured
                    })
                    print("OBSERVED FAILURE DEVELOPMENT",
                          name,
                          seed,
                          policy,
                          round(measured["mean_cumulative_collisions"], 3),
                          measured["F200"],
                          flush=True)
                    write_json(
                        RESULTS / "summary.json", {
                            "role":
                            "Real risk and collision feedback development; first confirmation unchanged",
                            "records": records,
                            "complete": len(records)
                            == len(pools) * len(SEEDS) * 2,
                            "unqueried_target_labels_in_policy": False
                        })


if __name__ == "__main__":
    main()
