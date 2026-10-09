"""Show inferred response coefficients and their uncertainty on development data."""
import math

import numpy as np
import torch

from methods.history_guided_testing.config import ROOT as BASELINE
from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import SOURCE_NAMES, predict

from .config import OUTPUT
from .session import AdaptiveTestingSession

CHECKPOINTS = (0, 1, 5, 10, 25, 50, 100, 200)


def summary(session, coefficient_covariance, source_count, scale):
    center = torch.eye(source_count,
                       device=session.mean_r.device,
                       dtype=torch.float64)
    center -= torch.ones_like(center) / source_count
    means, deviations = [], []
    for family in (0, 1):
        indices = slice(family * source_count, (family + 1) * source_count)
        mean = 1 / source_count + scale * (
            center @ session.coefficients[indices])
        covariance = scale**2 * center @ coefficient_covariance[
            indices, indices] @ center.T
        means.append(mean.cpu().tolist())
        deviations.append(covariance.diag().clamp(min=0).sqrt().cpu().tolist())
    return {
        "query_count": session.count,
        "effective_response_coefficients": means,
        "posterior_standard_deviation": deviations
    }


def main():
    torch.set_num_threads(1)
    options = read_json(OUTPUT /
                        "confirmation/protocol.json")["candidate_options"]
    pools = [("original", BASELINE / "target/responses.npz")]
    pools += [(path.parent.name, path)
              for path in sorted((OUTPUT /
                                  "development_pools").glob("*/responses.npz"))
              ]
    for name, path in pools:
        bank = np.load(path)
        prediction = predict(bank["x"], 11)
        session = AdaptiveTestingSession(bank["x"], *prediction, options)
        source_count = prediction[0].shape[1]
        coefficient_covariance = torch.eye(2 * source_count,
                                           dtype=torch.float64,
                                           device="cuda")
        scale = math.sqrt(options["source_scale"] / (source_count - 1))
        trace = [summary(session, coefficient_covariance, source_count, scale)]
        selected = []
        while (index := session.next_index()) is not None:
            cross = session.coefficient_r[:, index].clone()
            denominator = session.rr[index, index] + options["noise"]
            coefficient_covariance -= cross[:, None] * cross[
                None, :] / denominator
            session.observe(float(bank["risk"][index]))
            selected.append(index)
            if session.count in CHECKPOINTS:
                trace.append(
                    summary(session, coefficient_covariance, source_count,
                            scale))
        write_json(
            OUTPUT / "response_coefficients" / f"{name}.json", {
                "sources":
                SOURCE_NAMES,
                "trace":
                trace,
                "selected_indices":
                selected,
                "source_coefficient_covariance":
                coefficient_covariance.cpu().tolist(),
                "interpretation":
                "signed coefficients in latent risk/probit response coordinates; not source identity probabilities",
                "stage":
                "development explanation; not prospective target adaptation or parameter selection"
            })
        print("RESPONSE COEFFICIENTS",
              name,
              trace[-1]["effective_response_coefficients"],
              flush=True)


if __name__ == "__main__":
    main()
