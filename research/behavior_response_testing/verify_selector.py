"""Validate isolated selectors against disclosed development query sequences."""
import multiprocessing as mp

import numpy as np
import torch

from methods.history_guided_testing.history import historical_risk, load_history, split_indices, subset
from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import predict
from research.history_response_testing.scenarios import ras_predictions
from research.response_adaptive_testing.confirmation import CONFIRMATION as DEVELOPMENT

from .confirm import FROZEN_METHODS, disclose, selector_worker
from .log_probability import LogProbabilityTestingSession
from .model import BehaviorResponseModel
from .prediction import behavior_grid, response_tables
from .train import OUTPUT


def main():
    torch.set_num_threads(1)
    seed = 11
    training = read_json(OUTPUT / "protocol.json")
    options = read_json(OUTPUT / "development_protocol.json")
    calibrators = {
        int(key): value
        for key, value in options["historical_risk_calibration"].items()
    }
    discrepancy = read_json(OUTPUT /
                            "calibration/discrepancy_11.json")["parameters"]
    grid = behavior_grid()
    historical = np.array(list(
        training["historical_behavior_descriptors"].values()),
                          dtype=np.float32)
    states = torch.load(OUTPUT / "models/predictor_11.pt",
                        map_location="cuda",
                        weights_only=True)
    models = []
    for family in (0, 1):
        model = BehaviorResponseModel().cuda().eval()
        model.load_state_dict(states[str(family)])
        models.append(model)
    history = load_history()
    train, _ = split_indices(next(iter(history.values()))["x"])
    allowed = subset(history, train)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=selector_worker, args=(child, ))
    process.start()
    child.close()
    checks = []
    try:
        for name in ("idm_07", "fvdm_11"):
            pool = DEVELOPMENT / name / "pool_0"
            with np.load(pool / "responses.npz") as bank:
                x = bank["x"]
                original, ras = predict(x, seed), ras_predictions(x, seed)
                prior = historical_risk(allowed, x)
                continuous = tuple(value.cpu().numpy()
                                   for value in response_tables(
                                       x, grid, models, return_logits=True))
                history_tensors = response_tables(x,
                                                  historical,
                                                  models,
                                                  return_logits=True)
                discrete = tuple(value.cpu().numpy()
                                 for value in history_tensors)
                matched = (discrete[0].T,
                           history_tensors[1].sigmoid().cpu().numpy().T)
                for method in (*FROZEN_METHODS, "candidate",
                               "matched_collision_gp", "fixed_behavior",
                               "discrete_history"):
                    task = {"method": method, "seed": seed, "x": x}
                    if method in FROZEN_METHODS:
                        task.update(prediction=original, ras=ras, prior=prior)
                        expected = read_json(
                            pool / "selection" /
                            f"{method}_{seed}.json")["selected_indices"]
                    elif method == "matched_collision_gp":
                        task.update(prediction=matched,
                                    calibrators=calibrators)
                        expected = read_json(
                            OUTPUT / "development" / name / "pool_0" /
                            f"matched_collision_gp_{seed}.json"
                        )["selected_indices"]
                    else:
                        prediction = discrete if method == "discrete_history" else continuous
                        task.update(prediction=prediction,
                                    discrepancy=discrepancy)
                        direct = LogProbabilityTestingSession(
                            x, *prediction, discrepancy)
                        if method == "fixed_behavior":
                            expected = torch.argsort(
                                direct.log_safe_probabilities(),
                                stable=True)[:200].tolist()
                        else:
                            expected = []
                            while (index := direct.next_index()) is not None:
                                direct.observe(float(bank["risk"][index]))
                                expected.append(index)
                            if method == "candidate":
                                saved = read_json(
                                    OUTPUT / "log_probability" / name /
                                    "pool_0" /
                                    f"log_calibrated_behavior_{seed}.json")
                                assert saved["selected_indices"] == expected
                    result = disclose(parent, task, bank)
                    assert result["selected_indices"] == expected, (name,
                                                                    method)
                    checks.append({
                        "profile": name,
                        "method": method,
                        "exact_200_query_parity": True,
                        "F200": result["F200"]
                    })
                    print("ISOLATED SELECTOR VERIFIED",
                          name,
                          method,
                          flush=True)
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
        process.join()
        assert process.exitcode == 0
    write_json(
        OUTPUT / "selector_verification.json", {
            "role":
            "Verification on disclosed development, no prospective target outcome",
            "checks": checks,
            "verified_disclosures": len(checks) * 200,
            "no_target_parameters_in_payload": True,
            "no_full_responses_in_payload": True,
            "all_checks_passed": True
        })


if __name__ == "__main__":
    main()
