"""Inspect working behavioral posteriors on already disclosed development only."""
import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.response_adaptive_testing.confirmation import CONFIRMATION as DEVELOPMENT
from research.response_adaptive_testing.develop import metrics

from .confirmation import CONFIRMATION, verify_lock
from .log_probability import LogProbabilityTestingSession
from .model import BehaviorResponseModel, behavior_coordinates
from .prediction import response_tables
from .train import OUTPUT

CHECKPOINTS = (0, 10, 30, 50, 100, 150, 200)


def posterior_snapshot(session, grid, true_behavior):
    weights = session.log_weights.exp().cpu().numpy()
    mean = weights @ grid
    low, high = [], []
    for dimension in range(3):
        order = np.argsort(grid[:, dimension], kind="stable")
        cumulative = weights[order].cumsum()
        for quantile, values in ((0.05, low), (0.95, high)):
            index = np.searchsorted(cumulative, quantile)
            values.append(float(grid[order[index], dimension]))
    probability = session.probabilities()
    remaining = session.remaining
    true_kind = int(true_behavior[3])
    type_probability = float(weights[grid[:, 3] == true_kind].sum())
    fvdm_probability = float(weights[grid[:, 3] == 1].sum())
    predicted_kind = int(fvdm_probability > 0.5 + 1e-12)
    return {
        "posterior_mean_normalized_parameters":
        mean[:3].tolist(),
        "working_90_interval_low":
        low,
        "working_90_interval_high":
        high,
        "true_kind_probability":
        type_probability,
        "predicted_kind_is_correct":
        predicted_kind == true_kind,
        "normalized_parameter_interval_contains_true":
        ((true_behavior[:3] >= low) & (true_behavior[:3] <= high)).tolist(),
        "posterior_entropy":
        float(-(weights * session.log_weights.cpu().numpy()).sum()),
        "expected_pool_failures":
        float(probability.sum()),
        "expected_unqueried_failures":
        float(probability[remaining].sum())
    }


def main():
    torch.set_num_threads(1)
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    options = protocol["candidate"]
    grid = np.array(options["continuous_behaviors"], dtype=np.float32)
    training = read_json(OUTPUT / "protocol.json")
    profiles = read_json(DEVELOPMENT / "protocol.json")["profiles"]
    records = []
    destination = OUTPUT / "behavior_diagnostics"
    write_json(
        destination / "protocol.json", {
            "role":
            "Past development diagnostics; never reads second-round target responses",
            "posterior":
            "Frozen second-round candidate, exact cached development selection parity",
            "parameters":
            "True configurations enter diagnostics after inference, never online selection",
            "intervals":
            "Working-model posterior quantiles, not calibrated physical confidence intervals",
            "mode_ties":
            "Predict IDM when FVDM probability is within 1e-12 of 0.5",
            "oracle":
            "Conditional collision NN evaluated at the true old configuration, fixed ranking; extra privileged information, not a fair baseline",
            "checkpoints": CHECKPOINTS
        })
    for seed in protocol["seeds"]:
        states = torch.load(OUTPUT / "models" / f"predictor_{seed}.pt",
                            map_location="cuda",
                            weights_only=True)
        models = []
        for family in (0, 1):
            model = BehaviorResponseModel().cuda().eval()
            model.load_state_dict(states[str(family)])
            models.append(model)
        for profile in profiles:
            name = profile["name"]
            true_behavior = behavior_coordinates(profile)
            for replicate in range(2):
                path = destination / name / f"pool_{replicate}" / f"trace_{seed}.json"
                if path.exists():
                    result = read_json(path)
                else:
                    with np.load(DEVELOPMENT / name / f"pool_{replicate}" /
                                 "responses.npz") as bank:
                        risk, logits = response_tables(bank["x"],
                                                       grid,
                                                       models,
                                                       return_logits=True)
                        session = LogProbabilityTestingSession(
                            bank["x"], risk, logits,
                            options["risk_discrepancy"][str(seed)])
                        trace = [{
                            "queries":
                            0,
                            **posterior_snapshot(session, grid, true_behavior)
                        }]
                        selected = []
                        while (index := session.next_index()) is not None:
                            session.observe(float(bank["risk"][index]))
                            selected.append(index)
                            if session.count in CHECKPOINTS:
                                trace.append({
                                    "queries":
                                    session.count,
                                    **posterior_snapshot(
                                        session, grid, true_behavior)
                                })
                        cached = read_json(
                            OUTPUT / "log_probability" / name /
                            f"pool_{replicate}" /
                            f"log_calibrated_behavior_{seed}.json")
                        assert cached["selected_indices"] == selected
                        _, true_logits = response_tables(
                            bank["x"],
                            true_behavior[None, :],
                            models,
                            return_logits=True)
                        oracle_order = torch.argsort(
                            true_logits[0], descending=True,
                            stable=True)[:200].cpu().tolist()
                        candidate = metrics(bank["collision"], selected)
                        oracle = metrics(bank["collision"], oracle_order)
                        count = candidate["pool_collisions"]
                        result = {
                            "profile":
                            name,
                            "replicate":
                            replicate,
                            "seed":
                            seed,
                            "split":
                            "forward_training"
                            if name in training["training_profiles"] else
                            "forward_validation",
                            "true_behavior":
                            true_behavior.tolist(),
                            "trace":
                            trace,
                            "candidate_metrics":
                            candidate,
                            "parameter_oracle_diagnostic":
                            oracle,
                            "parameter_oracle_area_minus_candidate":
                            (oracle["mean_cumulative_collisions"] -
                             candidate["mean_cumulative_collisions"]) /
                            count if count else 0,
                            "parameter_oracle_recall_minus_candidate":
                            oracle["recall"] - candidate["recall"],
                            "actual_pool_failures":
                            count,
                            "exact_cached_query_parity":
                            True
                        }
                        write_json(path, result)
                records.append(result)
                write_json(
                    destination / "summary.json", {
                        "role":
                        "Development interpretation only, not independent confirmation",
                        "records": records,
                        "complete": len(records) == 240
                    })
                print("BEHAVIOR DIAGNOSTIC", name, replicate, seed, flush=True)
    comparison = {}
    for split in ("all", "forward_training", "forward_validation"):
        rows = [
            row for row in records if split == "all" or row["split"] == split
        ]
        comparison[split] = {
            "runs":
            len(rows),
            "mean_oracle_area_gain":
            float(
                np.mean([
                    r["parameter_oracle_area_minus_candidate"] for r in rows
                ])),
            "mean_oracle_recall_gain":
            float(
                np.mean([
                    r["parameter_oracle_recall_minus_candidate"] for r in rows
                ])),
            "checkpoints": {
                str(count): {
                    "mean_true_kind_probability":
                    float(
                        np.mean([
                            r["trace"][i]["true_kind_probability"]
                            for r in rows
                        ])),
                    "kind_mode_accuracy":
                    float(
                        np.mean([
                            r["trace"][i]["predicted_kind_is_correct"]
                            for r in rows
                        ])),
                    "mean_parameter_interval_coverage":
                    np.mean([
                        r["trace"][i]
                        ["normalized_parameter_interval_contains_true"]
                        for r in rows
                    ],
                            axis=0).tolist(),
                    "mean_pool_count_bias":
                    float(
                        np.mean([
                            r["trace"][i]["expected_pool_failures"] -
                            r["actual_pool_failures"] for r in rows
                        ]))
                }
                for i, count in enumerate(CHECKPOINTS)
            }
        }
    write_json(destination / "comparison.json", comparison)
    print("BEHAVIOR DIAGNOSTIC COMPLETE", comparison, flush=True)


if __name__ == "__main__":
    main()
