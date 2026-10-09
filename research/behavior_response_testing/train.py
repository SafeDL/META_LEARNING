"""Fit scene/behavior response functions on sixteen historicalized profiles."""
from pathlib import Path

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.response_adaptive_testing.confirmation import CONFIRMATION, verify_lock

from .model import BehaviorResponseModel, behavior_coordinates, response_inputs, response_loss

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
SEEDS = (11, 23, 37, 53, 71)
STEPS = 4000
BATCH_SIZE = 512
VALIDATION_INTERVAL = 250


def dataset(names, profiles, family):
    coordinates, risks, collisions = [], [], []
    for name in names:
        behavior = behavior_coordinates(profiles[name])
        for replicate in range(2):
            path = CONFIRMATION / name / f"pool_{replicate}" / "responses.npz"
            with np.load(path) as bank:
                chosen = bank["x"][:, 4] == family
                coordinates.append(response_inputs(bank["x"][chosen],
                                                   behavior))
                risks.append(bank["risk"][chosen])
                collisions.append(bank["collision"][chosen])
    return tuple(
        torch.as_tensor(
            np.concatenate(values), dtype=torch.float32, device="cuda")
        for values in (coordinates, risks, collisions))


def main():
    torch.set_num_threads(1)
    first = read_json(CONFIRMATION / "summary.json")
    assert first["success"] is False
    verify_lock(first["protocol"])
    split = read_json(ROOT.parent / "response_adaptive_testing" / "results" /
                      "budget_training" / "protocol.json")
    profiles = {
        profile["name"]: profile
        for profile in first["protocol"]["profiles"]
    }
    training, validation = split["training_profiles"], split[
        "validation_profiles"]
    assert set(training).isdisjoint(validation)
    protocol = {
        "role":
        "Development predictor fitting; not target testing or independent confirmation",
        "training_profiles":
        training,
        "validation_profiles":
        validation,
        "historical_behavior_descriptors": {
            name: behavior_coordinates(profiles[name]).tolist()
            for name in training
        },
        "training_physical_records":
        len(training) * 2 * 2048,
        "validation_physical_records":
        len(validation) * 2 * 2048,
        "input":
        "Four scene coordinates and four offline behavioral coordinates: brake, gap, delay and controller kind",
        "architecture": [8, 128, 256, 128, 2],
        "seeds":
        SEEDS,
        "steps":
        STEPS,
        "batch_size":
        BATCH_SIZE,
        "validation_interval":
        VALIDATION_INTERVAL,
        "loss":
        "(1+4R) risk MSE + 0.1 collision BCE",
        "optimizer":
        "AdamW lr=0.001 weight_decay=0.0001",
        "parameter_access":
        "Known historical parameters train the forward model. A future selector must infer latent behavioral parameters from its queried risk; true target parameters cannot enter selection",
        "validation_role":
        "Known-profile predictor validation, not a permitted target-parameter oracle for search",
        "scope":
        "Same two controller families and three parameter ranges; no claim about arbitrary AD architectures"
    }
    write_json(OUTPUT / "protocol.json", protocol)
    data = {
        family: (dataset(training, profiles,
                         family), dataset(validation, profiles, family))
        for family in (0, 1)
    }
    for seed in SEEDS:
        output = OUTPUT / "models" / f"predictor_{seed}.pt"
        if output.exists():
            continue
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        states, curves, summaries = {}, {}, {}
        for family, (train, held_out) in data.items():
            model = BehaviorResponseModel().cuda()
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=0.001,
                                          weight_decay=0.0001)
            best_loss, curve, best_state, best_summary = float(
                "inf"), [], None, None
            for step in range(STEPS):
                indices = rng.integers(len(train[0]), size=BATCH_SIZE)
                prediction, logits = model(train[0][indices])
                loss = response_loss(prediction, logits, train[1][indices],
                                     train[2][indices])
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if (step + 1) % VALIDATION_INTERVAL == 0:
                    with torch.no_grad():
                        prediction, logits = model(held_out[0])
                        measured = float(
                            response_loss(prediction, logits, held_out[1],
                                          held_out[2]))
                        row = {
                            "step":
                            step + 1,
                            "loss":
                            measured,
                            "risk_rmse":
                            float((prediction -
                                   held_out[1]).square().mean().sqrt()),
                            "collision_bce":
                            float(
                                torch.nn.functional.
                                binary_cross_entropy_with_logits(
                                    logits, held_out[2])),
                            "collision_accuracy":
                            float(((logits
                                    > 0) == held_out[2].bool()).float().mean())
                        }
                    curve.append(row)
                    if measured < best_loss:
                        best_loss = measured
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                        best_summary = row
                    print("BEHAVIOR MODEL",
                          seed,
                          family,
                          step + 1,
                          round(row["risk_rmse"], 5),
                          round(row["collision_bce"], 5),
                          flush=True)
            states[str(family)] = best_state
            curves[str(family)] = curve
            summaries[str(family)] = best_summary
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(states, output)
        write_json(
            output.with_suffix(".json"), {
                "seed":
                seed,
                "known_profile_validation":
                summaries,
                "curve":
                curves,
                "role":
                "Development forward response predictor; not a search result"
            })


if __name__ == "__main__":
    main()
