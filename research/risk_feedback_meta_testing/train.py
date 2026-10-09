"""Compare ordinary supervision with deployment-matched risk meta-adaptation."""
import copy

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.model import behavior_coordinates
from research.behavior_response_testing.prediction import behavior_grid

from .config import (COHORT, LEARNING_RATE, MODELS, QUERY_COUNT, RESULTS, SEEDS,
                     SUPPORT_COUNTS, TRAINING_STEPS, VALIDATION_INTERVAL,
                     WEIGHT_DECAY)
from .model import FailureDecoder, FrozenResponseBackbone
from .posterior import condition_risk


def load_banks(names):
    result = {}
    for name in names:
        for replicate in range(2):
            folder = COHORT / name / f"pool_{replicate}"
            with np.load(folder / "responses.npz") as bank:
                result[name, replicate] = {
                    key: bank[key].copy() for key in ("x", "risk", "collision")
                }
            selected = read_json(folder / "selection" / "candidate_11.json")
            result[name, replicate]["frozen_order"] = np.asarray(
                selected["selected_indices"], dtype=np.int64)
    return result


def sample_task(name, replicate, count, policy, banks, rng):
    bank = banks[name, replicate]
    if policy == "uniform":
        support = rng.choice(len(bank["x"]), count, replace=False)
    else:
        support = bank["frozen_order"][:count]
    remaining = np.setdiff1d(np.arange(len(bank["x"])), support)
    query = rng.choice(remaining, QUERY_COUNT, replace=False)
    return name, replicate, support, query


@torch.no_grad()
def residual_statistics(backbone, banks, names, profiles):
    moments = np.zeros((2, 3))
    device = next(backbone.parameters()).device
    for name in names:
        behavior = torch.as_tensor(behavior_coordinates(profiles[name]),
                                   device=device)[None, :]
        for replicate in range(2):
            bank = banks[name, replicate]
            x = torch.as_tensor(bank["x"], dtype=torch.float32, device=device)
            prediction = backbone.features(x, behavior)[0][0].cpu().numpy()
            residual = bank["risk"] - prediction
            for family in (0, 1):
                values = residual[bank["x"][:, 4] == family]
                moments[family] += [len(values), values.sum(), np.square(values).sum()]
    center = moments[:, 1] / moments[:, 0]
    scale = np.sqrt(moments[:, 2] / moments[:, 0] - center ** 2)
    return center, scale


def meta_inputs(task, banks, backbone, grid, discrepancy):
    name, replicate, support, query = task
    bank = banks[name, replicate]
    indices = np.concatenate((support, query))
    device = grid.device
    coordinates = torch.as_tensor(bank["x"][indices], dtype=torch.float64,
                                   device=device)
    prediction, features = backbone.features(coordinates.float(), grid)
    weights, mean, variance = condition_risk(
        coordinates, prediction.double(), np.arange(len(support)),
        bank["risk"][support], discrepancy)
    selected = slice(len(support), None)
    return (features[:, selected], coordinates[selected, 4].long(), weights,
            mean[:, selected], variance[:, selected])


def supervised_inputs(task, banks, profiles, backbone):
    name, replicate, _, query = task
    bank = banks[name, replicate]
    device = next(backbone.parameters()).device
    coordinates = torch.as_tensor(bank["x"][query], dtype=torch.float32,
                                   device=device)
    behavior = torch.as_tensor(behavior_coordinates(profiles[name]),
                               device=device)[None, :]
    prediction, features = backbone.features(coordinates, behavior)
    residual = (torch.as_tensor(bank["risk"][query], dtype=torch.float64,
                               device=device)[None, :] - prediction.double())
    return (features, coordinates[:, 4].long(), residual.new_zeros(1),
            residual, residual.new_zeros(len(query)))


def collision_loss(decoder, inputs, labels):
    collision, safe = decoder.log_probabilities(*inputs)
    return -(labels * collision + (1 - labels) * safe).mean()


@torch.no_grad()
def validate(decoders, tasks, banks, backbone, grid, discrepancy):
    losses = {mode: [] for mode in decoders}
    briers = {mode: [] for mode in decoders}
    for task in tasks:
        inputs = meta_inputs(task, banks, backbone, grid, discrepancy)
        name, replicate, _, query = task
        labels = torch.as_tensor(banks[name, replicate]["collision"][query],
                                 dtype=torch.float64, device=grid.device)
        for mode, decoder in decoders.items():
            collision, safe = decoder.log_probabilities(*inputs)
            losses[mode].append(float(
                -(labels * collision + (1 - labels) * safe).mean()))
            briers[mode].append(float((collision.exp() - labels).square().mean()))
    return {
        mode: {"log_loss": float(np.mean(losses[mode])),
               "brier_score": float(np.mean(briers[mode]))}
        for mode in decoders
    }


def fit_seed(seed, protocol, banks):
    paths = {mode: RESULTS / "models" / f"decoder_{mode}_{seed}.pt"
             for mode in ("supervised", "meta")}
    manifest_path = RESULTS / "models" / f"training_{seed}.json"
    if manifest_path.exists() and all(path.exists() for path in paths.values()):
        completed = read_json(manifest_path)
        if (completed["steps_completed"] != TRAINING_STEPS
                or completed["training_profiles"] != protocol["split"]["training"]
                or completed["validation_profiles"] != protocol["split"]["validation"]):
            raise ValueError(f"Completed training does not match the current study: {seed}")
        return
    torch.manual_seed(seed)
    states = torch.load(MODELS / f"predictor_{seed}.pt",
                        map_location="cuda", weights_only=True)
    backbone = FrozenResponseBackbone(states).cuda()
    training, validation = (protocol["split"][role]
                            for role in ("training", "validation"))
    profiles = {profile["name"]: profile for profile in protocol["profiles"]}
    center, scale = residual_statistics(backbone, banks, training, profiles)
    weight, bias = backbone.initial_head()
    decoders = {mode: FailureDecoder(weight, bias, center, scale).cuda()
                for mode in ("supervised", "meta")}
    optimizers = {
        mode: torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY)
        for mode, decoder in decoders.items()
    }
    grid = torch.as_tensor(behavior_grid(), device="cuda")
    discrepancy = read_json(COHORT / "protocol.json")["candidate"][
        "risk_discrepancy"][str(seed)]
    rng = np.random.default_rng(seed)
    validation_rng = np.random.default_rng(20261027)
    validation_tasks = [
        sample_task(name, offset % 2, count,
                    "uniform" if offset % 2 == 0 else "frozen", banks,
                    validation_rng)
        for name in validation for offset, count in enumerate(SUPPORT_COUNTS)
    ]
    best_loss = {mode: float("inf") for mode in decoders}
    best_state, curves = {}, []
    for step in range(TRAINING_STEPS):
        name = training[int(rng.integers(len(training)))]
        count = SUPPORT_COUNTS[int(rng.integers(len(SUPPORT_COUNTS)))]
        task = sample_task(name, int(rng.integers(2)), count,
                           "uniform" if step % 2 == 0 else "frozen", banks, rng)
        query = task[3]
        labels = torch.as_tensor(banks[task[0], task[1]]["collision"][query],
                                 dtype=torch.float64, device="cuda")
        inputs = {
            "supervised": supervised_inputs(task, banks, profiles, backbone),
            "meta": meta_inputs(task, banks, backbone, grid, discrepancy),
        }
        for mode, decoder in decoders.items():
            optimizer = optimizers[mode]
            optimizer.zero_grad()
            loss = collision_loss(decoder, inputs[mode], labels)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss: {seed}, {mode}, {step}")
            loss.backward()
            optimizer.step()
        if (step + 1) % VALIDATION_INTERVAL == 0:
            measured = validate(decoders, validation_tasks, banks, backbone,
                                grid, discrepancy)
            curves.append({"step": step + 1, **measured})
            for mode, values in measured.items():
                if values["log_loss"] < best_loss[mode]:
                    best_loss[mode] = values["log_loss"]
                    best_state[mode] = copy.deepcopy(decoders[mode].state_dict())
            print("META TRAIN", seed, step + 1, measured, flush=True)
    for mode, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.cpu() for key, value in best_state[mode].items()}, path)
    write_json(manifest_path, {
        "seed": seed, "steps_completed": TRAINING_STEPS,
        "training_profiles": training, "validation_profiles": validation,
        "risk_backbone_frozen": True, "risk_gp_frozen": True,
        "risk_observation_model": "censored Gaussian; same moment filter during training and deployment",
        "trainable_parameters_per_mode": sum(p.numel() for p in decoders["meta"].parameters()),
        "collision_labels_in_target_context": False,
        "query_risk_in_meta_input": False,
        "same_query_training_labels_and_optimization_steps": True,
        "residual_center": center.tolist(), "residual_scale": scale.tolist(),
        "checkpoint_selection": "lowest validation deployment-conditioned collision log loss",
        "best_log_loss": best_loss, "validation_curve": curves,
    })


def main():
    torch.set_num_threads(1)
    protocol = read_json(RESULTS / "protocol.json")
    names = protocol["split"]["training"] + protocol["split"]["validation"]
    banks = load_banks(names)
    for seed in SEEDS:
        fit_seed(seed, protocol, banks)


if __name__ == "__main__":
    main()
