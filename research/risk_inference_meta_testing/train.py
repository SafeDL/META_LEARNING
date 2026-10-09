"""A fixed two-seed comparison of point supervision and learned risk inference."""
import copy
import time

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.model import behavior_coordinates
from research.behavior_response_testing.prediction import behavior_grid
from research.risk_feedback_meta_testing.evaluate import verify_or_lock
from research.risk_feedback_meta_testing.model import (FrozenResponseBackbone,
                                                       collision_probabilities)
from research.risk_feedback_meta_testing.train import load_banks, sample_task

from .config import (BACKBONES, BUDGET, COHORT, KERNEL_NUGGET, LEARNING_RATE,
                     PREVIOUS, RESULTS, SEEDS, SUPPORT_COUNTS, TRAINING_STEPS,
                     VALIDATION_INTERVAL, WARMUP_STEPS, WEIGHT_DECAY)
from .model import ResponsePrior, failure_log_probabilities, risk_log_likelihood
from .posterior import condition_field, point_condition


@torch.no_grad()
def known_features(backbone, names, banks, profiles):
    result, moments = {}, np.zeros((2, 257))
    for name in names:
        behavior = torch.as_tensor(behavior_coordinates(profiles[name]),
                                   device="cuda")[None, :]
        for replicate in range(2):
            bank = banks[name, replicate]
            x = torch.as_tensor(bank["x"], dtype=torch.float32, device="cuda")
            risk, features = backbone.features(x, behavior)
            result[name, replicate] = risk, features
            for family in (0, 1):
                hidden = features[0, x[:, 4] == family].double()
                moments[family, 0] += len(hidden)
                moments[family, 1:129] += hidden.sum(0).cpu().numpy()
                moments[family, 129:] += hidden.square().sum(0).cpu().numpy()
    center = moments[:, 1:129] / moments[:, :1]
    scale = np.sqrt(np.maximum(moments[:, 129:] / moments[:, :1] - center ** 2,
                               1e-8))
    return result, center, scale


def point_inputs(model, task, banks, cached):
    name, replicate, _, query = task
    bank = banks[name, replicate]
    frozen, features = cached[name, replicate]
    features, frozen = features[:, query], frozen[:, query]
    family = torch.as_tensor(bank["x"][query, 4], dtype=torch.long, device="cuda")
    observed = torch.as_tensor(bank["risk"][query], dtype=torch.float64, device="cuda")
    fields = model.risk_fields(features, family, frozen)
    variance = fields[1].square() * (1 + KERNEL_NUGGET) + fields[2].square()
    risk_loss = -risk_log_likelihood(fields[0], variance, observed).mean()
    mean, variance = point_condition(*fields, observed, 1 + KERNEL_NUGGET)
    return risk_loss, (features, family, mean.new_zeros(1), mean, variance)


def meta_inputs(model, task, banks, backbone, grid, lengths):
    name, replicate, support, query = task
    bank = banks[name, replicate]
    selected = np.concatenate((support, query))
    x = torch.as_tensor(bank["x"][selected], dtype=torch.float64, device="cuda")
    frozen, features = backbone.features(x.float(), grid)
    fields = model.risk_fields(features, x[:, 4].long(), frozen)
    weights, mean, variance = condition_field(
        x, *fields, np.arange(len(support)), bank["risk"][support], lengths)
    remaining = slice(len(support), None)
    return (features[:, remaining], x[remaining, 4].long(), weights,
            mean[:, remaining], variance[:, remaining])


def collision_loss(model, inputs, labels):
    collision, safe = model.failure_logs(*inputs)
    return -(labels * collision + (1 - labels) * safe).mean()


@torch.no_grad()
def validate(models, names, banks, profiles, backbone, grid, lengths):
    records = []
    for position, name in enumerate(names):
        replicate = position % 2
        bank = banks[name, replicate]
        x = torch.as_tensor(bank["x"], dtype=torch.float64, device="cuda")
        frozen, features = backbone.features(x.float(), grid)
        family = x[:, 4].long()
        rng = np.random.default_rng(20261031 + position)
        supports = {"uniform": rng.permutation(len(x))[:150],
                    "historical": bank["frozen_order"][:150]}
        for mode, model in models.items():
            fields = model.risk_fields(features, family, frozen)
            failure_means = model.failure_means(features, family)
            for policy, maximum_support in supports.items():
                for count in (0, 10, 50, 150):
                    support = maximum_support[:count]
                    query = np.setdiff1d(np.arange(len(x)), support)
                    weights, mean, variance = condition_field(
                        x, *fields, support, bank["risk"][support], lengths)
                    collision, safe = failure_log_probabilities(
                        failure_means[:, query], family[query], weights,
                        mean[:, query], variance[:, query], model.sensitivity)
                    labels = torch.as_tensor(bank["collision"][query],
                                             dtype=torch.float64, device="cuda")
                    probability = collision_probabilities(collision, safe)
                    selected = torch.topk(collision - safe, BUDGET - count).indices
                    future_found = float(labels[selected].sum())
                    support_found = int(bank["collision"][support].sum())
                    total = int(bank["collision"].sum())
                    true_flag = profiles[name]["controller"] == "FVDM"
                    posterior_mass = float(weights.exp()[grid[:, 3] == true_flag].sum())
                    records.append({
                        "profile": name, "controller": profiles[name]["controller"],
                        "replicate": replicate, "mode": mode, "policy": policy,
                        "support_count": count, "unqueried_count": len(query),
                        "log_loss": float(-(labels * collision + (1 - labels) * safe).mean()),
                        "brier_score": float((probability - labels).square().mean()),
                        "static_continuation_failures": future_found,
                        "support_failures_for_audit_only": support_found,
                        "static_projected_F200": support_found + future_found,
                        "static_budget_deficit": min(BUDGET, total) - support_found - future_found,
                        "true_controller_mass_for_audit_only": posterior_mass,
                    })
        print("INFERENCE VALIDATION", name, len(records), flush=True)
    aggregate = {}
    for mode in models:
        historical = [row for row in records if row["mode"] == mode
                      and row["policy"] == "historical" and row["support_count"] in (10, 50)]
        aggregate[mode] = {
            "historical_early_log_loss": float(np.mean([row["log_loss"] for row in historical])),
            "historical_early_static_deficit":
                float(np.mean([row["static_budget_deficit"] for row in historical])),
            "historical_early_controller_mass":
                float(np.mean([row["true_controller_mass_for_audit_only"] for row in historical])),
        }
    return {"aggregate": aggregate, "records": records}


def fit_seed(seed, protocol, banks):
    torch.manual_seed(seed)
    states = torch.load(BACKBONES / f"predictor_{seed}.pt",
                        map_location="cuda", weights_only=True)
    backbone = FrozenResponseBackbone(states).cuda()
    profiles = {row["name"]: row for row in protocol["profiles"]}
    training, validation = (protocol["split"][key] for key in ("training", "validation"))
    cached, center, scale = known_features(backbone, training, banks, profiles)
    discrepancy = read_json(COHORT / "protocol.json")["candidate"]["risk_discrepancy"][str(seed)]
    lengths = discrepancy["length"]
    weight, bias = backbone.initial_head()
    initial = ResponsePrior(weight, bias, center, scale, discrepancy).cuda()
    optimizer = torch.optim.AdamW(initial.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    for step in range(WARMUP_STEPS):
        name = training[int(rng.integers(len(training)))]
        task = sample_task(name, int(rng.integers(2)), 0, "uniform", banks, rng)
        loss = point_inputs(initial, task, banks, cached)[0]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(initial.parameters(), 10)
        optimizer.step()
        if (step + 1) % 50 == 0:
            print("INFERENCE WARMUP", seed, step + 1, float(loss.detach()), flush=True)
    models = {mode: copy.deepcopy(initial) for mode in ("supervised", "meta")}
    optimizers = {mode: torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                        weight_decay=WEIGHT_DECAY)
                  for mode, model in models.items()}
    grid = torch.as_tensor(behavior_grid(), device="cuda")
    curves, best, best_state = [], {mode: float("inf") for mode in models}, {}
    validation_output = RESULTS / "validation"
    validation_output.mkdir(parents=True, exist_ok=True)
    for step in range(WARMUP_STEPS, TRAINING_STEPS):
        name = training[int(rng.integers(len(training)))]
        count = SUPPORT_COUNTS[int(rng.integers(len(SUPPORT_COUNTS)))]
        task = sample_task(name, int(rng.integers(2)), count,
                           "uniform" if step % 2 == 0 else "frozen", banks, rng)
        labels = torch.as_tensor(banks[task[0], task[1]]["collision"][task[3]],
                                 dtype=torch.float64, device="cuda")
        for mode, model in models.items():
            risk_loss, point = point_inputs(model, task, banks, cached)
            inputs = point if mode == "supervised" else meta_inputs(
                model, task, banks, backbone, grid, lengths)
            loss = risk_loss + collision_loss(model, inputs, labels)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite inference loss: {seed}, {mode}, {step}")
            optimizers[mode].zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            if not torch.isfinite(norm):
                raise ValueError(f"Nonfinite inference gradient: {seed}, {mode}, {step}")
            optimizers[mode].step()
        if (step + 1) % 50 == 0:
            print("INFERENCE TRAIN", seed, step + 1, float(loss.detach()),
                  time.perf_counter() - start, flush=True)
        if (step + 1) % VALIDATION_INTERVAL == 0:
            measurement = validate(models, validation, banks, profiles, backbone,
                                   grid, lengths)
            write_json(validation_output / f"seed_{seed}_step_{step+1}.json", measurement)
            curves.append({"step": step + 1, **measurement["aggregate"]})
            for mode, value in measurement["aggregate"].items():
                score = value["historical_early_log_loss"]
                if score < best[mode]:
                    best[mode] = score
                    best_state[mode] = copy.deepcopy(models[mode].state_dict())
            print("INFERENCE CHECKPOINT", seed, step + 1, measurement["aggregate"], flush=True)
    folder = RESULTS / "models"
    folder.mkdir(parents=True, exist_ok=True)
    for mode, state in best_state.items():
        torch.save({key: value.cpu() for key, value in state.items()},
                   folder / f"prior_{mode}_{seed}.pt")
    write_json(folder / f"training_{seed}.json", {
        "seed": seed, "steps_per_mode_including_common_warmup": TRAINING_STEPS,
        "shared_risk_warmup_steps": WARMUP_STEPS,
        "trainable_parameters_per_mode": sum(p.numel() for p in initial.parameters()),
        "training_profiles": training, "validation_profiles": validation,
        "target_context": "support risk only; no query outcomes or true target parameters",
        "teacher_risk_labels": "paired historical query risk, known historical behavior; calibration supervision only",
        "learning_objective": "proper marginal risk log likelihood plus mode-specific collision log likelihood",
        "checkpoint_selection": "validation collision log loss after 10 and 50 historical-strategy risk observations",
        "gradient_norm_cap": 10, "validation_curve": curves, "best_log_loss": best,
        "elapsed_s": time.perf_counter() - start,
    })


def main():
    torch.set_num_threads(1)
    verify_or_lock()
    protocol = read_json(PREVIOUS / "results" / "protocol.json")
    RESULTS.mkdir(parents=True, exist_ok=True)
    registered = {"role": "fixed two-seed inference pilot; no blind confirmation",
                  "source_protocol": str(PREVIOUS / "results" / "protocol.json"),
                  "split": protocol["split"], "seeds": list(SEEDS),
                  "training_steps": TRAINING_STEPS, "warmup_steps": WARMUP_STEPS,
                  "physical_settings_unchanged": True,
                  "held_out_development_accessed_by_training": False}
    write_json(RESULTS / "pilot_protocol.json", registered)
    names = protocol["split"]["training"] + protocol["split"]["validation"]
    banks = load_banks(names)
    for seed in SEEDS:
        fit_seed(seed, protocol, banks)
    verify_or_lock()
    print("INFERENCE PILOT TRAINING COMPLETE", list(SEEDS), flush=True)


if __name__ == "__main__":
    main()
