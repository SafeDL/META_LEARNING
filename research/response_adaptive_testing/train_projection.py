"""Fit source-contrast mappings on a stratified development-profile split."""
import numpy as np
import torch
from torch.nn import functional as F

from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import SOURCE_NAMES, predict

from .config import OUTPUT
from .confirmation import CONFIRMATION, verify_lock
from .model import risk_coordinate
from .response_projection import projected_prior

RESULTS = OUTPUT / "response_projection"
CONTEXT_COUNTS = (0, 10, 30, 60, 100, 160)
STEPS = 1200
CHECKPOINTS = (400, 800, 1200)


def development_split(protocol):
    rng = np.random.default_rng(20261011)
    training, validation = [], []
    for controller in ("IDM", "FVDM"):
        profiles = [
            profile["name"] for profile in protocol["profiles"]
            if profile["controller"] == controller
        ]
        shuffled = rng.permutation(profiles).tolist()
        training.extend(shuffled[:8])
        validation.extend(shuffled[8:])
    return training, validation


def tasks(names, protocol):
    rows = []
    for name in names:
        for replicate in range(protocol["replicates_per_profile"]):
            path = CONFIRMATION / name / f"pool_{replicate}" / "responses.npz"
            with np.load(path) as bank:
                risks, collisions = predict(bank["x"], 11)
                rows.append((name, bank["x"].copy(), risks, collisions,
                             bank["risk"].copy(), bank["collision"].copy()))
    return rows


def episode(task, options, projection, mean_logits, rng):
    _, x, risks, collisions, truth_r, truth_c = task
    count = int(rng.choice(CONTEXT_COUNTS))
    available = np.arange(len(x))
    proposal = np.argsort(-collisions.mean(1))[:600] if rng.integers(
        2) else available
    context = rng.choice(proposal, count, replace=False)
    remaining = np.setdiff1d(available, context)
    positive, negative = remaining[
        truth_c[remaining]], remaining[~truth_c[remaining]]
    positive_count = min(96, len(positive))
    query = np.concatenate((rng.choice(positive, positive_count,
                                       replace=False),
                            rng.choice(negative,
                                       192 - positive_count,
                                       replace=False)))
    indices = np.concatenate((context, query))
    mr, mc, rr, cr, cc, _ = projected_prior(x[indices], risks[indices],
                                            collisions[indices], options,
                                            projection, mean_logits,
                                            list(range(len(SOURCE_NAMES))))
    covariance = rr[:count, :count] + options["noise"] * torch.eye(
        count, dtype=mr.dtype, device=mr.device)
    cross = cr[count:, :count]
    residual = mr.new_tensor(risk_coordinate(truth_r[context],
                                             "logit")) - mr[:count]
    mean = mc[count:] + cross @ torch.linalg.solve(covariance, residual)
    variance = (
        cc[count:] -
        (cross * torch.linalg.solve(covariance, cross.T).T).sum(1)).clamp(
            min=0)
    q = torch.special.ndtr(mean / (1 + variance).sqrt()).clamp(1e-8, 1 - 1e-8)
    if positive_count:
        score = torch.logit(q)
        ranking = F.softplus(score[positive_count:][None, :] -
                             score[:positive_count, None]).mean()
        prevalence = float(truth_c[remaining].mean())
        calibration = (-prevalence * q[:positive_count].log().mean() -
                       (1 - prevalence) *
                       (1 - q[positive_count:]).log().mean())
    else:
        ranking = q.sum() * 0
        calibration = -(1 - q).log().mean()
    identity = torch.eye(len(SOURCE_NAMES),
                         dtype=projection.dtype,
                         device=projection.device)
    regularization = 0.001 * (projection - identity).square().mean()
    return ranking + 0.1 * calibration + regularization


def main():
    torch.set_num_threads(1)
    first = read_json(CONFIRMATION / "summary.json")
    assert first["success"] is False
    protocol = first["protocol"]
    verify_lock(protocol)
    training, validation = development_split(protocol)
    write_json(
        RESULTS / "protocol.json", {
            "role":
            "Failed first confirmation is development; validation profiles are not independent confirmation",
            "training_profiles":
            training,
            "validation_profiles":
            validation,
            "source_predictor_seed":
            11,
            "contexts":
            CONTEXT_COUNTS,
            "steps":
            STEPS,
            "checkpoints":
            CHECKPOINTS,
            "teacher_labels":
            "Only training profiles; true collisions supervise offline query ranking",
            "candidate_feedback":
            "Selected continuous risk only",
            "controls": ["mean_only", "projection_and_mean"],
            "frozen_parameters":
            "Risk prior, local loading, local variances, work observation variance, source predictor"
        })
    data = tasks(training, protocol)
    for control in ("mean_only", "projection_and_mean"):
        if (RESULTS / "models" / f"{control}_{STEPS}.json").exists():
            continue
        projection = torch.nn.Parameter(
            torch.eye(len(SOURCE_NAMES), dtype=torch.float64,
                      device="cuda").repeat(2, 1, 1),
            requires_grad=control == "projection_and_mean")
        mean_logits = torch.nn.Parameter(
            torch.zeros((2, len(SOURCE_NAMES)),
                        dtype=torch.float64,
                        device="cuda"))
        parameters = [mean_logits
                      ] + ([projection] if projection.requires_grad else [])
        optimizer = torch.optim.Adam(parameters, lr=0.01)
        rng = np.random.default_rng(20261012)
        curve = []
        for step in range(STEPS):
            loss = episode(data[int(rng.integers(len(data)))],
                           protocol["candidate_options"], projection,
                           mean_logits, rng)
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10)
            optimizer.step()
            curve.append(float(loss.detach()))
            if step + 1 in CHECKPOINTS:
                write_json(
                    RESULTS / "models" / f"{control}_{step + 1}.json", {
                        "control":
                        control,
                        "projection":
                        projection.detach().cpu().tolist(),
                        "mean_logits":
                        mean_logits.detach().cpu().tolist(),
                        "options":
                        protocol["candidate_options"],
                        "curve":
                        curve,
                        "steps":
                        step + 1,
                        "source_names":
                        SOURCE_NAMES,
                        "training_profiles":
                        training,
                        "validation_profiles":
                        validation,
                        "role":
                        "Development fitting; needs new prospective confirmation"
                    })
            if (step + 1) % 100 == 0:
                print("SOURCE RESPONSE MAPPING",
                      control,
                      step + 1,
                      round(float(np.mean(curve[-100:])), 5),
                      flush=True)


if __name__ == "__main__":
    main()
