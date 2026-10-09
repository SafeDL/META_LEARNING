"""Fit the frozen development-only risk-residual collision decoder."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from research.behavior_response_testing.model import (
    BehaviorResponseModel, behavior_coordinates, response_inputs)
from research.behavior_response_testing.train import SEEDS


ROOT = Path(__file__).resolve().parent
COHORT = ROOT.parent / "behavior_response_testing" / "results" / "confirmation"
MODELS = ROOT.parent / "behavior_response_testing" / "results" / "models"
OUTPUT = ROOT / "results" / "decoder.json"
REGULARIZATION_C = 1e4


def fit_family(family, profiles, model):
    logits, residuals, collisions = [], [], []
    with torch.inference_mode():
        for profile in profiles:
            behavior = behavior_coordinates(profile)
            for replicate in range(2):
                path = (COHORT / profile["name"] / f"pool_{replicate}" /
                        "responses.npz")
                with np.load(path) as bank:
                    selected = bank["x"][:, 4] == family
                    x = bank["x"][selected]
                    inputs = torch.as_tensor(
                        response_inputs(x, behavior),
                        dtype=torch.float32,
                        device="cuda")
                    risk, collision_logit = model(inputs)
                    logits.append(collision_logit.cpu().numpy())
                    residuals.append(bank["risk"][selected] -
                                     risk.cpu().numpy())
                    collisions.append(bank["collision"][selected].astype(
                        np.int64))
    return (np.concatenate(logits), np.concatenate(residuals),
            np.concatenate(collisions))


def fit_parameters(logit, residual, collision):
    scaler = StandardScaler().fit(residual[:, None])
    standardized = scaler.transform(residual[:, None])[:, 0]
    joint = LogisticRegression(C=REGULARIZATION_C, max_iter=2000)
    joint.fit(np.column_stack((logit, standardized)), collision)
    collision_only = LogisticRegression(C=REGULARIZATION_C, max_iter=2000)
    collision_only.fit(logit[:, None], collision)
    joint_probability = joint.predict_proba(
        np.column_stack((logit, standardized)))[:, 1]
    logit_probability = collision_only.predict_proba(logit[:, None])[:, 1]
    return {
        "risk_residual_mean": float(scaler.mean_[0]),
        "risk_residual_scale": float(scaler.scale_[0]),
        "intercept": float(joint.intercept_[0]),
        "collision_logit_scale": float(joint.coef_[0, 0]),
        "standardized_risk_residual_scale": float(joint.coef_[0, 1]),
        "collision_only": {
            "intercept": float(collision_only.intercept_[0]),
            "collision_logit_scale": float(collision_only.coef_[0, 0]),
        },
        "training_log_loss": {
            "risk_conditioned": float(log_loss(collision, joint_probability)),
            "collision_only": float(log_loss(collision, logit_probability)),
        },
        "training_auc": {
            "risk_conditioned": float(roc_auc_score(collision,
                                                     joint_probability)),
            "collision_only": float(roc_auc_score(collision,
                                                   logit_probability)),
        },
        "records": int(len(collision)),
        "regularization_C": REGULARIZATION_C,
    }


def main():
    torch.set_num_threads(1)
    protocol = json.loads((COHORT / "protocol.json").read_text(
        encoding="utf-8"))
    fitted = {}
    for seed in SEEDS:
        states = torch.load(MODELS / f"predictor_{seed}.pt",
                            map_location="cuda", weights_only=True)
        fitted[str(seed)] = {}
        for family in (0, 1):
            model = BehaviorResponseModel().cuda().eval()
            model.load_state_dict(states[str(family)])
            logits, residual, collision = fit_family(
                family, protocol["profiles"], model)
            fitted[str(seed)][str(family)] = fit_parameters(
                logits, residual, collision)
            current = fitted[str(seed)][str(family)]
            print("FROZEN DECODER", seed, family,
                  "residual coefficient",
                  current["standardized_risk_residual_scale"],
                  "training AUC gain",
                  current["training_auc"]["risk_conditioned"] -
                  current["training_auc"]["collision_only"], flush=True)
    payload = {
        "role": "Pre-confirmation development fit; never reads round-three responses",
        "development_cohort": str(COHORT.relative_to(ROOT.parents[1])),
        "development_profiles": len(protocol["profiles"]),
        "predictor_seeds": list(SEEDS),
        "families": [0, 1],
        "features": ["frozen collision logit", "standardized risk residual"],
        "training_outcome": "historical simulator collision label",
        "fit": "per-seed, per-family logistic regression, L2 C=1e4",
        "uncertainty_integration": "logistic-normal moment approximation sigma(mu/sqrt(1+pi*v/8))",
        "calibrators": fitted,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("DECODER FIT COMPLETE", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
