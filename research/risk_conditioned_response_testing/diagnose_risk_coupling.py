"""Measure held-profile collision information in risk prediction residuals."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from research.behavior_response_testing.model import (
    BehaviorResponseModel,
    behavior_coordinates,
    response_inputs,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "behavior_response_testing" / "results" / "confirmation"
OUTPUT = ROOT / "results" / "risk_coupling_diagnostic.json"
SEEDS = (11, 23, 37, 53, 71)
CALIBRATOR_C = 1e4


def load_predictions(seed, family, profile, models):
    behavior = behavior_coordinates(profile)
    rows = []
    labels = []
    groups = []
    for replicate in range(2):
        path = SOURCE / profile["name"] / f"pool_{replicate}" / "responses.npz"
        with np.load(path) as bank:
            selected = bank["x"][:, 4] == family
            x = bank["x"][selected]
            inputs = torch.as_tensor(
                response_inputs(x, behavior), dtype=torch.float32, device="cuda"
            )
            with torch.inference_mode():
                risk, collision_logit = models[family](inputs)
            prediction = np.column_stack(
                (risk.cpu().numpy(), collision_logit.cpu().numpy())
            )
            rows.append(
                np.column_stack(
                    (
                        prediction,
                        bank["risk"][selected],
                        bank["collision"][selected],
                    )
                )
            )
            groups.extend([profile["name"]] * len(x))
    return np.concatenate(rows), np.asarray(groups)


def fit_calibrator(features, collision):
    residual_scaler = StandardScaler().fit(features[:, 1:2])
    standardized = np.column_stack(
        (features[:, :1], residual_scaler.transform(features[:, 1:2]))
    )
    model = LogisticRegression(C=CALIBRATOR_C, max_iter=2000)
    model.fit(standardized, collision)
    return {
        "role": "Historical development fit for a risk-conditioned collision decoder",
        "regularization_C": CALIBRATOR_C,
        "risk_residual_mean": float(residual_scaler.mean_[0]),
        "risk_residual_scale": float(residual_scaler.scale_[0]),
        "intercept": float(model.intercept_[0]),
        "collision_logit_scale": float(model.coef_[0, 0]),
        "standardized_risk_residual_scale": float(model.coef_[0, 1]),
        "profiles": 48,
        "records": int(len(collision)),
    }


def holdout_scores(train_features, train_collision, test_features, test_collision):
    scaler = StandardScaler().fit(train_features[:, 1:2])
    train_augmented = np.column_stack(
        (train_features[:, :1], scaler.transform(train_features[:, 1:2]))
    )
    test_augmented = np.column_stack(
        (test_features[:, :1], scaler.transform(test_features[:, 1:2]))
    )
    model = LogisticRegression(C=CALIBRATOR_C, max_iter=2000)
    model.fit(train_augmented, train_collision)
    probability = model.predict_proba(test_augmented)[:, 1]
    baseline_logit = (
        model.intercept_[0]
        + model.coef_[0, 0] * test_augmented[:, 0]
    )
    baseline_probability = 1 / (1 + np.exp(-baseline_logit))
    return {
        "baseline_log_loss": float(log_loss(test_collision, baseline_probability)),
        "risk_residual_log_loss": float(log_loss(test_collision, probability)),
        "log_loss_improvement": float(
            log_loss(test_collision, baseline_probability)
            - log_loss(test_collision, probability)
        ),
        "baseline_auc": float(roc_auc_score(test_collision, baseline_probability)),
        "risk_residual_auc": float(roc_auc_score(test_collision, probability)),
        "auc_improvement": float(
            roc_auc_score(test_collision, probability)
            - roc_auc_score(test_collision, baseline_probability)
        ),
    }


def main():
    torch.set_num_threads(1)
    protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    profiles = protocol["profiles"]
    results = []
    calibrators = {}
    for seed in SEEDS:
        states = torch.load(
            SOURCE.parent / "models" / f"predictor_{seed}.pt",
            map_location="cuda",
            weights_only=True,
        )
        models = []
        for family in (0, 1):
            model = BehaviorResponseModel().cuda().eval()
            model.load_state_dict(states[str(family)])
            models.append(model)
        for family in (0, 1):
            profile_features = {}
            profile_collisions = {}
            for profile in profiles:
                rows, profile_groups = load_predictions(
                    seed, family, profile, models
                )
                del profile_groups
                profile_features[profile["name"]] = np.column_stack(
                    (rows[:, 1], rows[:, 2] - rows[:, 0])
                )
                profile_collisions[profile["name"]] = rows[:, 3].astype(
                    np.int64
                )
            train_profiles = [
                profile["name"]
                for profile in profiles
                if int(profile["name"].split("_")[1]) < 12
            ]
            validation_profiles = [
                profile["name"]
                for profile in profiles
                if int(profile["name"].split("_")[1]) >= 12
            ]
            train_features = np.concatenate(
                [profile_features[name] for name in train_profiles]
            )
            train_collision = np.concatenate(
                [profile_collisions[name] for name in train_profiles]
            )
            validation_features = np.concatenate(
                [profile_features[name] for name in validation_profiles]
            )
            validation_collision = np.concatenate(
                [profile_collisions[name] for name in validation_profiles]
            )
            result = holdout_scores(
                train_features,
                train_collision,
                validation_features,
                validation_collision,
            )
            result.update({"seed": seed, "family": family})
            results.append(result)
            calibrator = fit_calibrator(train_features, train_collision)
            calibrator["fit_profiles"] = train_profiles
            calibrators.setdefault(str(seed), {})[str(family)] = calibrator
    payload = {
        "role": "Development diagnostic; not a method confirmation",
        "source": "Second independent confirmation cohort, unsealed only after its full audit",
        "profiles": len(profiles),
        "profile_split": {
            "fit": "First twelve profiles of each controller family",
            "holdout": "Last twelve profiles of each controller family",
            "fit_profile_count": 24,
            "holdout_profile_count": 24,
        },
        "predictor_seeds": list(SEEDS),
        "baseline": "Logistic recalibration fit only on the first twelve profiles of each family",
        "augmentation": "Add the frozen predictor's continuous-risk residual",
        "no_target_selector_access": True,
        "calibrators": calibrators,
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("RISK COUPLING DIAGNOSTIC", OUTPUT)
    for family in (0, 1):
        subset = [row for row in results if row["family"] == family]
        print(
            "FAMILY",
            family,
            "LOGLOSS GAIN",
            np.mean([row["log_loss_improvement"] for row in subset]),
            "AUC GAIN",
            np.mean([row["auc_improvement"] for row in subset]),
        )


if __name__ == "__main__":
    main()
