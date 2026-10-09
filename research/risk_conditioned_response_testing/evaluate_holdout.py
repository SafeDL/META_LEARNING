"""Replay the risk-conditioned selector on already measured development pools."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.model import BehaviorResponseModel
from research.behavior_response_testing.prediction import response_tables
from research.behavior_response_testing.train import OUTPUT as MODEL_OUTPUT, SEEDS
from research.response_adaptive_testing.develop import metrics

from .risk_session import RiskConditionedTestingSession


ROOT = Path(__file__).resolve().parent
BEHAVIOR_RESULTS = ROOT.parent / "behavior_response_testing" / "results"
COHORT = BEHAVIOR_RESULTS / "confirmation"
OUTPUT = ROOT / "results" / "holdout_replay.json"


def normalized_area(collision, selected):
    total = int(np.sum(collision))
    if total == 0:
        return 0.0
    return float(np.cumsum(collision[selected]).mean() / total)


def main():
    torch.set_num_threads(1)
    protocol = read_json(COHORT / "protocol.json")
    diagnostic = read_json(ROOT / "results" / "risk_coupling_diagnostic.json")
    calibrators = diagnostic["calibrators"]
    behaviors = np.asarray(protocol["candidate"]["continuous_behaviors"],
                           dtype=np.float32)
    discrepancy = protocol["candidate"]["risk_discrepancy"]
    models_by_seed = {}
    for seed in SEEDS:
        states = torch.load(MODEL_OUTPUT / "models" / f"predictor_{seed}.pt",
                            map_location="cuda", weights_only=True)
        models = []
        for family in (0, 1):
            model = BehaviorResponseModel().cuda().eval()
            model.load_state_dict(states[str(family)])
            models.append(model)
        models_by_seed[seed] = models

    rows = []
    started = time.perf_counter()
    for profile in protocol["profiles"]:
        if int(profile["name"].split("_")[1]) < 12:
            continue
        for replicate in range(2):
            folder = COHORT / profile["name"] / f"pool_{replicate}"
            with np.load(folder / "responses.npz") as bank:
                x = bank["x"].copy()
                target_risk = bank["risk"].copy()
                target_collision = bank["collision"].copy()
            for seed in SEEDS:
                risk, logits = response_tables(
                    x, behaviors, models_by_seed[seed], return_logits=True)
                risk = risk.cpu().numpy()
                logits = logits.cpu().numpy()
                session = RiskConditionedTestingSession(
                    x, risk, logits, discrepancy[str(seed)],
                    calibrators[str(seed)], budget=200)
                selected = []
                while (index := session.next_index()) is not None:
                    session.observe(float(target_risk[index]))
                    selected.append(index)
                baseline = read_json(folder / "selection" /
                                     f"candidate_{seed}.json")
                base_indices = baseline["selected_indices"]
                area, recall = normalized_area(target_collision, selected), (
                    metrics(target_collision, selected)["recall"])
                base_area = normalized_area(target_collision, base_indices)
                base_recall = metrics(target_collision, base_indices)["recall"]
                rows.append({
                    "profile": profile["name"],
                    "replicate": replicate,
                    "seed": seed,
                    "risk_conditioned_area": area,
                    "risk_conditioned_recall": recall,
                    "candidate_area": base_area,
                    "candidate_recall": base_recall,
                    "area_delta": area - base_area,
                    "recall_delta": recall - base_recall,
                    "risk_conditioned_F200": int(np.sum(target_collision[selected])),
                    "candidate_F200": int(np.sum(target_collision[base_indices])),
                    "pool_failures": int(np.sum(target_collision)),
                    "queries": len(selected),
                })
                print("HOLDOUT REPLAY", profile["name"], replicate, seed,
                      round(area - base_area, 5),
                      round(recall - base_recall, 5), flush=True)

    profile_rows = []
    for name in sorted({row["profile"] for row in rows}):
        subset = [row for row in rows if row["profile"] == name]
        profile_rows.append({
            "profile": name,
            "area_delta": float(np.mean([row["area_delta"] for row in subset])),
            "recall_delta": float(np.mean([row["recall_delta"] for row in subset])),
            "mean_new_area": float(np.mean([row["risk_conditioned_area"] for row in subset])),
            "mean_old_area": float(np.mean([row["candidate_area"] for row in subset])),
            "mean_new_recall": float(np.mean([row["risk_conditioned_recall"] for row in subset])),
            "mean_old_recall": float(np.mean([row["candidate_recall"] for row in subset])),
            "mean_new_F200": float(np.mean([row["risk_conditioned_F200"] for row in subset])),
            "mean_old_F200": float(np.mean([row["candidate_F200"] for row in subset])),
        })
    payload = {
        "role": "Development-only replay; not a prospective confirmation",
        "cohort": "Second round's previously measured final-24-profile holdout",
        "calibration_profiles": diagnostic["profile_split"]["fit_profile_count"],
        "holdout_profiles": len(profile_rows),
        "pool_seed_runs": len(rows),
        "feedback": "Only queried continuous risk is passed to the policy step",
        "approximation": "Logistic-normal moment approximation",
        "mean_area_delta": float(np.mean([row["area_delta"] for row in rows])),
        "mean_recall_delta": float(np.mean([row["recall_delta"] for row in rows])),
        "mean_F200_delta": float(np.mean([
            row["risk_conditioned_F200"] - row["candidate_F200"]
            for row in rows])),
        "elapsed_seconds": time.perf_counter() - started,
        "profile_results": profile_rows,
        "run_results": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT, payload)
    print("HOLDOUT REPLAY COMPLETE", OUTPUT,
          "AREA DELTA", payload["mean_area_delta"],
          "RECALL DELTA", payload["mean_recall_delta"],
          "F200 DELTA", payload["mean_F200_delta"], flush=True)


if __name__ == "__main__":
    main()
