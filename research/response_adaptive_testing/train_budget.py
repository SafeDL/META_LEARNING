"""Learn response priors from test-policy contexts and discovery utility."""
import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import SOURCE_NAMES, predict

from .budget_training import (condition_parts, conditional_scores,
                              ranking_loss, response_parts)
from .config import BUDGET, OUTPUT
from .confirmation import CONFIRMATION, verify_lock
from .response_projection import apply_projection
from .session import AdaptiveTestingSession

RESULTS = OUTPUT / "budget_training"
CONTEXT_COUNTS = (0, 10, 30, 60, 100, 160)
SOURCE_SEEDS = (11, 37)
STEPS = 800
CHECKPOINTS = (200, 400, 800)
CONTROLS = ("replay_uniform_mean", "replay_budget_mean",
            "replay_budget_loading")


def replay_episodes(protocol, initial):
    episodes = []
    for name in protocol["training_profiles"]:
        for replicate in range(protocol["replicates_per_profile"]):
            path = CONFIRMATION / name / f"pool_{replicate}" / "responses.npz"
            with np.load(path) as bank:
                truth = torch.as_tensor(bank["collision"].copy(),
                                        dtype=torch.bool,
                                        device="cuda")
                for seed in SOURCE_SEEDS:
                    cache = RESULTS / "contexts" / name / f"pool_{replicate}" / f"source_{seed}.npz"
                    if not cache.exists():
                        prediction = predict(bank["x"], seed)
                        parts = response_parts(bank["x"], *prediction,
                                               initial["options"])
                        session = AdaptiveTestingSession(
                            bank["x"], *prediction, initial["options"])
                        apply_projection(session, bank["x"], prediction,
                                         initial)
                        arrays = {
                            key: parts[key].cpu().numpy()
                            for key in ("family", "features",
                                        "collision_values")
                        }
                        selected, observations = [], []
                        for count in range(max(CONTEXT_COUNTS) + 1):
                            if count in CONTEXT_COUNTS:
                                moments = condition_parts(
                                    parts, selected, observations,
                                    initial["options"])
                                for key, value in moments.items():
                                    arrays[f"{key}_{count}"] = value.cpu(
                                    ).numpy()
                                arrays[f"context_{count}"] = np.array(
                                    selected, dtype=np.int64)
                            if count == max(CONTEXT_COUNTS):
                                break
                            index = session.next_index()
                            risk = float(bank["risk"][index])
                            session.observe(risk)
                            selected.append(index)
                            observations.append(risk)
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(cache, **arrays)
                    with np.load(cache) as saved:
                        shared = {
                            key: torch.as_tensor(saved[key].copy(),
                                                 device="cuda")
                            for key in ("family", "features",
                                        "collision_values")
                        }
                        for count in CONTEXT_COUNTS:
                            context = saved[f"context_{count}"]
                            assert len(context) == len(set(context)) == count
                            remaining = torch.ones(len(truth),
                                                   dtype=torch.bool,
                                                   device="cuda")
                            remaining[context] = False
                            episode = {
                                key: value[remaining]
                                for key, value in shared.items()
                            }
                            for key in ("source_mean", "local_mean",
                                        "source_variance", "local_variance",
                                        "source_local_covariance"):
                                episode[key] = torch.as_tensor(
                                    saved[f"{key}_{count}"].copy(),
                                    device="cuda")[remaining]
                            episode["truth"] = truth[remaining]
                            episode["remaining_budget"] = BUDGET - count
                            episodes.append(episode)
                    print("POLICY CONTEXTS",
                          name,
                          replicate,
                          seed,
                          len(episodes),
                          flush=True)
    return episodes


def main():
    torch.set_num_threads(1)
    first = read_json(CONFIRMATION / "summary.json")
    assert first["success"] is False
    verify_lock(first["protocol"])
    split = read_json(OUTPUT / "response_projection" / "protocol.json")
    initial = read_json(OUTPUT / "response_projection" / "models" /
                        "projection_and_mean_400.json")
    initial["projection"] = np.eye(len(SOURCE_NAMES))[None].repeat(2,
                                                                   0).tolist()
    protocol = {
        "role":
        "Development after unsuccessful first confirmation; requires fresh prospective confirmation",
        "training_profiles": split["training_profiles"],
        "validation_profiles": split["validation_profiles"],
        "replicates_per_profile": first["protocol"]["replicates_per_profile"],
        "source_seeds": SOURCE_SEEDS,
        "context_counts": CONTEXT_COUNTS,
        "context_policy":
        "Frozen learned-mean risk-only greedy posterior; no target collision input",
        "initial_state": initial,
        "steps": STEPS,
        "checkpoints": CHECKPOINTS,
        "controls": CONTROLS,
        "objective":
        "Pairwise ranking plus 0.1 population binary log loss; budget controls weight pairs by the absolute change in equal endpoint/discovery-area utility",
        "candidate_feedback": "Queried continuous risk only",
        "teacher_labels": "Full collisions on sixteen training profiles only",
        "frozen_parameters":
        "Risk prior, source projection I, independent collision variance, work observation variance, mean offsets and source predictors",
        "learning_rate": 0.003,
        "episode_seed": 20261013
    }
    if (RESULTS / "protocol.json").exists():
        stored = read_json(RESULTS / "protocol.json")
        assert stored["initial_state"] == initial
        assert stored["training_profiles"] == protocol["training_profiles"]
    write_json(RESULTS / "protocol.json", protocol)
    episodes = replay_episodes(protocol, initial)
    for control in CONTROLS:
        if (RESULTS / "models" / f"{control}_{STEPS}.json").exists():
            continue
        mean_logits = torch.nn.Parameter(
            torch.tensor(initial["mean_logits"],
                         dtype=torch.float64,
                         device="cuda"))
        loading = torch.nn.Parameter(
            torch.tensor(initial["options"]["loading"],
                         dtype=torch.float64,
                         device="cuda"),
            requires_grad=control == "replay_budget_loading")
        parameters = [mean_logits
                      ] + ([loading] if loading.requires_grad else [])
        optimizer = torch.optim.Adam(parameters, lr=protocol["learning_rate"])
        rng = np.random.default_rng(protocol["episode_seed"])
        curve = []
        reference_loading = loading.detach().clone()
        for step in range(STEPS):
            episode = episodes[int(rng.integers(len(episodes)))]
            score = conditional_scores(episode, mean_logits, loading,
                                       initial["options"])
            loss = ranking_loss(score,
                                episode["truth"],
                                episode["remaining_budget"],
                                weighted=control != "replay_uniform_mean")
            loss += 0.001 * (loading - reference_loading).square().mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10)
            optimizer.step()
            curve.append(float(loss.detach()))
            if step + 1 in CHECKPOINTS:
                options = {
                    **initial["options"], "loading":
                    loading.detach().cpu().tolist()
                }
                write_json(
                    RESULTS / "models" / f"{control}_{step + 1}.json", {
                        "control": control,
                        "steps": step + 1,
                        "options": options,
                        "mean_logits": mean_logits.detach().cpu().tolist(),
                        "projection": initial["projection"],
                        "curve": curve,
                        "training_profiles": protocol["training_profiles"],
                        "validation_profiles": protocol["validation_profiles"],
                        "role":
                        "Development fitting; not independently confirmed"
                    })
            if (step + 1) % 100 == 0:
                print("BUDGET TRAINING",
                      control,
                      step + 1,
                      round(float(np.mean(curve[-100:])), 5),
                      flush=True)


if __name__ == "__main__":
    main()
