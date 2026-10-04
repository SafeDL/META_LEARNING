"""RAS-FRT-UQ on the shared history and two-family testing protocol."""
import time

import numpy as np
import torch
from torch.nn import functional as F

from methods.history_guided_testing.config import BUDGET, PROFILES, ROOT, SEEDS
from methods.history_guided_testing.history import split_indices
from methods.history_guided_testing.io import write_json
from .coverage_selector import similarities
from .fusion_selector import select_fusion_sequence
from .response_encoder import ResponseEncoder, predict_response
from .transfer_uncertainty import physical_kernel


SOURCES = tuple(profile.name for profile in PROFILES)
MODEL_ROOT = ROOT / "models/ras_frt_uq"
TRAINING = {"epochs": 300, "batch_size": 128, "lr": .001,
            "validation_interval": 10}
# Frozen on the original historical S01 experiment, before this target pool.
SIMILARITY = {"sigma_x": .3, "sigma_r": .25, "regularizer": .1}


def train_seed(bank, seed):
    destination = MODEL_ROOT / f"seed_{seed}.pt"
    if destination.exists():
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train, validation = split_indices(bank["x"])
    labels = np.column_stack([bank[source + "_collision"] for source in SOURCES])
    states, records = {}, {}
    started = time.perf_counter()
    for family in (0, 1):
        training = train[bank["x"][train, 4] == family]
        held_out = validation[bank["x"][validation, 4] == family]
        x_train = torch.as_tensor(bank["x"][training, :4], dtype=torch.float32, device="cuda")
        y_train = torch.as_tensor(labels[training], dtype=torch.float32, device="cuda")
        x_valid = torch.as_tensor(bank["x"][held_out, :4], dtype=torch.float32, device="cuda")
        y_valid = torch.as_tensor(labels[held_out], dtype=torch.float32, device="cuda")
        model = ResponseEncoder(len(SOURCES)).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["lr"])
        best, selected_epoch, curve = np.inf, None, []
        for epoch in range(1, TRAINING["epochs"] + 1):
            model.train()
            for batch in torch.randperm(len(training), device="cuda").split(TRAINING["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(x_train[batch]), y_train[batch])
                loss.backward()
                optimizer.step()
            if epoch % TRAINING["validation_interval"] == 0:
                model.eval()
                with torch.no_grad():
                    error = float(F.binary_cross_entropy_with_logits(model(x_valid), y_valid))
                curve.append({"epoch": epoch, "validation_bce": error})
                if error < best:
                    best, selected_epoch = error, epoch
                    states[str(family)] = {name: value.detach().cpu().clone()
                                           for name, value in model.state_dict().items()}
            if epoch % 100 == 0:
                print("RAS training:", seed, "family", family, "epoch", epoch,
                      "best validation BCE", round(best, 5), flush=True)
        records[str(family)] = {"train_count": len(training), "validation_count": len(held_out),
                                "selected_epoch": selected_epoch, "validation_bce": best,
                                "curve": curve}
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({"encoders": states, "sources": SOURCES, "seed": seed}, destination)
    write_json(MODEL_ROOT / f"training_{seed}.json", {
        "training": TRAINING, "families": records, "elapsed_s": time.perf_counter() - started,
        "historical_labels": "binary collision", "target_labels_used": False,
        "checkpoint_selection": "minimum historical validation BCE per family",
    })


def train():
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Run in the metadrive Conda environment with CUDA")
    with np.load(ROOT / "history/responses.npz") as bank:
        for seed in SEEDS:
            train_seed(bank, seed)


def family_similarities(x, responses):
    same_family = x[:, 4, None] == x[None, :, 4]
    similarity = similarities(x[:, :4], responses, SIMILARITY["sigma_x"], SIMILARITY["sigma_r"])
    kernel = physical_kernel(x[:, :4])
    return similarity * same_family, kernel * same_family


def select_from_responses(x, cells, responses, oracle, seed, budget=BUDGET):
    prior = responses.mean(axis=1)
    similarity, kernel = family_similarities(x, responses)
    records = []

    class BinaryOracle:
        def query(self, index):
            observation = oracle.query(index)
            if observation.collision is None:
                raise ValueError("RAS-FRT-UQ requires the queried binary collision outcome")
            records.append({"query_number": observation.query_number, "index": index,
                            "scenario_id": observation.scenario_id,
                            "collision": bool(observation.collision)})
            return float(observation.collision)

    selected, _, scores = select_fusion_sequence(
        prior, similarity, kernel, cells, BinaryOracle(), budget, SIMILARITY["regularizer"])
    return {"selected_indices": selected, "queries": records, "final_risk_mean": scores.tolist(),
            "method": "RAS-FRT-UQ", "seed": seed, "feedback_used": "binary collision",
            "sources": SOURCES, "similarity": SIMILARITY,
            "adaptation": "two four-input encoders; six shared sources; family-block similarities",
            "original_fusion_weights": True}


def select(x, cells, oracle, seed, budget=BUDGET):
    state = torch.load(MODEL_ROOT / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    responses = np.empty((len(x), len(SOURCES)))
    for family in (0, 1):
        indices = np.flatnonzero(x[:, 4] == family)
        model = ResponseEncoder(len(SOURCES)).to(device).eval().requires_grad_(False)
        model.load_state_dict(state["encoders"][str(family)])
        responses[indices] = predict_response(model, x[indices, :4])
    return select_from_responses(x, cells, responses, oracle, seed, budget)


if __name__ == "__main__":
    train()
