"""Historical-build episodic training; final target is never an input."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from methods.failure_memory_regression.fm2_memory import build_memory, memory_arrays
from methods.failure_memory_regression.fm2_model import FM2Model, support_row
from methods.failure_memory_regression.fm2_schema import SOURCES, TARGET, valid_label


TRAINING_SEEDS = (7319, 4179931, 20260928)
VALIDATION_BUILD = "nl3_v2"
IMPORTANT_SUPPORT = (0, 1, 3, 5, 10, 20, 30, 40)


def _episode_data(rows: list[dict], coords: np.ndarray, held_out: str):
    memory_rows = [row for row in rows if row["build_id"] != held_out]
    cards = build_memory(memory_rows, coords, exclude=held_out)
    memory = memory_arrays(cards, memory_rows)
    target = {r["scenario_id"]: r for r in rows if r["build_id"] == held_out}
    return memory, target


def _sample_support(rng: np.random.Generator, valid_indices: list[int]) -> list[int]:
    upper = min(49, len(valid_indices) - 1)
    if upper < 0:
        return []
    if rng.random() < 0.7:
        size = int(rng.choice([n for n in IMPORTANT_SUPPORT if n <= upper]))
    else:
        size = int(rng.integers(upper + 1))
    return rng.choice(valid_indices, size=size, replace=False).tolist() if size else []


def _support(model: FM2Model, coords: np.ndarray, ids: list[str], target: dict,
             positions: list[int], memory: dict) -> list[dict]:
    if not positions:
        return []
    with torch.no_grad():
        p0 = torch.sigmoid(model(coords[positions], memory, [])).cpu().numpy()
    result = []
    for position, probability in zip(positions, p0):
        row = support_row(coords[position], target[ids[position]], float(probability))
        if row is not None:
            result.append(row)
    return result


def _loss(logits: torch.Tensor, labels: torch.Tensor, rank_weight: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    positives = logits[labels == 1]
    negatives = logits[labels == 0]
    ranking = (F.softplus(-(positives[:, None] - negatives[None])).mean()
               if len(positives) and len(negatives) else logits.new_zeros(()))
    return bce + rank_weight * ranking


def train(rows: list[dict], scenarios: list[dict], coords: np.ndarray, output: Path,
          config: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; FM2 training stopped without CPU fallback")
    if any(row["build_id"] == TARGET for row in rows):
        raise ValueError("target build cannot enter historical episodic training")
    device = torch.device("cuda")
    ids = [row["scenario_id"] for row in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate IDs")
    available = {row["build_id"] for row in rows}
    if not set(SOURCES) <= available:
        raise ValueError("missing source bank for one or more declared builds")
    data = {build: _episode_data(rows, coords, build) for build in SOURCES}
    training_builds = [build for build in SOURCES if build != VALIDATION_BUILD]
    validation_memory, validation_target = data[VALIDATION_BUILD]
    # The held-out validation labels must not occur in any training memory.
    train_rows = [row for row in rows if row["build_id"] != VALIDATION_BUILD]
    training_data = {build: _episode_data(train_rows, coords, build) for build in training_builds}
    output.mkdir(parents=True, exist_ok=True)
    logs, summaries = [], []
    seeds = tuple(int(seed) for seed in config.get("training_seeds", TRAINING_SEEDS))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("training_seeds must contain distinct seeds")
    for seed in seeds:
        start_time = time.monotonic()
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = FM2Model(width=int(config["width"]), dropout=float(config["dropout"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                      weight_decay=float(config["weight_decay"]))
        best_loss, best_step, best_state = float("inf"), 0, None
        for step in range(1, int(config["training_steps"]) + 1):
            model.train()
            build = training_builds[(step - 1) % len(training_builds)]
            memory, target = training_data[build]
            valid = [i for i, sid in enumerate(ids) if sid in target and valid_label(target[sid]) is not None]
            chosen = _sample_support(rng, valid)
            query = [i for i in valid if i not in chosen]
            sample_size = min(int(config["query_batch"]), len(query))
            query = rng.choice(query, size=sample_size, replace=False).tolist()
            support = _support(model, coords, ids, target, chosen, memory)
            y = torch.as_tensor([valid_label(target[ids[i]]) for i in query],
                                dtype=torch.float32, device=device)
            logits = model(coords[query], memory, support)
            loss = _loss(logits, y, float(config["rank_weight"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % int(config["validation_interval"]) == 0 or step == int(config["training_steps"]):
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    valid_val = [i for i, sid in enumerate(ids) if sid in validation_target
                                 and valid_label(validation_target[sid]) is not None]
                    for size in (0, 5, 20, 40):
                        picked = valid_val[:min(size, len(valid_val) - 1)]
                        query_val = [i for i in valid_val if i not in picked]
                        support_val = _support(model, coords, ids, validation_target,
                                               picked, validation_memory)
                        labels = torch.as_tensor([valid_label(validation_target[ids[i]])
                                                  for i in query_val], dtype=torch.float32,
                                                 device=device)
                        pred = model(coords[query_val], validation_memory, support_val)
                        val_losses.append(float(_loss(pred, labels, float(config["rank_weight"]))))
                    validation_loss = float(np.mean(val_losses))
                logs.append({"seed": seed, "step": step, "train_loss": float(loss.detach()),
                             "validation_loss": validation_loss})
                if validation_loss < best_loss:
                    best_loss, best_step = validation_loss, step
                    best_state = {key: tensor.detach().cpu().clone()
                                  for key, tensor in model.state_dict().items()}
        assert best_state is not None
        checkpoint = output / f"checkpoint_seed_{seed}.pt"
        torch.save({"state_dict": best_state, "seed": seed, "step": best_step,
                    "validation_loss": best_loss, "config": config}, checkpoint)
        summaries.append({"seed": seed, "step": best_step, "validation_loss": best_loss,
                          "checkpoint": str(checkpoint),
                          "training_wall_seconds": round(time.monotonic() - start_time, 3)})
    best = min(summaries, key=lambda row: (row["validation_loss"], row["seed"]))
    with (output / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    (output / "validation_summary.json").write_text(
        json.dumps({"validation_build": VALIDATION_BUILD, "seed_results": summaries,
                    "selection_rule": "minimum mean held-out pseudo-target BCE+ranking loss at support sizes 0,5,20,40; tie lower seed",
                    "selected": best}, indent=2), encoding="utf-8")
    selected = torch.load(best["checkpoint"], map_location="cpu", weights_only=False)
    torch.save(selected, output / "checkpoint.pt")
    return best
