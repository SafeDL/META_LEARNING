"""History-only neural predictions with an explicit output for each source."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from methods.history_guided_testing.config import PROFILES, ROOT as BASELINE
from methods.history_guided_testing.history import split_indices
from methods.history_guided_testing.io import write_json

from .config import OUTPUT

SOURCE_NAMES = tuple(profile.name for profile in PROFILES)


class HistoryResponseModel(nn.Module):

    def __init__(self, source_count=6):
        super().__init__()
        self.source_count = source_count
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2 * source_count),
        )

    def forward(self, x):
        output = self.layers(x)
        return output[:, :self.source_count].sigmoid(), output[:, self.
                                                               source_count:]


def response_loss(prediction, logits, risks, collisions):
    return (((prediction - risks).square() * (1 + 4 * risks)).mean() +
            0.1 * F.binary_cross_entropy_with_logits(logits, collisions))


def fit_history_model(bank, source_names, train_indices, validation_indices,
                      rng):
    x = torch.as_tensor(bank["x"][:, :4], dtype=torch.float32, device="cuda")
    risks = torch.as_tensor(
        np.asarray([bank[name] for name in source_names]).T,
        dtype=torch.float32,
        device="cuda",
    )
    collisions = torch.as_tensor(
        np.asarray([bank[name + "_collision"] for name in source_names]).T,
        dtype=torch.float32,
        device="cuda",
    )
    model = HistoryResponseModel(len(source_names)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=0.001,
                                  weight_decay=0.0001)
    best_loss = float("inf")
    curve = []
    for epoch in range(1, 401):
        model.train()
        order = rng.permutation(train_indices)
        for start in range(0, len(order), 128):
            indices = order[start:start + 128]
            prediction, logits = model(x[indices])
            loss = response_loss(prediction, logits, risks[indices],
                                 collisions[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                prediction, logits = model(x[validation_indices])
                error = float(
                    response_loss(
                        prediction,
                        logits,
                        risks[validation_indices],
                        collisions[validation_indices],
                    ))
            curve.append({"epoch": epoch, "validation_loss": error})
            if error < best_loss:
                best_loss = error
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    return model, curve


def train(seed):
    path = OUTPUT / "models" / f"emulator_{seed}.pt"
    if path.exists():
        return
    bank = np.load(BASELINE / "history/responses.npz")
    train_indices, validation_indices = split_indices(bank["x"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    states, curves = {}, {}
    for family in (0, 1):
        training = train_indices[bank["x"][train_indices, 4] == family]
        validation = validation_indices[bank["x"][validation_indices,
                                                  4] == family]
        model, curve = fit_history_model(bank, SOURCE_NAMES, training,
                                         validation, rng)
        states[str(family)] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        curves[str(family)] = curve
        best_loss = min(row["validation_loss"] for row in curve)
        print("History model",
              seed,
              family,
              "validation",
              best_loss,
              flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "states": states,
        "source_names": SOURCE_NAMES,
        "seed": seed
    }, path)
    write_json(
        OUTPUT / "models" / f"emulator_{seed}.json", {
            "curves": curves,
            "epochs": 400,
            "target_data_used": False,
            "historical_collision_heads": True,
            "risk_tail_weight": 4,
        })


def predict(x, seed):
    state = torch.load(OUTPUT / "models" / f"emulator_{seed}.pt",
                       weights_only=True)
    risks = np.zeros((len(x), 6))
    collision_scores = np.zeros((len(x), 6))
    for family in (0, 1):
        indices = np.flatnonzero(x[:, 4] == family)
        model = HistoryResponseModel().cuda().eval().requires_grad_(False)
        model.load_state_dict(state["states"][str(family)])
        with torch.no_grad():
            risk, logits = model(
                torch.as_tensor(
                    x[indices, :4],
                    dtype=torch.float32,
                    device="cuda",
                ))
        risks[indices] = risk.cpu().numpy()
        collision_scores[indices] = logits.sigmoid().cpu().numpy()
    return risks, collision_scores
