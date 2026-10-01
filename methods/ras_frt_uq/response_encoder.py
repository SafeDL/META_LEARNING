"""Small shared scene encoder with one response head per IDM configuration."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


class ResponseEncoder(nn.Module):
    def __init__(self, source_count: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32),
        )
        self.heads = nn.Linear(32, source_count)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.heads(self.encoder(coordinates))


def fit_response(
    x: np.ndarray, y: np.ndarray, observed: np.ndarray,
    train_indices: np.ndarray, validation_indices: np.ndarray,
    seed: int, max_epochs: int = 300,
) -> tuple[ResponseEncoder, dict]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResponseEncoder(y.shape[1]).to(device)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
    mask = torch.as_tensor(observed, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    val_tensor = torch.as_tensor(validation_indices, dtype=torch.long, device=device)
    weights = mask[train_tensor]
    val_weights = mask[val_tensor]
    training_count = weights.sum()
    validation_count = val_weights.sum()
    if training_count.item() == 0 or validation_count.item() == 0:
        raise ValueError("training and validation need observed historical responses")
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tensor[train_tensor])
        loss = (loss_fn(logits, y_tensor[train_tensor]) * weights).sum() / training_count
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_tensor[val_tensor])
            val_loss = ((loss_fn(val_logits, y_tensor[val_tensor]) * val_weights).sum() /
                        validation_count).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 35:
                break
    if best_state is None:
        raise RuntimeError("response encoder produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"validation_bce": best_loss, "best_epoch": best_epoch,
                   "trained_epochs": epoch + 1, "device": str(device)}


def predict_response(model: ResponseEncoder, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        prediction = model(torch.as_tensor(x, dtype=torch.float32,
                                           device=device)).sigmoid()
    return prediction.cpu().numpy()


def fit_full_history(x: np.ndarray, y: np.ndarray, observed: np.ndarray,
                     seed: int, epochs: int) -> ResponseEncoder:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResponseEncoder(y.shape[1]).to(device)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
    weights = torch.as_tensor(observed, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    observed_count = weights.sum()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = (loss_fn(logits, y_tensor) * weights).sum() / observed_count
        loss.backward()
        optimizer.step()
    model.eval()
    return model
