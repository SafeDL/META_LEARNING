"""A-only tail readout of the frozen TNP mean and local historical response."""
import numpy as np
from scipy.special import expit
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.nn import functional as F

from .common import write_json
from .data import risk_logit
from .s01 import ROOT, configuration
from .train import setup, split_indices, subset

STEPS = 1000


def local_mean(context, x):
    distances = cdist(x, context.x)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :configuration()["baselines"]["knn_neighbors"]]
    r = np.sqrt(5.) * np.take_along_axis(distances, nearest, axis=1) / configuration()["residual_gp"]["lengthscale_local"]
    weights = (1 + r + r * r / 3) * np.exp(-r)
    risk = (weights * expit(context.z[nearest, 0])).sum(1) / weights.sum(1)
    return risk_logit(risk)[:, None]


def tail_loss(prediction, target):
    p, z = prediction.reshape(-1), target.reshape(-1)
    pairs = (z[:, None] > 0) & (z[:, None] - z[None, :] > .25)
    return F.softplus(-(p[:, None] - p[None, :]))[pairs].mean()


def fit_readout(model, context, seed):
    path = ROOT / "model/mean_readout" / f"seed_{seed}.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        return np.asarray(state["coefficients"]), state["training"]
    setup(seed)
    train, validation = split_indices()
    features = np.zeros((len(context.x), 2))
    for query in np.array_split(train, 4):
        allowed = subset(context, np.setdiff1d(train, query))
        neural = model.predict([allowed], context.x[query]).m
        features[query] = np.c_[neural[:, 0], local_mean(allowed, context.x[query])[:, 0]]
    allowed = subset(context, train)
    neural = model.predict([allowed], context.x[validation]).m
    features[validation] = np.c_[neural[:, 0], local_mean(allowed, context.x[validation])[:, 0]]
    z = context.z[:, 0]
    labels = z[validation] > 0
    original_ap = float(average_precision_score(labels, features[validation, 0]))
    local_ap = float(average_precision_score(labels, features[validation, 1]))
    head = nn.Linear(2, 1).cuda()
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1., 0.]], device="cuda"))
        head.bias.zero_()
    optimizer = torch.optim.Adam(head.parameters(), lr=.01)
    tx = torch.as_tensor(features, device="cuda", dtype=torch.float32)
    tz = torch.as_tensor(context.z, device="cuda", dtype=torch.float32)
    high, ordinary = train[z[train] > 0], train[z[train] <= 0]
    rng = np.random.default_rng(seed)
    best, best_step, coefficients = original_ap, 0, [1., 0., 0.]
    curve = []
    for step in range(1, STEPS + 1):
        ids = np.r_[rng.choice(high, 64), rng.choice(ordinary, 64)]
        prediction = head(tx[ids])
        errors = (prediction - tz[ids]).square().reshape(2, -1).mean(1)
        fraction = len(high) / len(train)
        reconstruction = fraction * errors[0] + (1 - fraction) * errors[1]
        loss = .25 * reconstruction + tail_loss(prediction, tz[ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            with torch.no_grad():
                scores = head(tx[validation]).cpu().numpy()[:, 0]
            ap = float(average_precision_score(labels, scores))
            curve.append({"step": step, "tail_AP": ap})
            if ap > best:
                best, best_step = ap, step
                coefficients = [*head.weight.detach().cpu().numpy()[0].tolist(), float(head.bias.detach().cpu()[0])]
    training = {"steps": STEPS, "chosen_step": best_step, "coefficients": coefficients,
                "original_TNP_tail_AP": original_ap, "local_tail_AP": local_ap, "readout_tail_AP": best,
                "A_train": len(train), "A_validation": len(validation), "A_cross_fit_folds": 4,
                "D_used": False, "backbone_h_kernel_gradient_steps": 0, "curve": curve}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"coefficients": coefficients, "training": training}, path)
    write_json(path.with_suffix(".json"), training)
    print("Correlation readout", seed, training, flush=True)
    return np.asarray(coefficients), training
