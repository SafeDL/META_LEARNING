"""Frozen historical features and reusable candidate-pool predictions."""

import numpy as np
import torch

from benchmark.config import SUT_IDS
from benchmark.data import bank_path
from benchmark.initialization import initial_indices
from benchmark.timing import now
from methods.baselines.dart.encoder import fixed_context
from methods.baselines.dart.train import load_encoder
from methods.baselines.hem.experts import HistoricalExperts
from methods.baselines.hgt.config import PROFILES
from methods.baselines.hgt.history import split_indices, subset
from methods.baselines.hgt.io import write_json
from methods.baselines.hgt.train import load_kernel

from .prior import build_prior

SEED = 11
INITIAL = 10


def assemble(x, sources, seed, *, workspace=None):
    sources = {
        profile.name: sources[profile.name]
        for profile in PROFILES
        if profile.name in sources
    }
    prediction = HistoricalExperts(sources).predict(x)
    with torch.no_grad():
        base = (
            load_kernel(
                seed,
                "global_feedback",
                **({"root": workspace.kernel_root} if workspace else {}),
            )
            .covariance(x)
            .numpy()
        )
        encoder = load_encoder(
            seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
            **({"root": workspace.encoder_root} if workspace else {}),
        )
        enhanced = (
            encoder.covariance(x, fixed_context(sources, seed), base).cpu().numpy()
        )
    common = {
        "x": x,
        "covariance": base,
        "priors": prediction.mean,
        "probability": prediction.group_weights,
        "fixed_prior": prediction.fixed_mean(),
        "seed": seed,
    }
    return common, build_prior(x, sources, enhanced)


def prepare_pool(workspace, pool):
    path = workspace.root / "pool_cache" / f"pool_{pool}.pt"
    if path.exists():
        return torch.load(path, weights_only=False, map_location="cpu")
    with np.load(bank_path(pool, SUT_IDS[0])) as bank:
        x = bank["x"][bank["x"][:, 4] == workspace.family].copy()
    train, _ = split_indices(next(iter(workspace.history.values()))["x"])
    sources = subset(workspace.history, train)
    started = now()
    prediction = HistoricalExperts(sources).predict(x)
    with torch.no_grad():
        base = (
            load_kernel(SEED, "global_feedback", root=workspace.kernel_root)
            .covariance(x)
            .numpy()
        )
    common = {
        "x": x,
        "covariance": base,
        "priors": prediction.mean,
        "probability": prediction.group_weights,
        "fixed_prior": prediction.fixed_mean(),
        "seed": SEED,
        "initial_indices": initial_indices(len(x), INITIAL, SEED),
    }
    base_s = now() - started
    started = now()
    with torch.no_grad():
        encoder = load_encoder(
            SEED,
            device="cuda" if torch.cuda.is_available() else "cpu",
            root=workspace.encoder_root,
        )
        enhanced = (
            encoder.covariance(x, fixed_context(sources, SEED), base).cpu().numpy()
        )
    encoder_s = now() - started
    started = now()
    prior = build_prior(x, sources, enhanced)
    prior_s = now() - started
    cache = {
        "common": common,
        "prior": prior,
        "preparation_s": {"base": base_s, "encoder": encoder_s, "archive": prior_s},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    write_json(path.with_suffix(".json"), cache["preparation_s"])
    return cache
