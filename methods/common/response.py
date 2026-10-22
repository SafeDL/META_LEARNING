"""Historical response archives, frozen predictions and a response-set kernel."""

from dataclasses import asdict, dataclass, field
from functools import cached_property
import hashlib
import json

import numpy as np
from scipy.special import logit
import torch
from torch import nn
from torch.nn import functional as F

from benchmark.scenarios import difficult, features


@dataclass
class ResponseArchive:
    source_id: str
    family: str
    lineage: str
    role: str
    scenarios: list
    collision: np.ndarray
    safety: np.ndarray
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.collision = np.asarray(self.collision)
        self.safety = np.asarray(self.safety, dtype=np.float64)
        if self.collision.dtype != bool or self.collision.shape != (len(self.scenarios),):
            raise ValueError("Archives require actual Boolean collision labels")
        if self.safety.shape != (len(self.scenarios), 2) or np.isnan(self.safety).any():
            raise ValueError("Archive safety columns are clearance and TTC")
        if self.role not in ("history", "development"):
            raise ValueError("Final targets cannot enter representation/model construction")
        if any(difficult(s) for s in self.scenarios):
            raise ValueError("Held difficult combinations cannot enter method archives")
        if len({s.coordinate() for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("Repeated coordinates must be stored as environment repetitions")
        features(self.scenarios)

    @cached_property
    def x(self):
        return features(self.scenarios)

    @cached_property
    def auxiliary(self):
        return np.column_stack((np.clip(self.safety[:, 0], -5, 100) / 100,
                                np.minimum(self.safety[:, 1], 10) / 10))


def archive_fingerprints(archives):
    """A changed response bank cannot silently reuse a model with identical SUT IDs."""
    fingerprints = {}
    for archive in archives:
        metadata = {"source_id": archive.source_id, "family": archive.family,
                    "lineage": archive.lineage, "role": archive.role,
                    "execution_contract": archive.provenance.get("contract"),
                    "scenarios": [asdict(s) for s in archive.scenarios]}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        digest.update(archive.collision.tobytes())
        digest.update(archive.safety.astype("<f8").tobytes())
        fingerprints[archive.source_id] = digest.hexdigest()
    return fingerprints


def source_weights(archives):
    families = [a.family for a in archives]
    return np.asarray([1 / (len(set(families)) * families.count(f)) for f in families])


class HistoricalPredictor:
    def __init__(self, archives, bandwidth=0.25):
        if not archives or any(a.role != "history" for a in archives):
            raise ValueError("Historical prediction accepts historical sources only")
        self.archives = tuple(archives)
        self.bandwidth = bandwidth

    def predict(self, x):
        predictions = []
        for archive in self.archives:
            distance = ((x[:, None] - archive.x[None])**2).sum(-1)
            weight = np.exp(-distance / (2 * self.bandwidth**2))
            predictions.append((weight @ archive.collision + 0.5) / (weight.sum(1) + 1))
        return np.asarray(predictions)

    def latent_means(self, x):
        return logit(np.clip(self.predict(x), 1e-6, 1 - 1e-6))


class ResponseEncoder(nn.Module):
    def __init__(self, dimension, width=48, embedding=24):
        super().__init__()
        self.dimension = dimension
        self.context_input = nn.Linear(dimension + 3, width)
        self.query_input = nn.Linear(dimension, width)
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0,
                                       activation="gelu", batch_first=True, norm_first=True),
            1, enable_nested_tensor=False)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, embedding))
        self.auxiliary_head = nn.Linear(embedding + dimension, 2)
        self.log_lengthscale = nn.Parameter(torch.full((dimension,), -1.2))
        self.log_physical_scale = nn.Parameter(torch.tensor(0.0))
        self.log_response_scale = nn.Parameter(torch.tensor(-0.7))

    def tensor(self, value):
        parameter = self.log_lengthscale
        return torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)

    def physical_kernel(self, x):
        x = self.tensor(x)
        distance = torch.cdist(x / self.log_lengthscale.exp().clamp_min(0.03),
                               x / self.log_lengthscale.exp().clamp_min(0.03))**2
        return self.log_physical_scale.exp().square() * (-distance / 2).exp()

    def encode(self, x, contexts):
        if not contexts or any(a.role != "history" for a in contexts):
            raise ValueError("Response context must contain historical sources only")
        x = self.tensor(x)
        query = self.query_input(x)
        grouped = {}
        for position, archive in enumerate(contexts):
            # Fixed coordinate-order subsampling, independent of collision labels.
            indices = np.linspace(0, len(archive.scenarios) - 1,
                                  min(64, len(archive.scenarios)), dtype=int)
            tokens = np.column_stack((archive.x[indices], archive.collision[indices],
                                      archive.auxiliary[indices]))
            grouped.setdefault(len(tokens), []).append((position, tokens))
        output = [None] * len(contexts)
        for group in grouped.values():
            # Equal-length sources share a batch dimension; attention never crosses sources.
            tokens = self.tensor(np.stack([tokens for _, tokens in group]))
            context = self.context_encoder(self.context_input(tokens))
            queries = query[None].expand(len(group), -1, -1)
            attended, _ = self.attention(queries, context, context, need_weights=False)
            encoded = F.normalize(self.output(queries + attended), dim=-1)
            for index, (position, _) in enumerate(group):
                output[position] = encoded[index]
        return torch.stack(output)

    def covariance(self, x, contexts, *, response=True):
        physical = self.physical_kernel(x).double()
        if not response:
            return physical
        encoded = self.encode(x, contexts)
        weight = self.tensor(source_weights(contexts))
        gram = torch.einsum("s,sid,sjd->ij", weight, encoded, encoded).double()
        return physical + self.log_response_scale.exp().double().square() * gram

    def auxiliary_prediction(self, x, contexts):
        encoded = self.encode(x, contexts)
        weights = self.tensor(source_weights(contexts))
        pooled = torch.einsum("s,sid->id", weights, encoded)
        return self.auxiliary_head(torch.cat((self.tensor(x), pooled), -1))
