from __future__ import annotations

import numpy as np
import torch

from mvr.map.hptr_encoder import HPTRMapEncoder
from mvr.map.interaction_encoder import InteractionEncoder
from mvr.map.schema import MapPolyline, MapTokens
from mvr.scenario.interaction import InteractionCandidate


HASH = "b" * 64


def _tokens(rotation: float = 0.0) -> MapTokens:
    matrix = np.asarray([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]], dtype=np.float32)
    base = (np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.float32), np.asarray([[0, 3], [1, 3], [2, 3]], dtype=np.float32))
    polylines = tuple(MapPolyline(str(index), "lane", points @ matrix.T, np.full(3, rotation, dtype=np.float32), np.zeros(3, dtype=np.float32), 3.5, 20.0, {}) for index, points in enumerate(base))
    return MapTokens(HASH, polylines, {"left": ((0, 1),), "right": ((1, 0),)})


def test_hptr_shapes_and_se2_invariance() -> None:
    torch.manual_seed(0)
    encoder = HPTRMapEncoder(embedding_dim=16, heads=4).eval()
    local, global_embedding = encoder(_tokens())
    rotated_local, rotated_global = encoder(_tokens(np.pi / 2))
    assert local.shape == (2, 16) and global_embedding.shape == (16,)
    torch.testing.assert_close(local, rotated_local, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(global_embedding, rotated_global, atol=2e-5, rtol=2e-5)


def test_interaction_encoder_uses_geometry_not_candidate_labels() -> None:
    tokens = MapTokens(
        HASH,
        tuple(
            MapPolyline(
                str(index), "lane", points, np.zeros(3), np.zeros(3), 3.5, 20.0,
                {"lane_index": ("a", "b", index)},
            )
            for index, points in enumerate((
                np.asarray(((0, 0), (1, 0), (2, 0)), dtype=np.float32),
                np.asarray(((0, 1), (1, 1), (2, 1)), dtype=np.float32),
            ))
        ),
        {},
    )
    candidate = InteractionCandidate("opaque", (("a", "b", 0),), (("a", "b", 1),), (1.0, 0.5), 0.0, 1.0, 1.0, 0.0, 0.0, "zone")
    encoded = InteractionEncoder(8)(torch.randn(2, 8), torch.randn(8), tokens, (candidate,))
    assert encoded.global_embedding.shape == (8,)
    assert encoded.candidate_embeddings.shape == (1, 8)
