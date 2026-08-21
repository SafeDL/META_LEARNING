from __future__ import annotations

import numpy as np
import torch

from meta_testing.map.cache import MapCache
from meta_testing.map.hptr_encoder import HPTRMapEncoder
from meta_testing.map.schema import MapPolyline, MapTokens


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


def test_raw_and_frozen_embedding_cache(tmp_path) -> None:
    cache, tokens = MapCache(tmp_path), _tokens()
    cache.save_raw(tokens)
    assert cache.load_raw(HASH).map_hash == HASH
    local, global_embedding = HPTRMapEncoder(embedding_dim=16, heads=4)(tokens)
    cache.save_embeddings(HASH, local, global_embedding, encoder_frozen=True)
    loaded = cache.load_embeddings(HASH, encoder_frozen=True)
    assert loaded is not None
    assert cache.load_embeddings(HASH, encoder_frozen=False) is None
