from types import SimpleNamespace

import pytest
import torch

from mvr.training.updates import _scene_embeddings


def test_inner_scene_embeddings_deduplicate_geometry_without_detaching_gradients() -> None:
    class StubModel:
        def __init__(self) -> None:
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.calls: list[str] = []

        def encode_scene(self, map_tokens, interactions):
            del interactions
            self.calls.append(map_tokens)
            return SimpleNamespace(
                global_embedding=self.scale * torch.tensor(map_tokens, dtype=torch.float32)
            )

    model = StubModel()
    rows = [
        SimpleNamespace(geometry_hash="geometry-a", map_tokens=(1.0, 2.0), interactions=()),
        SimpleNamespace(geometry_hash="geometry-a", map_tokens=(1.0, 2.0), interactions=()),
        SimpleNamespace(geometry_hash="geometry-b", map_tokens=(3.0, 4.0), interactions=()),
    ]

    embeddings = _scene_embeddings(model, rows)
    embeddings.sum().backward()

    assert model.calls == [(1.0, 2.0), (3.0, 4.0)]
    assert embeddings.shape == (3, 2)
    assert model.scale.grad.item() == pytest.approx(13.0)
