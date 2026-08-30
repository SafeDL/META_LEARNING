from types import SimpleNamespace

import pytest
import torch

from mvr.training.updates import _concrete_inputs, _episode_concrete, _scene_embeddings


def test_inner_scene_embeddings_deduplicate_geometry_without_detaching_gradients() -> None:
    class StubModel:
        def __init__(self) -> None:
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.calls: list[str] = []

        def encode_scene(self, map_tokens, interactions):
            del interactions
            self.calls.append(map_tokens)
            return SimpleNamespace(
                global_embedding=self.scale * torch.tensor(map_tokens, dtype=torch.float32),
                candidate_embeddings=torch.zeros((1, 1)),
            )

        def encode_task_structure(self, embedding, _bounds, _mask):
            return embedding

    model = StubModel()
    rows = [
        SimpleNamespace(geometry_hash="geometry-a", map_tokens=(1.0, 2.0), interactions=(), logical_domain_bounds={"x": (-1.0, 1.0)}, logical_parameter_mask=(True,)),
        SimpleNamespace(geometry_hash="geometry-a", map_tokens=(1.0, 2.0), interactions=(), logical_domain_bounds={"x": (-1.0, 1.0)}, logical_parameter_mask=(True,)),
        SimpleNamespace(geometry_hash="geometry-b", map_tokens=(3.0, 4.0), interactions=(), logical_domain_bounds={"x": (-1.0, 1.0)}, logical_parameter_mask=(True,)),
    ]

    embeddings = _scene_embeddings(model, rows)
    embeddings.sum().backward()

    assert model.calls == [(1.0, 2.0), (3.0, 4.0)]
    assert embeddings.shape == (3, 2)
    assert model.scale.grad.item() == pytest.approx(13.0)


def test_replay_rebuilds_candidate_embedding_from_current_encoder() -> None:
    class StubModel:
        def __init__(self) -> None:
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.device = torch.device("cpu")

        def encode_scene(self, _map_tokens, interactions):
            values = torch.tensor(interactions, dtype=torch.float32)
            return SimpleNamespace(
                global_embedding=self.scale * values.sum().reshape(1),
                candidate_embeddings=self.scale * values.reshape(-1, 1),
            )

        def encode_task_structure(self, embedding, _bounds, _mask):
            return embedding

        def concrete_features(self, candidate, continuous, mask):
            mask = torch.as_tensor(mask, dtype=continuous.dtype).reshape_as(continuous)
            return torch.cat((candidate, continuous * mask, mask), dim=-1)

    row = SimpleNamespace(
        geometry_hash="geometry", map_tokens=(), interactions=(2.0, 3.0),
        logical_domain_bounds={"x": (-1.0, 1.0)}, logical_parameter_mask=(True,),
        candidate_index=1, continuous=(0.5,),
    )
    model = StubModel()
    _, concrete = _concrete_inputs(model, [row])
    concrete.sum().backward()
    assert model.scale.grad is not None
    with torch.no_grad():
        model.scale.add_(1.0)
    _, rebuilt = _concrete_inputs(model, [row])
    assert rebuilt[0, 0].item() == pytest.approx(6.0)
    episode = SimpleNamespace(**row.__dict__)
    assert _episode_concrete(model, episode)[0].item() == pytest.approx(6.0)
