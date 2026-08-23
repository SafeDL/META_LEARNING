"""Build strict held-out posterior batches from online episodes."""
from __future__ import annotations

import torch

from ..context.outcome_schema import encode_outcome
from ..context.outcome_decoder import PosteriorTrainingBatch
from ..model import TransferableScenarioMiner
from .online_meta_test import OnlineEpisode


def posterior_batch_from_episodes(model: TransferableScenarioMiner, episodes: list[OnlineEpisode]) -> PosteriorTrainingBatch:
    if len(episodes) < 2:
        raise ValueError("posterior training requires at least one support episode and one held-out target")
    device = model.device

    def token(episode: OnlineEpisode) -> torch.Tensor:
        trajectory = episode.rollout.trajectory.to(device).unsqueeze(0)
        mask = torch.ones(trajectory.shape[:2], dtype=torch.bool, device=device)
        outcome = encode_outcome(episode.outcome).to(device).unsqueeze(0)
        return model.episode_token_builder(
            episode.scene_embedding.to(device).unsqueeze(0), episode.config.to(device).unsqueeze(0),
            episode.option_index.to(device).reshape(1), trajectory, mask, outcome,
        ).squeeze(0)

    support, target = episodes[:-1], episodes[-1]
    support_tokens = torch.stack([token(episode) for episode in support]).unsqueeze(0)
    batch = PosteriorTrainingBatch(
        support_tokens,
        torch.ones((1, len(support)), dtype=torch.bool, device=device),
        target.scene_embedding.to(device).unsqueeze(0),
        target.config.to(device).unsqueeze(0),
        target.option_index.to(device).reshape(1),
        encode_outcome(target.outcome).to(device).unsqueeze(0),
        (tuple(episode.episode_id for episode in support),),
        (target.episode_id,),
    )
    batch.validate()
    return batch
