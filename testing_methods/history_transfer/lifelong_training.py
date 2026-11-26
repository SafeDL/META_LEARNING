"""Anchored shared-model updates and leave-one-source-out acceptance."""

import copy
from dataclasses import replace

import numpy as np
import torch
from torch.nn import functional as F

from .adaptation import adapted_online
from .inference import JointPrior, historical_prior, initial_weights, joint_evidence, source_tensors
from .lifelong_archive import (
    WeightedJointPosterior, effective_size, point_mean, source_weights, weighted_prior,
)
from .lifelong_reliability import effective_weights, update_scores
from .lifelong_selection import predict_moments

STEPS = 300
LEARNING_RATE = 3e-4
ANCHOR = 1e-2
VALIDATION_SEED = 9101
VALIDATION_MARGIN = 0.002
INITIAL_COUNT = 20
CONTEXT_COUNT = 64


def anchor_penalty(models, parents):
    return ANCHOR / 2 * sum(
        (parameter - old.detach()).square().sum()
        for model, parent in zip(models, parents, strict=True)
        for parameter, old in zip(model.parameters(), parent.parameters(), strict=True)
    )


def validation_folds(archives):
    folds = []
    training_coordinates, evaluation_coordinates = set(), set()
    for archive in archives:
        coordinates = [scene.coordinate() for scene in archive.scenarios]
        prefix = INITIAL_COUNT if archive.provenance.get("completed_task") else 0
        canonical = sorted(range(prefix, len(coordinates)), key=coordinates.__getitem__)
        count = len(coordinates) // 5
        if len(coordinates) - count < INITIAL_COUNT + 1 or len(canonical) < count or count == 0:
            raise ValueError("A source is too small for the registered internal split")
        order = np.asarray(canonical)[np.random.default_rng(VALIDATION_SEED).permutation(len(canonical))]
        training = np.concatenate((np.arange(prefix), order[:-count]))
        evaluation = order[-count:]
        training_coordinates.update(coordinates[i] for i in training)
        evaluation_coordinates.update(coordinates[i] for i in evaluation)
        folds.append((training, evaluation))
    if training_coordinates & evaluation_coordinates:
        raise ValueError("Internal validation coordinates enter a source's training fold")
    return folds


def round_robin(weights, steps):
    shares = np.sqrt([effective_size(weight) for weight in weights])
    shares /= shares.sum()
    balance = np.zeros_like(shares)
    for _ in range(steps):
        balance += shares
        index = int(balance.argmax())
        balance[index] -= 1
        yield index


def gate_decision(models, previous, candidate):
    finite = all(
        parameter.isfinite().all().item() for model in models for parameter in model.state_dict().values()
    )
    return bool(finite and np.isfinite([previous, candidate]).all() and candidate <= previous - VALIDATION_MARGIN)


def _subset(archive, indices):
    provenance = dict(archive.provenance)
    if "record_ids" in provenance:
        provenance["record_ids"] = [provenance["record_ids"][i] for i in indices]
    return replace(
        archive,
        scenarios=[archive.scenarios[i] for i in indices],
        collision=archive.collision[indices],
        safety=archive.safety[indices],
        rss_margin=archive.continuous_margin[indices],
        provenance=provenance,
    )


def _joint_loss(model, archive, indices, weight, anchor):
    x, labels, margins = archive.x[indices], archive.collision[indices], archive.continuous_margin[indices]
    if anchor is None:
        return joint_evidence(model, x, labels, margins)
    state = WeightedJointPosterior(
        point_mean(model, len(indices), anchor), model.kernel(x), labels, margins, weight[indices],
    )
    return state.nll / len(indices)


def _context(archives, folds, target, indices, generator):
    prohibited = {archives[target].scenarios[i].coordinate() for i in indices}
    contexts = []
    for i, archive in enumerate(archives):
        if i == target:
            continue
        available = [j for j in folds[i][0] if archive.scenarios[j].coordinate() not in prohibited]
        selected = generator.permutation(available)[:CONTEXT_COUNT]
        if len(selected):
            contexts.append((i, selected, _subset(archive, selected)))
    if not contexts:
        raise ValueError("LOSO prediction needs at least one coordinate-disjoint source")
    return contexts


def _next_probability(models, archive, indices, contexts, weights, anchors, online):
    target, encoder = models
    x = archive.x[indices]
    labels, margins = archive.collision[indices], archive.continuous_margin[indices]
    z_encoder = encoder.features(x)
    means, blocks, used_weights = [], [], []
    for index, selected, context in contexts:
        weight, anchor = weights[index][selected], anchors[index]
        if anchor is None:
            mean, block = historical_prior(encoder, context, x, encoded_query=z_encoder)
        else:
            mean, block = weighted_prior(encoder, context, x, weight, anchor, encoded_query=z_encoder)
        means.append(mean)
        blocks.append(block)
        used_weights.append(weights[index])
    sources = {
        "means": torch.stack(means), "blocks": torch.stack(blocks),
        "names": [context.source_id for _, _, context in contexts],
        "weights": source_weights(used_weights),
    }
    reference = target.point_mean(1)[0].detach()
    z_target = target.features(x).detach()
    raw = type(target)(target.dimension, deep=False).to(reference).eval()
    with torch.no_grad():
        raw.mean.fill_(online.raw_initial_mean)
    raw.requires_grad_(False)
    z_raw = raw.features(x).detach()
    scores = initial_weights(sources, reference).log()
    tensors = source_tensors(sources, target)
    bandwidth = target.log_lengthscale.exp().clamp_min(0.01).detach()
    likelihoods, paid = [], []
    for position in range(INITIAL_COUNT + 1):
        if position:
            fitted = adapted_online(
                target, x[paid], labels[paid], margins[paid], online.learned, encoded=z_target[paid],
            )
            fitted_raw = adapted_online(
                raw, x[paid], labels[paid], margins[paid], online.raw, encoded=z_raw[paid],
            )
        else:
            fitted, fitted_raw = target, raw
        priors = [JointPrior(fitted, x, reference, item) for item in tensors]
        priors.extend((JointPrior(fitted, x, reference), JointPrior(fitted_raw, x, reference)))
        query = [position] if position < INITIAL_COUNT else list(range(position, len(x)))
        probabilities, _, _ = predict_moments(
            priors, ((fitted, z_target), (fitted_raw, z_raw)),
            paid, labels[paid], margins[paid], query, 128,
        )
        if position == INITIAL_COUNT:
            weight = effective_weights(
                scores, z_target[query], z_target[paid], likelihoods, bandwidth,
                spatial=True, batch_size=128,
            )
            return (weight * probabilities).sum(0)
        scores, likelihood = update_scores(scores, probabilities[:, 0], labels[position], tempered=True)
        likelihoods.append(likelihood)
        paid.append(position)


def _validate(models, archives, folds, weights, anchors, online):
    generator = np.random.default_rng(VALIDATION_SEED)
    losses = []
    with torch.no_grad():
        for index, archive in enumerate(archives):
            training, evaluation = folds[index]
            indices = np.concatenate((training[:INITIAL_COUNT], evaluation))
            contexts = _context(archives, folds, index, indices, generator)
            prediction = _next_probability(models, archive, indices, contexts, weights, anchors, online)
            labels = prediction.new_tensor(archive.collision[evaluation])
            losses.append(float(F.binary_cross_entropy(prediction.clamp(1e-8, 1 - 1e-8), labels)))
    return losses


def update_shared(parents, archives, weighting, online, *, device="cpu"):
    """Return new models only after internal held-out next-response NLL improves."""
    archives = tuple(archives)
    ids = [archive.source_id for archive in archives]
    if len(set(ids)) != len(ids) or len(ids) < 2 or set(weighting) - set(ids):
        raise ValueError("Anchored updates require distinct available sources")
    if any(a.role != "history" and not a.provenance.get("completed_task") for a in archives):
        raise ValueError("Only history and completed paid tasks may enter the update")
    folds = validation_folds(archives)
    weights, anchors = zip(*[
        weighting.get(archive.source_id, (np.ones(len(archive.scenarios)), None))
        for archive in archives
    ])
    frozen = tuple(copy.deepcopy(model).to(device).double().eval().requires_grad_(False) for model in parents)
    candidate = tuple(copy.deepcopy(model).requires_grad_(True) for model in frozen)
    parameters = [parameter for model in candidate for parameter in model.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    generator = np.random.default_rng(1301)
    visits = dict.fromkeys(ids, 0)
    status = "finite"
    for index in round_robin(weights, STEPS):
        archive = archives[index]
        visits[archive.source_id] += 1
        order = generator.permutation(folds[index][0])
        indices = order[:INITIAL_COUNT + 1]
        contexts = _context(archives, folds, index, indices, generator)
        optimizer.zero_grad(set_to_none=True)
        try:
            evidence = _joint_loss(candidate[0], archive, order[:CONTEXT_COUNT], weights[index], anchors[index])
            prediction = _next_probability(candidate, archive, indices, contexts, weights, anchors, online)
            truth = prediction.new_tensor(archive.collision[indices[-1:]])
            next_loss = F.binary_cross_entropy(
                prediction.clamp(1e-8, 1 - 1e-8), truth,
            ) * weights[index][indices[-1]]
            loss = evidence + next_loss + anchor_penalty(candidate, frozen)
            if not loss.isfinite():
                raise FloatingPointError("Nonfinite anchored objective")
            loss.backward()
            if not torch.nn.utils.clip_grad_norm_(parameters, 5).isfinite():
                raise FloatingPointError("Nonfinite anchored gradient")
            optimizer.step()
        except (FloatingPointError, torch.linalg.LinAlgError) as error:
            status = repr(error)
            break
    old = _validate(frozen, archives, folds, weights, anchors, online)
    new = None
    if status == "finite" and all(parameter.isfinite().all() for parameter in parameters):
        try:
            new = _validate(candidate, archives, folds, weights, anchors, online)
        except (FloatingPointError, torch.linalg.LinAlgError) as error:
            status = repr(error)
    if not np.isfinite(old).all():
        raise FloatingPointError("Nonfinite parent validation")
    if new is not None and not np.isfinite(new).all():
        status, new = "nonfinite_candidate_validation", None
    accepted = new is not None and gate_decision(candidate, np.mean(old), np.mean(new))
    report = {
        "decision": "gate_accepted" if accepted else "gate_rejected_retained_parent",
        "training_status": status, "training_visits": visits,
        "validation_old_nll": float(np.mean(old)),
        "validation_new_nll": float(np.mean(new)) if new is not None else None,
        "validation_sources": ids,
        "validation_count_by_source": [len(evaluation) for _, evaluation in folds],
        "validation_seed": VALIDATION_SEED, "training_steps": sum(visits.values()),
    }
    return tuple(model.requires_grad_(False) for model in candidate) if accepted else parents, report
