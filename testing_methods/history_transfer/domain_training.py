"""Historical construction with the deployed global Boolean-feedback inference."""

import copy
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.construction import load_construction
from experiments.discovery_config import read_json, write_json
from experiments.protocol import ROOT
from experiments.scenarios import DOMAINS
from experiments.source_registration import source_contract_matches, specification_matches
from testing_methods.common.gp_classifier import ClassificationKernel, fit_target, laplace
from testing_methods.common.response import archive_fingerprints, source_weights

from .converged_response import component_probabilities
from .domain_consistency import adaptation_states, historical_prior, query_probabilities
from .joint_response import JointResponseKernel


@dataclass(frozen=True)
class Config:
    steps: int = 1500
    learning_rate: float = 0.001
    support_counts: tuple = (8, 16, 32, 64, 128)
    query_count: int = 32
    historical_context: int = 64
    validation_interval: int = 100
    seed: int = 1301
    validation_seed: int = 1319


DEFAULT = Config()
DEVICE = "cpu"


def training_contract():
    return {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in (
            "testing_methods/history_transfer/domain_training.py",
            "testing_methods/history_transfer/domain_consistency.py",
            "testing_methods/history_transfer/joint_response.py",
            "testing_methods/history_transfer/converged_response.py",
            "testing_methods/common/gp_classifier.py",
            "testing_methods/common/gp_laplace.py",
            "testing_methods/common/classification.py",
            "testing_methods/common/response.py",
            "experiments/scenarios.py",
        )
    }


def specification(history, development, held_family, config, covariance_mode):
    if covariance_mode not in ("full", "diagonal"):
        raise ValueError("Choose the admitted historical covariance representation")
    if (
        len(history) < 2
        or not development
        or any(a.role != "history" for a in history)
        or any(a.role != "development" for a in development)
        or len({a.source_id for a in (*history, *development)}) != len(history) + len(development)
    ):
        raise ValueError(
            "Consistent construction requires disjoint historical and development systems"
        )
    if held_family is not None and any(a.family == held_family for a in (*history, *development)):
        raise ValueError("Held family entered training, response context or checkpoint selection")
    if len({a.scenarios[0].kind for a in (*history, *development)}) != 1:
        raise ValueError("Construct one functional scenario and source scope at a time")
    return json.loads(
        json.dumps(
            {
                "history_fingerprints": archive_fingerprints(history),
                "development_fingerprints": archive_fingerprints(development),
                "held_family": held_family,
                "config": asdict(config),
                "covariance_mode": covariance_mode,
                "training_contract": training_contract(),
                "online_features": "float64",
                "training_device": DEVICE,
                "training_threads": 1,
                "target_feedback": "collision Boolean only",
                "historical_feedback": "collision and clearance",
            }
        )
    )


def selected_indices(archive, config, seed):
    return np.random.default_rng(seed).choice(
        len(archive.scenarios), max(config.support_counts) + config.query_count, replace=False
    )


def target_validation(model, development, config):
    rows = []
    for archive in development:
        indices = selected_indices(archive, config, config.validation_seed)
        x, labels = archive.x[indices], archive.collision[indices]
        for count in config.support_counts:
            adapted = copy.deepcopy(model)
            fit_target(adapted, x[:count], labels[:count], steps=10, learning_rate=0.003)
            with torch.no_grad():
                p, _ = component_probabilities(
                    adapted.mean.expand(1, len(x)),
                    adapted.covariance(x)[None],
                    list(range(count)),
                    labels[:count],
                )
                truth = torch.as_tensor(
                    labels[-config.query_count :], dtype=p.dtype, device=p.device
                )
                rows.append(
                    float(
                        F.binary_cross_entropy(
                            p[0, -config.query_count :].clamp(1e-8, 1 - 1e-8), truth
                        )
                    )
                )
    return float(np.mean(rows))


def save_model(path, model, part, spec, step):
    torch.save(
        {
            "format": "domain-consistent-collision",
            "part": part,
            "dimension": model.dimension,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "specification": spec,
            "step": step,
        },
        path,
    )


def load_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != "domain-consistent-collision" or not source_contract_matches(
        saved["specification"]["training_contract"], training_contract()
    ):
        raise ValueError("Consistent model construction differs from its registered source")
    model = (
        ClassificationKernel(saved["dimension"], deep=True)
        if saved["part"] == "target"
        else JointResponseKernel(saved["dimension"])
    )
    model = model.double()
    model.load_state_dict(saved["state_dict"])
    return model.eval(), saved


def train_target(history, development, destination, spec, config):
    destination = Path(destination)
    report_path = destination.with_suffix(".json")
    if destination.exists() or report_path.exists():
        report = read_json(report_path)
        if report["status"] != "complete" or not specification_matches(
            report["specification"], spec
        ):
            raise ValueError(
                "Existing target construction is incomplete or uses another source scope"
            )
        load_model(destination)
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    generator = np.random.default_rng(config.seed)
    device = DEVICE
    model = ClassificationKernel(history[0].x.shape[1], deep=True).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    started, best, curve = time.perf_counter(), float("inf"), []
    for step in range(1, config.steps + 1):
        archive = history[(step - 1) % len(history)]
        count = int(generator.choice(config.support_counts))
        indices = generator.choice(len(archive.scenarios), count, replace=False)
        optimizer.zero_grad(set_to_none=True)
        loss = laplace(model, archive.x[indices], archive.collision[indices])[0] / count
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite historical target-kernel evidence")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step % config.validation_interval == 0 or step == config.steps:
            score = target_validation(model, development, config)
            curve.append(
                {"step": step, "training_evidence": float(loss.detach()), "validation_nll": score}
            )
            if score < best:
                best = score
                save_model(destination, model, "target", spec, step)
            report = {
                "status": "complete" if step == config.steps else "running",
                "specification": spec,
                "curve": curve,
                "best_validation_nll": best,
                "training_s": time.perf_counter() - started,
            }
            write_json(report_path, report)
            print(
                "consistent_target",
                step,
                "/",
                config.steps,
                "validation_nll",
                round(score, 6),
                flush=True,
            )
    return report


def cache_episode(base, archive, config, seed):
    indices = selected_indices(archive, config, seed)
    x = archive.x[indices]
    labels = archive.collision[indices[: max(config.support_counts)]]
    return {
        "target": archive,
        "indices": indices,
        "x": x,
        "labels": labels,
        "query_labels": archive.collision[indices[-config.query_count :]],
        "states": adaptation_states(base, x, labels),
    }


def contexts_for_episode(history, episode, generator, config):
    target = episode["target"]
    excluded = {target.scenarios[i].coordinate() for i in episode["indices"]}
    contexts = []
    for archive in history:
        if archive.source_id == target.source_id:
            continue
        available = [
            i for i, scene in enumerate(archive.scenarios) if scene.coordinate() not in excluded
        ]
        indices = generator.choice(available, config.historical_context, replace=False)
        contexts.append(
            replace(
                archive,
                scenarios=[archive.scenarios[i] for i in indices],
                collision=archive.collision[indices],
                safety=archive.safety[indices],
            )
        )
    return contexts


def episode_probabilities(encoder, contexts, episode, support_counts, config, covariance_mode):
    priors = [historical_prior(encoder, archive, episode["x"]) for archive in contexts]
    means, covariances = [torch.stack(values) for values in zip(*priors)]
    weights = means.new_tensor(source_weights(contexts))
    maximum = max(support_counts)
    return query_probabilities(
        means,
        covariances,
        weights,
        episode["states"][: maximum + 1],
        episode["labels"][:maximum],
        support_counts,
        config.query_count,
        covariance_mode=covariance_mode,
    )


def encoder_validation(encoder, history, episodes, config, covariance_mode):
    rows = []
    encoder.eval()
    with torch.no_grad():
        for episode in episodes:
            probabilities = episode_probabilities(
                encoder, history, episode, config.support_counts, config, covariance_mode
            )
            for count, p in probabilities.items():
                truth = p.new_tensor(episode["query_labels"], dtype=p.dtype)
                rows.append(
                    {
                        "source_id": episode["target"].source_id,
                        "support_count": count,
                        "collision_nll": float(
                            F.binary_cross_entropy(p.clamp(1e-8, 1 - 1e-8), truth)
                        ),
                        "brier": float((p - truth).square().mean()),
                    }
                )
    encoder.train()
    return {
        metric: float(np.mean([r[metric] for r in rows])) for metric in ("collision_nll", "brier")
    }, rows


def train(history, development, destination, *, covariance_mode, held_family=None, config=DEFAULT):
    torch.set_num_threads(1)
    history, development = tuple(history), tuple(development)
    spec = specification(history, development, held_family, config, covariance_mode)
    destination = Path(destination)
    target_path, encoder_path = (
        destination / "target_kernel.pt",
        destination / "historical_encoder.pt",
    )
    target_report = train_target(history, development, target_path, spec, config)
    if encoder_path.exists() or encoder_path.with_suffix(".json").exists():
        report = read_json(encoder_path.with_suffix(".json"))
        if report["status"] != "complete" or not specification_matches(
            report["specification"], spec
        ):
            raise ValueError(
                "Existing encoder construction is incomplete or uses another allowed data scope"
            )
        load_model(encoder_path)
        return {"target": target_report, "encoder": report}
    device = DEVICE
    base, _ = load_model(target_path)
    base = base.to(device)
    started = time.perf_counter()
    training = [cache_episode(base, a, config, config.seed + i) for i, a in enumerate(history)]
    validation = [cache_episode(base, a, config, config.validation_seed) for a in development]
    cache_s = time.perf_counter() - started
    print(
        "consistent_prefix_caches",
        "history",
        len(training),
        "development",
        len(validation),
        "wall_s",
        round(cache_s, 2),
        flush=True,
    )
    torch.manual_seed(config.seed)
    encoder = JointResponseKernel(history[0].x.shape[1]).double().to(device)
    with torch.no_grad():
        parameters = dict(encoder.named_parameters())
        for name, value in base.named_parameters():
            parameters[name].copy_(value)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)
    generator = np.random.default_rng(config.seed)
    best, curve = (float("inf"), float("inf")), []
    for step in range(1, config.steps + 1):
        episode = training[(step - 1) % len(training)]
        contexts = contexts_for_episode(history, episode, generator, config)
        count = int(generator.choice(config.support_counts))
        optimizer.zero_grad(set_to_none=True)
        p = episode_probabilities(encoder, contexts, episode, (count,), config, covariance_mode)[
            count
        ]
        truth = p.new_tensor(episode["query_labels"], dtype=p.dtype)
        loss = F.binary_cross_entropy(p.clamp(1e-8, 1 - 1e-8), truth)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite matched global-feedback episode loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
        optimizer.step()
        if step % config.validation_interval == 0 or step == config.steps:
            metrics, rows = encoder_validation(
                encoder, history, validation, config, covariance_mode
            )
            score = metrics["collision_nll"], metrics["brier"]
            curve.append(
                {
                    "step": step,
                    "training_nll": float(loss.detach()),
                    "validation": metrics,
                    "episodes": rows,
                }
            )
            if score < best:
                best = score
                save_model(encoder_path, encoder, "historical_encoder", spec, step)
            report = {
                "status": "complete" if step == config.steps else "running",
                "specification": spec,
                "curve": curve,
                "best_score": list(best),
                "prefix_cache_s": cache_s,
                "training_s": time.perf_counter() - started,
            }
            write_json(encoder_path.with_suffix(".json"), report)
            print(
                "consistent_encoder",
                step,
                "/",
                config.steps,
                "validation_nll",
                round(score[0], 6),
                flush=True,
            )
    return {"target": target_report, "encoder": report}


def build_scope(kind, held_family, covariance_mode):
    history, development = load_construction(kind, held_family)
    scope = "all_sources" if held_family is None else f"without_{held_family}"
    destination = ROOT / "experiment_outputs/gcrt/collision/uniform_models" / kind / scope
    report = train(
        history, development, destination, covariance_mode=covariance_mode, held_family=held_family
    )
    return {
        "scenario": kind,
        "scope": scope,
        "destination": str(destination.relative_to(ROOT)),
        **report,
    }


def main():
    decision = read_json(
        ROOT
        / "experiment_outputs/benchmark/collision/discovery/domain_comparison/covariance/decision.json"
    )
    if decision["status"] != "complete_matched_covariance_control":
        raise ValueError(
            "Finish method finalization before constructing the final historical models"
        )
    jobs = [
        (kind, held)
        for kind in DOMAINS
        for held in (None, "rule", "optimization", "search", "learning")
    ]
    reports = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(build_scope, kind, held, decision["selected_covariance"])
            for kind, held in jobs
        ]
        for future in as_completed(futures):
            reports.append(future.result())
            write_json(
                ROOT / "experiment_outputs/gcrt/collision/uniform_models/construction.json",
                {
                    "status": "complete" if len(reports) == len(jobs) else "running",
                    "tasks": reports,
                    "covariance_mode": decision["selected_covariance"],
                    "new_physical_calls": 0,
                },
            )
    return reports


if __name__ == "__main__":
    main()
