"""Historical joint evidence and next-response training with paid-prefix replay."""

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.costs import PeakMemory, synchronized_time
from experiments.protocol import ROOT
from experiments.rss import measurement_contract
from experiments.scenarios import TEST_SCENARIOS
from experiments.source_registration import source_contract_matches, specification_matches
from experiments.storage import read_json, write_json
from testing_methods.common.response import archive_fingerprints, shared_domain, source_weights

from .inference import adapted_model, historical_prior, joint_evidence, sequential_probability
from .model import JointResponseKernel


@dataclass(frozen=True)
class Config:
    steps: int = 1500
    learning_rate: float = 0.001
    initial_count: int = 20
    budget: int = 200
    historical_context: int = 64
    diagnostic_interval: int = 100
    seed: int = 1301
    device: str = "cpu"
    historical_rss: bool = True

    def __post_init__(self):
        if self.steps < 1 or self.learning_rate <= 0 or not 0 < self.initial_count < self.budget:
            raise ValueError("Training needs positive steps/rate and initialization below budget")
        if self.historical_context < 1 or self.diagnostic_interval < 1:
            raise ValueError("Training context and diagnostic interval must be positive")
        if torch.device(self.device).type not in ("cpu", "cuda"):
            raise ValueError("Training supports CPU or CUDA")


DEFAULT = Config()
MODEL_ROOT = ROOT / "artifacts/runs/joint_response/models"
COVARIANCE_MODE = "point_joint_blocks"


def model_directory(kind, held_family=None):
    scope = "all_sources" if held_family is None else f"without_{held_family}"
    return MODEL_ROOT / kind / scope


def training_contract(kernel_type=JointResponseKernel):
    contract = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in (
            "testing_methods/history_transfer/training.py",
            "testing_methods/history_transfer/inference.py",
            "testing_methods/history_transfer/model.py",
            "testing_methods/history_transfer/selection.py",
            "testing_methods/common/gp_classifier.py",
            "testing_methods/common/gp_laplace.py",
            "testing_methods/common/classification.py",
            "testing_methods/common/response.py",
            "experiments/scenarios.py",
            "experiments/rss.py",
            "experiments/history.py",
            "experiments/schema.py",
            "experiments/query.py",
            "experiments/costs.py",
        )
    }
    if kernel_type is not JointResponseKernel:
        path = kernel_type.source_path
        contract[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    return contract


def specification(history, held_family, config, kernel_type=JointResponseKernel):
    if (
        len(history) < 2
        or any(
            a.role != "history" and not (a.role == "target" and a.provenance.get("completed_task"))
            for a in history
        )
        or len({a.source_id for a in history}) != len(history)
    ):
        raise ValueError("Construction requires distinct historical or admitted completed systems")
    if held_family is not None and any(a.family == held_family for a in history):
        raise ValueError("Held family entered historical construction")
    if len({a.scenarios[0].kind for a in history}) != 1:
        raise ValueError("Construct one functional scenario at a time")
    if not any(len(a.scenarios) >= 2 for a in history):
        raise ValueError("Supervised training needs a task with at least two executed records")
    for archive in history:
        archive.continuous_margin
    domain = shared_domain(history)
    domain_record = {} if domain is None else {"parameter_definition": domain.to_dict()}
    return json.loads(
        json.dumps(
            {
                **domain_record,
                "scenario": history[0].scenarios[0].kind,
                "history_fingerprints": archive_fingerprints(history),
                "held_family": held_family,
                "config": asdict(config),
                "covariance_mode": COVARIANCE_MODE,
                "training_contract": training_contract(kernel_type),
                "online_features": "float64",
                "training_device": config.device,
                "training_threads": 1,
                "target_feedback": "paid collision and RSS",
                "historical_feedback": "collision and ego-normalized RSS margin"
                if config.historical_rss
                else "collision only",
                "continuous_response": measurement_contract(),
                "supervision": "uniform next-response time after initialization; all preceding feedback",
                "checkpoint": "final fixed step, no development data",
            }
        )
    )


def episode_layout(archive, config):
    length = min(len(archive.scenarios), config.budget)
    if length < 2:
        raise ValueError("Supervised episodes require at least two executed records")
    return min(config.initial_count, length - 1), length


def selected_indices(archive, config, seed, segment=0):
    """Cycle full-length windows over one fixed historical permutation."""
    _, length = episode_layout(archive, config)
    if archive.provenance.get("completed_task"):
        return np.arange(length)
    order = np.random.default_rng(seed).permutation(len(archive.scenarios))
    return np.take(order, np.arange(segment * length, (segment + 1) * length), mode="wrap")


def scheduled_item(items, new_source_ids, step):
    new = [a for a in items if a.source_id in new_source_ids]
    old = [a for a in items if a.source_id not in new_source_ids]
    if new and old:
        group = new if step % 2 else old
        return group[((step - 1) // 2) % len(group)]
    return items[(step - 1) % len(items)]


def validate_model(model):
    values = [*model.state_dict().values(), model.point_covariance(1), model.point_mean(1)]
    if any(not torch.isfinite(value).all() for value in values):
        raise FloatingPointError("Shared model contains nonfinite parameters or joint moments")


def optimize(model, optimizer, loss):
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    if not torch.isfinite(norm):
        raise FloatingPointError("Shared model training produced nonfinite gradients")
    optimizer.step()


def save_model(path, model, part, spec, step):
    validate_model(model)
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format": "sequential-joint-collision-rss",
            "part": part,
            "dimension": model.dimension,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "specification": spec,
            "step": step,
        },
        temporary,
    )
    temporary.replace(path)


def load_model(path, *, device="cpu", kernel_type=JointResponseKernel):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    spec = saved["specification"]
    if (
        saved["format"] != "sequential-joint-collision-rss"
        or spec.get("continuous_response") != measurement_contract()
        or not source_contract_matches(spec["training_contract"], training_contract(kernel_type))
        or saved["step"] != spec["config"]["steps"]
    ):
        raise ValueError("Sequential model construction differs from its registered source")
    model = kernel_type(saved["dimension"]).double().to(device)
    model.load_state_dict(saved["state_dict"])
    model.construction_specification = spec
    validate_model(model)
    return model.eval(), saved


def completed_report(path, spec, kernel_type=JointResponseKernel):
    report_path = path.with_suffix(".json")
    if not path.exists() and not report_path.exists():
        return None
    report = read_json(report_path)
    if report["status"] != "complete" or not specification_matches(report["specification"], spec):
        raise ValueError("Existing construction is incomplete or uses another source scope")
    load_model(path, kernel_type=kernel_type)
    return report


def train_target(
    history,
    destination,
    spec,
    config,
    initial=None,
    new_source_ids=(),
    kernel_type=JointResponseKernel,
):
    items = [a for a in history if len(a.scenarios) >= 2]
    destination = Path(destination)
    previous = completed_report(destination, spec, kernel_type)
    if previous is not None:
        return previous
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    model = kernel_type(items[0].x.shape[1]).double().to(config.device)
    if initial is not None:
        model.load_state_dict(initial.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    started = synchronized_time(config.device)
    curve, visits, losses = [], {}, []
    for step in range(1, config.steps + 1):
        archive = scheduled_item(items, new_source_ids, step)
        visit = visits.get(archive.source_id, 0)
        visits[archive.source_id] = visit + 1
        indices = selected_indices(archive, config, config.seed + items.index(archive), visit)
        optimizer.zero_grad(set_to_none=True)
        loss = joint_evidence(
            model,
            archive.x[indices],
            archive.collision[indices],
            archive.continuous_margin[indices] if config.historical_rss else None,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite historical joint evidence")
        optimize(model, optimizer, loss)
        losses.append(float(loss.detach()))
        if step % config.diagnostic_interval == 0 or step == config.steps:
            curve.append(
                {
                    "step": step,
                    "training_joint_evidence": float(np.mean(losses)),
                    "diagnostic_steps": len(losses),
                }
            )
            losses.clear()
            if step == config.steps:
                save_model(destination, model, "target", spec, step)
            report = {
                "status": "complete" if step == config.steps else "running",
                "specification": spec,
                "curve": curve,
                "final_step": step,
                "training_s": synchronized_time(config.device) - started,
            }
            write_json(destination.with_suffix(".json"), report)
            print(
                "joint_target",
                step,
                "/",
                config.steps,
                "evidence",
                curve[-1]["training_joint_evidence"],
                flush=True,
            )
    return report


def cache_episode(base, archive, config, seed, segment=0):
    initial, length = episode_layout(archive, config)
    indices = selected_indices(archive, config, seed, segment)
    frozen = adapted_model(base, archive.x[:0], [], [])
    physical = type(base)(base.dimension, deep=False).double().to(base.mean.device)
    for parameter in physical.parameters():
        parameter.requires_grad_(False)
    return {
        "target": archive,
        "indices": indices,
        "initial_count": initial,
        "length": length,
        "x": archive.x[indices],
        "labels": archive.collision[indices],
        "margins": archive.continuous_margin[indices] if config.historical_rss else None,
        "base": frozen,
        "physical_base": physical,
        "encoded_target": frozen.features(archive.x[indices]).detach(),
        "encoded_physical": physical.features(archive.x[indices]).detach(),
        "states": [(frozen, physical)],
        "reference": base.point_mean(1)[0].detach().clone(),
    }


def ensure_states(episode, count):
    if not 0 <= count < episode["length"]:
        raise ValueError("Sequential time must precede the end of the episode")
    while len(episode["states"]) <= count:
        n = len(episode["states"])
        x, labels = episode["x"][:n], episode["labels"][:n]
        margins = None if episode["margins"] is None else episode["margins"][:n]
        episode["states"].append(
            (
                adapted_model(
                    episode["base"], x, labels, margins, encoded=episode["encoded_target"][:n]
                ),
                adapted_model(
                    episode["physical_base"],
                    x,
                    labels,
                    margins,
                    learning_rate=0.01,
                    encoded=episode["encoded_physical"][:n],
                ),
            )
        )


def contexts_for_episode(history, episode, generator, config):
    target = episode["target"]
    contexts = []
    for archive in history:
        if archive.source_id == target.source_id:
            continue
        if archive.scenarios[0].kind != target.scenarios[0].kind:
            raise ValueError("Historical contexts must share the target functional scenario")
        indices = generator.choice(
            len(archive.scenarios),
            min(len(archive.scenarios), config.historical_context),
            replace=False,
        )
        contexts.append(
            replace(
                archive,
                scenarios=[archive.scenarios[i] for i in indices],
                collision=archive.collision[indices],
                safety=archive.safety[indices],
                rss_margin=archive.continuous_margin[indices],
            )
        )
    if not contexts:
        raise ValueError("Historical encoder needs a source other than the simulated target")
    return contexts


def episode_probability(encoder, contexts, episode, count):
    if count < episode["initial_count"]:
        raise ValueError("Training supervision starts after the common initialization")
    ensure_states(episode, count)
    encoded_query = encoder.features(episode["x"])
    priors = [
        historical_prior(
            encoder,
            archive,
            episode["x"],
            encoded_query=encoded_query,
            use_rss=episode["margins"] is not None,
        )
        for archive in contexts
    ]
    means, blocks = [torch.stack(values) for values in zip(*priors)]
    sources = {
        "means": means,
        "blocks": blocks,
        "weights": source_weights(contexts),
        "names": [a.source_id for a in contexts],
    }
    return sequential_probability(
        sources,
        episode["states"],
        episode["reference"],
        episode["x"],
        episode["labels"],
        episode["margins"],
        count,
        encoded_target=episode["encoded_target"],
        encoded_physical=episode["encoded_physical"],
    )


def train(
    history,
    destination,
    *,
    held_family=None,
    config=DEFAULT,
    initial_models=None,
    new_source_ids=(),
    parent_version=None,
    kernel_type=JointResponseKernel,
):
    torch.set_num_threads(1)
    history = tuple(history)
    spec = specification(history, held_family, config, kernel_type)
    if kernel_type is not JointResponseKernel:
        spec["response_kernel"] = kernel_type.__name__
    for model in initial_models or ():
        construction = getattr(model, "construction_specification", None)
        if construction is not None and construction["scenario"] != spec["scenario"]:
            raise ValueError("Initial shared model belongs to another functional scenario")
        if construction is not None and construction.get("parameter_definition") != spec.get(
            "parameter_definition"
        ):
            raise ValueError("Initial shared model uses another parameter definition")
    if not set(new_source_ids).issubset({a.source_id for a in history}):
        raise ValueError("New-source schedule contains an unavailable historical task")
    spec.update(new_source_ids=list(new_source_ids), parent_version=parent_version)
    destination = Path(destination)
    target_path, encoder_path = (
        destination / "target_kernel.pt",
        destination / "historical_encoder.pt",
    )
    target_report = train_target(
        history,
        target_path,
        spec,
        config,
        None if initial_models is None else initial_models[0],
        new_source_ids,
        kernel_type,
    )
    previous = completed_report(encoder_path, spec, kernel_type)
    if previous is not None:
        return {"target": target_report, "encoder": previous}
    base, _ = load_model(target_path, device=config.device, kernel_type=kernel_type)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(config.seed)
    encoder = kernel_type(base.dimension).double().to(config.device)
    encoder.load_state_dict((base if initial_models is None else initial_models[1]).state_dict())
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)
    generator = np.random.default_rng(config.seed)
    items = [a for a in history if len(a.scenarios) >= 2]
    episodes, visits, curve, losses, errors = {}, {}, [], [], []
    started = synchronized_time(config.device)
    for step in range(1, config.steps + 1):
        archive = scheduled_item(items, new_source_ids, step)
        visit = visits.get(archive.source_id, 0)
        visits[archive.source_id] = visit + 1
        segments = (
            1
            if archive.provenance.get("completed_task")
            else ceil(len(archive.scenarios) / config.budget)
        )
        key = archive.source_id, visit % segments
        if key not in episodes:
            episodes[key] = cache_episode(
                base, archive, config, config.seed + items.index(archive), key[1]
            )
        episode = episodes[key]
        count = int(generator.integers(episode["initial_count"], episode["length"]))
        contexts = contexts_for_episode(history, episode, generator, config)
        optimizer.zero_grad(set_to_none=True)
        p = episode_probability(encoder, contexts, episode, count)
        truth = p.new_tensor(episode["labels"][count], dtype=p.dtype)
        loss = F.binary_cross_entropy(p.clamp(1e-8, 1 - 1e-8), truth)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite next-response collision loss")
        optimize(encoder, optimizer, loss)
        losses.append(float(loss.detach()))
        errors.append(float((p.detach() - truth).square()))
        if step % config.diagnostic_interval == 0 or step == config.steps:
            curve.append(
                {
                    "step": step,
                    "training_nll": float(np.mean(losses)),
                    "training_brier": float(np.mean(errors)),
                    "diagnostic_steps": len(losses),
                    "source_id": archive.source_id,
                    "prefix_count": count,
                    "next_record_index": int(episode["indices"][count]),
                }
            )
            losses.clear()
            errors.clear()
            if step == config.steps:
                save_model(encoder_path, encoder, "historical_encoder", spec, step)
            report = {
                "status": "complete" if step == config.steps else "running",
                "specification": spec,
                "curve": curve,
                "final_step": step,
                "cached_episodes": len(episodes),
                "training_s": synchronized_time(config.device) - started,
            }
            write_json(encoder_path.with_suffix(".json"), report)
            print(
                "historical_encoder",
                step,
                "/",
                config.steps,
                "next_nll",
                curve[-1]["training_nll"],
                flush=True,
            )
    return {"target": target_report, "encoder": report}


def build_scope(
    kind, held_family=None, *, config=DEFAULT, dataset_manifest=None, model_root=MODEL_ROOT
):
    from experiments.history import load_training_archives
    from experiments.prepare_joint import DATASET_ROOT

    manifest = (
        DATASET_ROOT / "manifest.json" if dataset_manifest is None else Path(dataset_manifest)
    )
    history = load_training_archives(manifest, kind, held_family)
    destination = (
        Path(model_root)
        / kind
        / ("all_sources" if held_family is None else f"without_{held_family}")
    )
    resource_path = destination / "resources.json"
    with PeakMemory(device=config.device) as memory:
        report = train(history, destination, held_family=held_family, config=config)
    if not resource_path.exists():
        write_json(resource_path, memory.result)
    return {
        "scenario": kind,
        "scope": destination.name,
        "destination": str(destination.resolve()),
        "resources": read_json(resource_path),
        **report,
    }


def main(*, config=DEFAULT, dataset_manifest=None, model_root=MODEL_ROOT):
    reports = []
    workers = min(4, len(TEST_SCENARIOS)) if torch.device(config.device).type == "cpu" else 1

    def record(report):
        reports.append(report)
        reports.sort(key=lambda r: TEST_SCENARIOS.index(r["scenario"]))
        write_json(
            Path(model_root) / "construction.json",
            {
                "status": "complete" if len(reports) == len(TEST_SCENARIOS) else "running",
                "tasks": reports,
                "covariance_mode": COVARIANCE_MODE,
                "new_physical_calls": 0,
                "parallel_workers": workers,
                "device": config.device,
            },
        )

    options = dict(config=config, dataset_manifest=dataset_manifest, model_root=model_root)
    if workers == 1:
        for kind in TEST_SCENARIOS:
            record(build_scope(kind, **options))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(build_scope, kind, **options) for kind in TEST_SCENARIOS]
            for future in as_completed(pending):
                record(future.result())
    return reports


if __name__ == "__main__":
    main()
