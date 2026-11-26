"""Shared baseline construction and selection for registered response tasks."""

from dataclasses import asdict
from pathlib import Path

from testing_methods.baselines.paired_system_active_evaluation import method as paired
from testing_methods.baselines.response_baselines import collision as random
from testing_methods.baselines.scenario_fuzz import model as fuzz_model
from testing_methods.baselines.scenario_fuzz import pool as scenario_fuzz
from testing_methods.common.response import archive_fingerprints, shared_domain

from .costs import measure_resources
from .storage import write_json

BASELINES = {
    "random": random,
    "scenario_fuzz": scenario_fuzz,
    "paired_system_active_evaluation": paired,
}


def historical_classifiers(
    history,
    destination,
    *,
    train_missing=True,
    names=("scenario_fuzz",),
    graph_fn=None,
    graph_definition=None,
):
    """Construct or load classifiers from admitted histories of one scenario."""
    if len({a.scenarios[0].kind for a in history}) != 1:
        raise ValueError("Baseline construction requires one functional scenario")
    domain = shared_domain(history)
    if domain is None and (graph_fn is not None or graph_definition is not None):
        raise ValueError("A custom graph builder requires an explicit parameter definition")
    classifiers, reports = {}, {}
    modules = {"scenario_fuzz": fuzz_model}
    for name in names:
        if name not in modules:
            raise ValueError(f"Unsupported historical classifier baseline: {name}")
        options = {}
        if domain is not None:
            if graph_fn is None or not graph_definition:
                raise ValueError("New-domain graph training requires its registered public builder")
            options = {"graph_fn": graph_fn, "graph_definition": graph_definition}
        module = modules[name]
        path = Path(destination) / name / "classifier.pt"
        if train_missing and not path.exists():
            module.train(history, path, config=module.DEFAULT, **options)
        model, checkpoint = module.load(path)
        if (
            checkpoint["history_fingerprints"] != archive_fingerprints(history)
            or checkpoint["config"] != asdict(module.DEFAULT)
            or checkpoint["held_family"] is not None
        ):
            raise ValueError("Baseline classifier differs from its historical construction")
        if checkpoint.get("parameter_definition") != (None if domain is None else domain.to_dict()):
            raise ValueError("Baseline classifier uses another parameter definition")
        if domain is not None and checkpoint.get("graph_definition") != graph_definition:
            raise ValueError("Baseline classifier uses another public graph definition")
        classifiers[name] = model
        reports[name] = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    if train_missing:
        write_json(Path(destination) / "construction.json", reports)
    return classifiers


@measure_resources
def baseline_selector(
    candidates, oracle, *, method, initial_indices, budget, seed, domain=None, **state
):
    if domain is not None:
        if method == "paired_system_active_evaluation":
            state["feature_fn"] = domain.features
        elif method == "scenario_fuzz":
            state["domain"] = domain
        elif method != "random":
            raise ValueError(f"Unsupported baseline for domain tasks: {method}")
    return BASELINES[method].run(
        candidates, oracle, initial_indices=initial_indices, budget=budget, seed=seed, **state
    )
