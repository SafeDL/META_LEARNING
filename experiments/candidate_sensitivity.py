"""Nested finite-pool comparisons with one evaluator-owned largest truth bank per ADS."""

import os
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from controllers.highway.catalog import CURRENT_INDEPENDENT_SUTS

from .joint_validation import DEFAULT, OUTPUT, freeze_protocol, initial_models, target_bank
from .pilot import MAIN_METHODS, fixed_comparison, historical_classifiers
from .prepare_joint import DATASET_ROOT
from .scenarios import candidate_pool
from .storage import write_json

SIZES = (1000, 2000, 4000)


class CandidateSubset:
    def __init__(self, bank, candidates):
        if candidates != bank.candidates[: len(candidates)]:
            raise ValueError("Candidate sensitivity requires exact nested coordinate prefixes")
        self.bank = bank
        self.candidates = candidates
        self.manifest = bank.manifest
        self.ids = {s.scenario_id for s in candidates}

    def __call__(self, scene):
        if scene.scenario_id not in self.ids:
            raise ValueError("Execution lies outside the registered candidate subset")
        return self.bank(scene)

    def execution_path(self, index):
        return self.bank.execution_path(index)


def run(
    root=OUTPUT / "sensitivity",
    *,
    config=replace(DEFAULT, scenarios=("front_brake",)),
    sizes=SIZES,
    systems=CURRENT_INDEPENDENT_SUTS,
    dataset_manifest=DATASET_ROOT / "manifest.json",
    model_loader=initial_models,
    classifier_builder=partial(historical_classifiers, train_missing=False),
    classifier_root=OUTPUT / "pilot/models",
):
    torch.set_num_threads(1)
    root = Path(root)
    if config.scenarios != ("front_brake",) or not sizes or tuple(sorted(set(sizes))) != sizes:
        raise ValueError("Sensitivity uses front braking with distinct increasing pool sizes")
    largest = replace(config, candidate_count=max(sizes))
    freeze_protocol(
        root,
        "sensitivity",
        largest,
        systems,
        dataset_manifest,
        extra={
            "candidate_sizes": list(sizes),
            "initial_pool_size": min(sizes),
            "candidate_replicate": 30,
            "main_methods": list(MAIN_METHODS),
            "controls": [],
            "source_availability": "fixed original same-scenario historical ADSs",
            "physical_bank": "largest nested pool only",
            "historical_classifier_root": os.path.relpath(classifier_root, root).replace("\\", "/"),
        },
    )
    banks, classifiers = {}, {}

    def load_bank(unused_root, sut, kind, task_config, *, replicate):
        if sut.sut_id not in banks:
            banks[sut.sut_id] = target_bank(
                Path(dataset_manifest).parent / "targets/sensitivity",
                sut,
                kind,
                largest,
                replicate=replicate,
            )
        return CandidateSubset(
            banks[sut.sut_id],
            candidate_pool(kind, task_config.candidate_count, replicate=replicate),
        )

    def load_classifier(history, unused_destination):
        kind = history[0].scenarios[0].kind
        if kind not in classifiers:
            classifiers[kind] = classifier_builder(history, Path(classifier_root) / kind)
        return classifiers[kind]

    rows = []
    for size in sizes:
        comparison = root / str(size)
        result = fixed_comparison(
            comparison,
            suite="sensitivity",
            config=replace(config, candidate_count=size),
            systems=systems,
            dataset_manifest=dataset_manifest,
            model_loader=model_loader,
            methods=MAIN_METHODS,
            candidate_replicate=30,
            bank_loader=load_bank,
            classifier_builder=load_classifier,
            initial_pool_size=min(sizes),
        )
        rows.extend(
            {
                **row,
                "candidate_count": size,
                "result_path": f"{size}/{row['result_path']}",
            }
            for row in result
        )
        write_json(
            root / "reports/summary.json",
            {
                "status": "complete" if size == sizes[-1] else "running",
                "runs": rows,
                "physical_truth_records": len(systems) * max(sizes),
                "logical_queries": sum(min(config.budget, row["candidate_count"]) for row in rows),
            },
        )
    return rows


if __name__ == "__main__":
    run()
