"""Build every preregistered representation from role-checked new responses."""

import argparse
from dataclasses import asdict
import json

import torch

from benchmark.construction import load_construction
from benchmark.protocol import ROOT
from benchmark.scenarios import DOMAINS
from benchmark.studies import SCOPES
from methods.baselines.fsbo.method import train as train_fsbo, CONFIG as FSBO_CONFIG
from methods.baselines.pooled_ft.method import train as train_pooled, CONFIG as POOLED_CONFIG
from methods.common.model_contract import validate_model_contract
from methods.gcrt.config import CONFIG
from methods.gcrt.train import train as train_main


def model_path(method, scope, scenario):
    root = (ROOT / "results/gcrt/collision/models" if method == "history_response" else
            ROOT / "results/baselines" / method / "collision/models")
    return root / scope / f"{scenario}.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("core", "complete"))
    stage = parser.parse_args().stage
    configurations = {"history_response": asdict(CONFIG), "fsbo": FSBO_CONFIG, "pooled_ft": POOLED_CONFIG}
    for kind in DOMAINS:
        historical, development = load_construction(kind)
        selected = [SCOPES["all"]] if stage == "core" else SCOPES.values()
        for scope in selected:
            history = [a for a in historical if a.source_id in scope.history_ids]
            validation = [a for a in development if a.source_id in scope.development_ids]
            trainers = [("history_response", train_main)]
            # Source-swap comparisons isolate adaptation; both arms use a common dev set.
            if not scope.name.startswith("swap_"):
                trainers += [("fsbo", train_fsbo), ("pooled_ft", train_pooled)]
            for method, trainer in trainers:
                destination = model_path(method, scope.name, kind)
                if destination.exists():
                    report = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
                    checkpoint = torch.load(destination, weights_only=True, map_location="cpu")
                    validate_model_contract(checkpoint, method, configuration=configurations[method])
                    from methods.common.response import archive_fingerprints
                    if (report["status"] != "complete"
                            or checkpoint["history_fingerprints"] != archive_fingerprints(history)
                            or checkpoint["development_fingerprints"] != archive_fingerprints(validation)
                            or checkpoint["held_family"] != scope.held_family
                            or report["curve"][-1]["step"] != configurations[method]["steps"]):
                        raise ValueError("Existing model is incomplete or belongs to different archives")
                    continue
                trainer(history, validation, destination, held_family=scope.held_family)


if __name__ == "__main__":
    main()
