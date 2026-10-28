"""Build Indago from history and check the authors' architecture before paid replay."""

import ast
import time
import urllib.request
from dataclasses import asdict
from typing import Callable

import torch
from torch import nn

from controllers.highway.catalog import BY_ID
from testing_methods.baselines.indago.method import DEFAULT as TRAINING
from testing_methods.baselines.indago.method import FailurePredictor, load, train
from testing_methods.baselines.indago.pool import DEFAULT as SEARCH
from testing_methods.baselines.indago.pool import run

from .collision_metrics import collision_recall, summarize
from .construction import load_archives
from .discovery_config import write_json
from .executions import ExecutionBank, execution_contract
from .initialization import initial_indices
from .protocol import DEFAULT, ROOT, TEST_RECORDS_ROOT
from .query import QueryLedger, run_isolated
from .source_registration import specification_matches

OUTPUT = ROOT / "experiment_outputs/indago/collision/discovery/domain_comparison/stage_a"
AUTHOR_COMMIT = "b1afed834f2a82d6f8a267217b31f8e375855ed1"
AUTHOR_SOURCE = (
    "https://raw.githubusercontent.com/matteobiagiola/drl-testing-experiments/"
    f"{AUTHOR_COMMIT}/indago/avf/avf_policy.py"
)


def check_author_architecture():
    # Execute only the reviewed architecture factory, without simulator imports.
    source = urllib.request.urlopen(AUTHOR_SOURCE, timeout=30).read().decode("utf-8")
    tree = ast.parse(source)
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AvfPolicy"
    )
    factory = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_model_architecture"
    )
    factory.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    namespace = {"nn": nn, "Callable": Callable}
    exec(compile(module, AUTHOR_SOURCE, "exec"), namespace)
    errors = []
    for layers in range(1, 5):
        model = FailurePredictor(6, layers=layers).eval()
        reference = namespace["get_model_architecture"](6, layers, "mlp")().eval()
        reference.load_state_dict(model.network.state_dict())
        x = torch.randn(11, 6)
        with torch.no_grad():
            error = float((reference(x) - model(x)).abs().max())
        if error != 0:
            raise ValueError("Indago MLP differs from the pinned author implementation")
        errors.append({"layers": layers, "maximum_logit_error": error})
    return {"source": AUTHOR_SOURCE, "commit": AUTHOR_COMMIT, "checks": errors}


def main():
    torch.set_num_threads(1)
    author = check_author_architecture()
    training = []
    for kind in ("cut_in", "cut_out"):
        history, _ = load_archives(kind)
        path = OUTPUT / "models" / kind / "classifier.pt"
        if path.exists():
            _, checkpoint = load(path)
            from testing_methods.common.response import archive_fingerprints

            if checkpoint["history_fingerprints"] != archive_fingerprints(history) or checkpoint[
                "config"
            ] != asdict(TRAINING):
                raise ValueError(
                    "Saved Indago model belongs to different allowed historical training"
                )
            report = {k: v for k, v in checkpoint.items() if k != "state_dict"}
        else:
            report = train(history, path)
        training.append({"scenario": kind, "path": str(path.relative_to(ROOT)), **report})
    kind, target, seed = "cut_in", "development_idm_mobil", 11
    _, development = load_archives(kind)
    archive = next(a for a in development if a.source_id == target)
    bank = ExecutionBank(TEST_RECORDS_ROOT / "physical/development" / kind / target)
    if bank.candidates != archive.scenarios or not specification_matches(
        bank.manifest["contract"], execution_contract(BY_ID[target])
    ):
        raise ValueError("Indago validation requires its qualified original development bank")
    classifier, _ = load(OUTPUT / "models" / kind / "classifier.pt")
    before = {k: v.clone() for k, v in classifier.state_dict().items()}
    initial = initial_indices(len(bank.candidates), DEFAULT.initial_count, seed)
    public = {
        "candidates": bank.candidates,
        "classifier": classifier,
        "initial_indices": initial,
        "budget": DEFAULT.budget,
        "seed": seed,
    }
    ledger = QueryLedger(bank.candidates, DEFAULT.budget, bank)
    started = time.perf_counter()
    result = run_isolated(run, public, ledger)
    wall_s = time.perf_counter() - started
    if not all(torch.equal(before[k], v) for k, v in classifier.state_dict().items()):
        raise ValueError("Indago's fixed classifier changed during target search")
    write_json(
        OUTPUT / "first_run.json",
        {
            "stage": "A mechanism validation; not the frozen development comparison",
            "target_id": target,
            "scenario": kind,
            "algorithm_seed": seed,
            "initial_indices": initial,
            "result": result,
            "query_ledger": ledger.records,
            "metrics": {
                **summarize(bank.candidates, result["queries"], ledger.trajectories),
                **collision_recall(bank.candidates, archive.collision.tolist(), result["queries"]),
            },
            "cost": {
                "selector_and_replay_wall_s": wall_s,
                "new_physical_calls": 0,
                "equivalent_queries": DEFAULT.budget,
            },
        },
    )
    report = {
        "status": "passed",
        "author_architecture": author,
        "training": training,
        "search_config": asdict(SEARCH),
        "queries": len(ledger.records),
        "classifier_fixed": True,
        "initialization_included": True,
        "new_physical_calls": 0,
        "formal_comparison_frozen": False,
    }
    write_json(OUTPUT / "audit.json", report)
    print({"status": "passed", "queries": len(ledger.records), "new_physical_calls": 0}, flush=True)
    return report


if __name__ == "__main__":
    main()
