"""Publish the four historical ScenarioFuzz models and paired-system proxies."""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np
import torch

from testing_methods.baselines.paired_system_active_evaluation.method import build_proxy
from testing_methods.baselines.scenario_fuzz import model as fuzz
from testing_methods.common.response import archive_fingerprints

from experiments.budget.protocol import INPUT, OUTPUT, ROOT, RUN_SCENARIOS, declaration, sha
from experiments.costs import PeakMemory
from experiments.history import model_version
from experiments.history_transfer_experiment import freeze, registered_populations, scenario_history
from experiments.parallel import TRAINING_DEVICE, initialize_worker, prepare_processes, training_workers
from experiments.storage import read_json, write_json


def construction_specification(history):
    paths = [ROOT / "testing_methods/baselines/scenario_fuzz" / name
             for name in ("model.py", "graphs.py", "formal_graphs.py")]
    paths += [ROOT / "testing_methods/baselines/paired_system_active_evaluation/method.py",
              ROOT / "testing_methods/common/response.py", ROOT / "experiments/budget/prepare.py"]
    return {
        "history_fingerprints": archive_fingerprints(history),
        "history_ids": [a.source_id for a in history],
        "configuration": asdict(fuzz.DEFAULT), "training_device": TRAINING_DEVICE,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        "target_labels_used": False,
    }


def train_scene(kind):
    initialize_worker()
    directory = OUTPUT / "models" / kind
    history = scenario_history(INPUT, kind)
    specification = construction_specification(history)
    freeze(directory / "construction.json", specification)
    path = directory / "scenario_fuzz/classifier.pt"
    receipt = directory / "training_completion.json"
    if receipt.exists():
        saved = read_json(receipt)
        if saved["specification"] != specification or saved["weights_sha256"] != sha(path):
            raise ValueError("Completed historical classifier changed")
        return saved
    print({"scene": kind, "stage": "historical_training", "pid": os.getpid()}, flush=True)
    with PeakMemory(device=TRAINING_DEVICE) as memory:
        if not path.exists():
            fuzz.train(history, path)
        model, checkpoint = fuzz.load(path)
    if (checkpoint["config"] != asdict(fuzz.DEFAULT)
            or checkpoint["history_fingerprints"] != archive_fingerprints(history)
            or checkpoint["training_device"] != TRAINING_DEVICE
            or len(checkpoint["curve"]) != fuzz.DEFAULT.epochs
            or not all(torch.isfinite(p).all() for p in model.state_dict().values())):
        raise ValueError("Historical graph model failed its declared construction")
    candidates = registered_populations(INPUT)["target", kind]
    with PeakMemory() as proxy_memory:
        proxy, proxy_spec = build_proxy(candidates, history)
    if proxy.shape != (2000,) or not np.isfinite(proxy).all():
        raise ValueError("Historical paired-system proxy is invalid")
    write_json(directory / "paired_proxy.json", {
        "values": proxy.tolist(), "specification": proxy_spec,
        "candidate_ids": [s.scenario_id for s in candidates], "resources": proxy_memory.result,
    })
    result = {
        "status": "complete", "specification": specification,
        "model_version": model_version(model), "weights_sha256": sha(path),
        "epochs": len(checkpoint["curve"]), "training_resources": memory.result,
        "training_rows": checkpoint["training_rows"],
        "validation_rows": checkpoint["validation_rows"],
        "paired_proxy_sha256": sha(directory / "paired_proxy.json"),
    }
    write_json(receipt, result)
    print({"scene": kind, "stage": "historical_training", "status": "complete"}, flush=True)
    return result


def run():
    initialize_worker()
    freeze(OUTPUT / "protocol.json", declaration())
    from experiments.transfer_admission import load_acceptance

    print({"stage": "verify_reused_e01_admission", "pid": os.getpid()}, flush=True)
    acceptance = load_acceptance(INPUT)
    freeze(OUTPUT / "input_admission.json", {
        "source": str(INPUT.relative_to(ROOT)),
        "acceptance_sha256": sha(INPUT / "reports/method_acceptance.json"),
        "proof_items": len(acceptance["proof_sha256"]),
        "scope": "reuse unchanged E01 data, ADS and GP identities; E02 admits its selectors separately",
    })
    print({"stage": "verify_reused_e01_admission", "status": "complete"}, flush=True)
    prepare_processes()
    with ProcessPoolExecutor(max_workers=training_workers(), mp_context=mp.get_context("spawn")) as pool:
        pending = {pool.submit(train_scene, kind): kind for kind in RUN_SCENARIOS}
        for future in as_completed(pending):
            future.result()
            print({"stage": "published", "scene": pending[future]}, flush=True)


if __name__ == "__main__":
    run()
