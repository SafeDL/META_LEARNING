"""Replay one frozen joint-feedback lookahead rule with the original response model."""

import copy
import hashlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import torch

from testing_methods.history_transfer.selection import select

from .attribution import (
    CHECKPOINTS,
    DATA,
    compact_result,
    models,
    prediction_metrics,
    read_source,
    source_path,
)
from .executions import ExecutionBank
from .failure_metrics import summarize
from .history import model_version
from .method_exploration import OUTPUT, task_inputs
from .protocol import ROOT
from .query import QueryLedger, run_isolated
from .scenarios import TEST_SCENARIOS, features
from .schema import Scenario
from .storage import read_json, write_json

ACQUISITION = {"shortlist_count": 4, "fantasy_count": 4, "seed": 11}
E3_CONFIGURATION = {
    "implementation": "testing_methods/history_transfer/lookahead.py",
    "response_model": "exact original joint kernel, encoder and source calculation record IDs",
    "utility": "p(x)+mean_fantasy(sum of top remaining_budget-1 probabilities excluding x)",
    "shortlist": "top 2 collision probabilities, then highest weighted component-probability "
    "variance until 4 unique candidates; ties by scenario ID at 1e-12 tolerance",
    **ACQUISITION,
    "sampling": "scrambled 4-D Sobol, seed=11+paid_prefix_count; common uniforms per shortlist; "
    "mixture component, 2 Gaussian latent coordinates, Bernoulli-logistic collision draw",
    "rss_observation_noise_sd": 0.05,
    "rss_fantasy_support": "unbounded Gaussian approximation; no clipping or conversion to "
    "physical QueryFeedback; record below-minus-one probability and sample fraction",
    "fantasy_update": "prequery collision likelihood updates global source weights, then "
    "target and raw kernels each adapt six scalars for 10 steps from fixed initialization; "
    "learning rates 0.003 and 0.01; predict remaining candidates in batches of 128",
    "last_query": "exact greedy choice with zero fantasy branches",
    "common_diagnostics": "compute hypothetical choices at 20,50,100 paid records while "
    "executing the saved Random order; at budget 200 only the unchanged prediction is recorded",
    "combination": "neither E1 local reliability nor E2 shared/private kernels",
    "additional_trainable_parameters": 0,
    "additional_real_adaptation_steps": 0,
    "device": "cpu",
    "precision": "float64",
    "torch_threads": 1,
    "parallel_workers": 4,
    "batch_size": 128,
    "max_task_wall_s": 3600,
    "max_task_cpu_gib": 8,
    "max_task_gpu_gib": 0,
    "resource_basis": "historical-only 2000-candidate,198-response,16-branch probe: "
    "13.15s choice and 0.57 GiB peak. Late-step cost times 180 plus real updates/prediction "
    "is about 2748s; 3600s is a fixed execution cap, not an estimated average runtime",
    "check_command": "python -B -m pytest tests/history_transfer/test_lookahead.py "
    "tests/history_transfer/test_joint_feedback.py tests/history_transfer/test_feature_cache.py "
    "tests/history_transfer/test_reliability.py tests/experiments/test_source_registration.py "
    "tests/experiments/test_source_audit.py -q",
    "regression_check_command": 'python -B -c "from experiments.lookahead_exploration '
    'import check_original_prefixes; check_original_prefixes()"',
    "expected_checks": "mixture moments; original component point posteriors; identical "
    "fantasy/real update; immutable paid state; last-query greedy; chunk/seed invariance; "
    "fixed-prefix original predictions and no future truth disclosure",
}


def check_original_prefixes():
    """Verify frozen paid-prefix predictions before admitting a selector change."""
    from testing_methods.history_transfer.inference import predict_components, source_tensors
    from testing_methods.history_transfer.model import JointResponseKernel
    from testing_methods.history_transfer.shared_private import SharedPrivateKernel
    from testing_methods.history_transfer.training import load_model

    torch.set_num_threads(1)
    rows = []
    for kind in TEST_SCENARIOS:
        task = next(t for t in task_inputs() if t["scenario"] == kind)
        original = read_json(ROOT / task["joint"])
        common = read_json(ROOT / task["common_feedback"])
        x = features([Scenario(**s) for s in original["specification"]["candidates"]])
        base, _ = models(kind)
        source = source_tensors(read_source(source_path(kind, True)), base)
        reference = base.point_mean(1)[0].detach()
        for snapshot in common["result"]["prediction_snapshots"]:
            model = copy.deepcopy(base)
            physical = JointResponseKernel(base.dimension, deep=False).double()
            with torch.no_grad():
                for name, value in snapshot["target_parameters"].items():
                    getattr(model, name).fill_(value)
                for name, value in snapshot["physical_parameters"].items():
                    getattr(physical, name).fill_(value)
                paid = snapshot["observed_indices"]
                queries = common["result"]["queries"][: len(paid)]
                components, _ = predict_components(
                    model,
                    physical,
                    x,
                    source,
                    reference,
                    paid,
                    [q["feedback"]["collision"] for q in queries],
                    [q["feedback"]["safety"]["rss_margin"] for q in queries],
                    batch_size=128,
                )
                weights = components.new_tensor(snapshot["component_weights"])
                probability = (weights[:, None] * components).sum(0).numpy()
            error = float(np.abs(probability - snapshot["probabilities"]).max())
            if error > 1e-6:
                raise AssertionError((kind, snapshot["paid_queries"], error))
            rows.append(
                {"scenario": kind, "paid_queries": snapshot["paid_queries"], "max_error": error}
            )
    typed = []
    for kind in TEST_SCENARIOS:
        resource = read_json(OUTPUT / "e2/models" / kind / "resources.json")
        for stage, key in (("target_kernel", "target"), ("historical_encoder", "encoder")):
            model, _ = load_model(
                OUTPUT / "e2/models" / kind / f"{stage}.pt", kernel_type=SharedPrivateKernel
            )
            if model_version(model) != resource["versions"][key]:
                raise ValueError("A maintained E2 checkpoint changed its model identity")
            typed.append({"scenario": kind, "stage": stage, "model_version": model_version(model)})
    result = {
        "status": "passed",
        "snapshots": rows,
        "shared_private_checkpoints": typed,
        "new_logical_queries": 0,
        "new_physical_executions": 0,
    }
    write_json(OUTPUT / "e3/default_kernel_regression.json", result)
    return result


def register_e3():
    if read_json(OUTPUT / "e2/reports/summary.json")["status"] != "complete":
        raise ValueError("Finish and verify E2 before freezing E3")
    engineering = read_json(OUTPUT / "e3/engineering.json")
    if engineering["status"] != "passed":
        raise ValueError("E3 selector integration and original-model regression must pass first")
    for path, digest in engineering["source_files"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError("E3 implementation changed after engineering acceptance")
    path = OUTPUT / "protocol.json"
    protocol = read_json(path)
    if protocol["tasks"] != task_inputs():
        raise ValueError("E3 must preserve the original 32 tasks")
    current = protocol["e3"]
    if current == {"configuration": "unfrozen", "status": "not_started"}:
        protocol["e3"] = E3_CONFIGURATION
        write_json(path, protocol)
    elif current != E3_CONFIGURATION:
        raise ValueError("E3 differs from its frozen configuration")


def replay_task(task, phase):
    torch.set_num_threads(1)
    kind = task["scenario"]
    original, common = read_json(ROOT / task["joint"]), read_json(ROOT / task["common_feedback"])
    base, _ = models(kind)
    source = read_source(source_path(kind, True))
    if model_version(base) != original["specification"]["target_model_version"]:
        raise ValueError("E3 changed the original fixed target model")
    if source["specification"] != original["specification"]["source_snapshot"]:
        raise ValueError("E3 changed the original source snapshot")
    specification = {
        "platform": "HighwayEnv",
        "stage": "method exploration E3",
        "sut_id": task["sut_id"],
        "scenario": kind,
        "phase": phase,
        "reference_joint": task["joint"],
        "reference_common": task["common_feedback"],
        "source_ref": str(source_path(kind, True).relative_to(ROOT)),
        "source_snapshot": source["specification"],
        "target_model_version": model_version(base),
        "configuration": E3_CONFIGURATION,
        "new_physical_executions": 0,
    }
    path = OUTPUT / "e3" / phase / task["sut_id"] / f"{kind}.json.gz"
    if path.exists():
        if read_json(path)["specification"] != specification:
            raise ValueError("Saved E3 replay has a different frozen configuration")
        return str(path.relative_to(OUTPUT))
    bank = ExecutionBank(DATA / "targets/fixed" / task["sut_id"] / kind)
    if list(bank.candidates) != [Scenario(**s) for s in original["specification"]["candidates"]]:
        raise ValueError("E3 candidate coordinates differ from the original comparison")
    ledger = QueryLedger(bank.candidates, 200, bank, joint_feedback=True)
    public = {
        "candidates": bank.candidates,
        "pretrained": base,
        "sources": source,
        "initial_indices": original["result"]["selected_indices"][:20],
        "budget": 200,
        "snapshot_counts": CHECKPOINTS,
        "query_order": common["result"]["selected_indices"] if phase == "common" else None,
        "lookahead": ACQUISITION,
        "device": "cpu",
        "batch_size": 128,
    }
    try:
        result = run_isolated(
            select,
            public,
            ledger,
            wall_limit_s=E3_CONFIGURATION["max_task_wall_s"],
            memory_limit_bytes=E3_CONFIGURATION["max_task_cpu_gib"] * 2**30,
        )
        labels = original["evaluation"]["candidate_collisions"]
        metrics = summarize(
            bank.candidates,
            labels,
            result["queries"],
            ledger.trajectories,
            initial_count=20,
            query_ledger=ledger.records,
        )
        if phase == "common":
            for actual, expected in zip(
                result["queries"], common["result"]["queries"], strict=True
            ):
                np.testing.assert_allclose(
                    actual["prequery_component_probabilities"],
                    expected["prequery_component_probabilities"],
                    atol=1e-6,
                    rtol=0,
                )
        for snapshot in result["prediction_snapshots"]:
            snapshot["evaluation"] = prediction_metrics(snapshot, labels)
        write_json(
            path,
            {
                "specification": specification,
                "result": compact_result(result, bank.root),
                "metrics": metrics,
            },
        )
    except Exception as error:
        write_json(
            path.with_name(f"{kind}_failure.json"),
            {
                "specification": specification,
                "error": repr(error),
                "completed_queries": len(ledger.records),
                "last_index": ledger.records[-1]["index"] if ledger.records else None,
                "physical_executions": 0,
            },
        )
        raise
    return str(path.relative_to(OUTPUT))


def run():
    register_e3()
    progress_path = OUTPUT / "e3/progress.json"
    if progress_path.exists() and read_json(progress_path)["status"] == "stopped":
        raise RuntimeError("E3 stopped; inspect its audit evidence before a directed repair")
    completed = []
    try:
        for phase in ("common", "adaptive"):
            tasks = iter(task_inputs())
            with ProcessPoolExecutor(max_workers=4) as pool:
                pending = {pool.submit(replay_task, next(tasks), phase) for _ in range(4)}
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    # Check every finished task before starting another after a failure.
                    paths = [future.result() for future in done]
                    completed.extend(paths)
                    write_json(
                        progress_path, {"status": "running", "phase": phase, "completed": completed}
                    )
                    for path in paths:
                        print("E3", phase, len(completed), path, flush=True)
                    for _ in done:
                        task = next(tasks, None)
                        if task is not None:
                            pending.add(pool.submit(replay_task, task, phase))
    except Exception as error:
        write_json(
            progress_path,
            {
                "status": "stopped",
                "completed": completed,
                "error": repr(error),
                "policy": "no new tasks after failure; already-running tasks "
                "finish within their registered caps",
            },
        )
        raise
    write_json(progress_path, {"status": "complete", "completed": completed})


if __name__ == "__main__":
    run()
