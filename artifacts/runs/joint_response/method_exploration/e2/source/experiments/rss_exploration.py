"""Train and replay one registered shared/private collision-RSS candidate."""

import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import psutil
import torch
from threadpoolctl import threadpool_limits

from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import Config, load_model, train

from .attribution import (
    CHECKPOINTS,
    DATA,
    compact_result,
    prediction_metrics,
    read_source,
)
from .costs import PeakMemory
from .executions import ExecutionBank
from .history import build_source, load_training_archives, model_version
from .method_exploration import E1_CONFIGURATION, OUTPUT, task_inputs
from .protocol import ROOT
from .query import QueryLedger, run_isolated
from .scenarios import TEST_SCENARIOS
from .schema import Scenario
from .storage import read_json, write_json

MODEL_ROOT = OUTPUT / "e2/models"
CONFIG = Config()
E2_CONFIGURATION = {
    "implementation": "testing_methods/history_transfer/shared_private.py",
    "latent_model": "f=sf*(sqrt(q)*u+sqrt(1-q)*v); "
    "r=sr*(a*u+sqrt(1-a^2)*w); u,v,w independent unit-variance Matern-5/2 GPs",
    "fraction": "q=sigmoid(raw_shared_fraction)",
    "sharing": "a=tanh(raw_correlation); all shared output covariance is zero when a=0",
    "covariance": {
        "ff": "sf^2*(q*k_shared+(1-q)*k_collision)",
        "rr": "sr^2*(a^2*k_shared+(1-a^2)*k_rss)",
        "fr": "sf*sr*sqrt(q)*a*k_shared",
        "source": "C_source'=abs(a)*C_source+(1-abs(a))*blockdiag(C_ff,C_rr)",
    },
    "scales": "sf=exp(log_scale); sr=exp(log_margin_scale), absolute RSS standard deviation",
    "initial_values": {
        "q": 0.5,
        "a": -0.5,
        "shared_length": 0.3,
        "collision_length": 0.3,
        "rss_length": 0.3,
        "sf": 1.0,
        "sr": 0.3,
        "collision_mean": -1.4,
        "rss_mean": 0.5,
    },
    "prior_loading_scope": "functional-scenario scalars; spatial differences through "
    "historical embedding, separate latent length scales and conditional posteriors",
    "zero_sharing_check": "set a=0 in target and raw kernels; source cross-output blocks "
    "also vanish; fixed remaining parameters reproduce collision-only conditioning",
    "rss_observation_noise_sd": 0.05,
    "offline_additional_scalars": [
        "raw_shared_fraction",
        "log_collision_lengthscale",
        "log_rss_lengthscale",
    ],
    "online_scalars": [
        "mean",
        "margin_mean",
        "log_lengthscale",
        "log_scale",
        "log_margin_scale",
        "raw_correlation",
    ],
    "online_initialization": "fixed learned base each round; raw component starts at fixed defaults",
    "online_steps": 10,
    "online_learning_rates": {"target": 0.003, "raw": 0.01},
    "online_gradient_clipping": "norm 5 separately for (margin_mean,log_margin_scale) "
    "and the other trainable parameters; original model retains its original single group",
    "offline": asdict(CONFIG),
    "offline_scope": "all eight original historical ADS archives per scenario; "
    "two stages trained from scratch, final checkpoints, no target or development responses",
    "source_views": "exact full-method view_record_ids from the original per-scenario snapshot",
    "source_weights": "global, query-before-feedback collision likelihood only; E1 not combined",
    "group_prior": {"history": 0.75, "independent_target": 0.25, "raw_mass": 0.1},
    "controls": "reuse archived original full-method and target-collision-only runs; "
    "training/model differences make this a workflow comparison",
    "conditional_diagnostic": "at each paid prefix hold parameters, source snapshot and "
    "weights fixed; compare p(U|Y,R) with p(U|Y) for the same new model; "
    "unqueried truth enters evaluation only",
    "device": "cpu",
    "precision": "float64",
    "workers": 4,
    "torch_threads": 1,
    "batch_size": 128,
    "max_build_wall_s": 86400,
    "max_build_cpu_gib": 24,
    "max_build_gpu_gib": 0,
    "max_task_wall_s": 1800,
    "max_task_cpu_gib": 8,
    "max_task_gpu_gib": 0,
    "resource_basis": "archived per-scenario training took 20,200-21,300 s and up to "
    "about 5 GiB CPU; allow three-kernel construction overhead without reducing training steps",
    "check_command": "python -B -m pytest tests/history_transfer/test_shared_private.py "
    "tests/history_transfer/test_joint_feedback.py tests/history_transfer/test_training.py "
    "tests/history_transfer/test_sequential_training.py tests/history_transfer/test_raw_target.py -q",
    "expected_checks": "independent latent covariance, PSD, source contraction, zero-sharing "
    "conditioning and optimizer isolation, dense/block agreement, six online scalars, "
    "strict typed checkpoints, batch invariant selection and future-feedback isolation",
}


def register_e2():
    path = OUTPUT / "protocol.json"
    protocol = read_json(path)
    if protocol["tasks"] != task_inputs() or protocol["e1"] != E1_CONFIGURATION:
        raise ValueError("E2 must preserve the original task list and E1 protocol")
    if read_json(OUTPUT / "e1/progress.json")["status"] != "complete":
        raise ValueError("Complete E1 before starting E2")
    current = protocol["e2"]
    if current.get("configuration") == "unfrozen" and current["status"] == "not_started":
        protocol["e2"] = E2_CONFIGURATION
        write_json(path, protocol)
    elif current != E2_CONFIGURATION:
        raise ValueError("E2 differs from its registered configuration")
    return protocol


def source_destination(kind):
    return OUTPUT / "e2/sources" / f"{kind}.npz"


def train_scenario(kind):
    torch.set_num_threads(1)
    with threadpool_limits(1), PeakMemory() as memory:
        archives = load_training_archives(DATA / "manifest.json", kind)
        report = train(archives, MODEL_ROOT / kind, config=CONFIG, kernel_type=SharedPrivateKernel)
        base, _ = load_model(
            MODEL_ROOT / kind / "target_kernel.pt", kernel_type=SharedPrivateKernel
        )
        encoder, _ = load_model(
            MODEL_ROOT / kind / "historical_encoder.pt", kernel_type=SharedPrivateKernel
        )
        task = next(t for t in task_inputs() if t["scenario"] == kind)
        original = read_json(ROOT / task["joint"])
        reference = original["specification"]["source_snapshot"]
        candidates = [Scenario(**s) for s in original["specification"]["candidates"]]
        source = build_source(
            encoder,
            archives,
            candidates,
            view_record_ids=reference["view_record_ids"],
            cache_path=source_destination(kind),
        )
        if source["specification"]["view_record_ids"] != reference["view_record_ids"]:
            raise ValueError("E2 changed the historical calculation record IDs")
        versions = {"target": model_version(base), "encoder": model_version(encoder)}
    write_json(
        MODEL_ROOT / kind / "resources.json",
        {**memory.result, "versions": versions, "training": report},
    )


def _build_worker(kind, connection):
    try:
        train_scenario(kind)
        connection.send({"status": "complete", "scenario": kind})
    except Exception:
        connection.send({"status": "failed", "scenario": kind, "error": traceback.format_exc()})
    finally:
        connection.close()


def build_models():
    context = mp.get_context("spawn")
    jobs, completed = [], []
    try:
        for kind in TEST_SCENARIOS:
            parent, child = context.Pipe(duplex=False)
            process = context.Process(target=_build_worker, args=(kind, child))
            process.start()
            child.close()
            jobs.append(
                {
                    "kind": kind,
                    "process": process,
                    "connection": parent,
                    "started": time.perf_counter(),
                    "memory": psutil.Process(process.pid),
                }
            )
        write_json(
            OUTPUT / "e2/build_progress.json",
            {
                "status": "running",
                "workers": {job["kind"]: job["process"].pid for job in jobs},
            },
        )
        while jobs:
            for job in list(jobs):
                process, connection = job["process"], job["connection"]
                if connection.poll():
                    message = connection.recv()
                    process.join(timeout=5)
                    connection.close()
                    if message["status"] != "complete":
                        raise RuntimeError(message["error"])
                    completed.append(job["kind"])
                    jobs.remove(job)
                    print("E2 model complete", job["kind"], flush=True)
                    continue
                if not process.is_alive():
                    raise RuntimeError(f"E2 {job['kind']} builder exited with {process.exitcode}")
                if time.perf_counter() - job["started"] > E2_CONFIGURATION["max_build_wall_s"]:
                    raise TimeoutError(f"E2 {job['kind']} exceeded its registered build wall limit")
                if job["memory"].memory_info().rss > E2_CONFIGURATION["max_build_cpu_gib"] * 2**30:
                    raise MemoryError(
                        f"E2 {job['kind']} exceeded its registered build memory limit"
                    )
            if jobs:
                time.sleep(0.2)
        write_json(
            OUTPUT / "e2/build_progress.json", {"status": "complete", "scenarios": completed}
        )
    except Exception as error:
        write_json(
            OUTPUT / "e2/build_failure.json",
            {
                "error": repr(error),
                "completed_scenarios": completed,
                "physical_executions": 0,
            },
        )
        raise
    finally:
        for job in jobs:
            job["connection"].close()
            if job["process"].is_alive():
                job["process"].terminate()
            job["process"].join(timeout=5)


def replay_task(task, phase):
    torch.set_num_threads(1)
    original = read_json(ROOT / task["joint"])
    common = read_json(ROOT / task["common_feedback"])
    kind, sut = task["scenario"], task["sut_id"]
    base, _ = load_model(MODEL_ROOT / kind / "target_kernel.pt", kernel_type=SharedPrivateKernel)
    source = read_source(source_destination(kind))
    spec = {
        "platform": "HighwayEnv",
        "stage": "method exploration E2",
        "sut_id": sut,
        "scenario": kind,
        "phase": phase,
        "configuration": E2_CONFIGURATION,
        "target_model_version": model_version(base),
        "source_snapshot": source["specification"],
        "reference_joint": task["joint"],
        "reference_common": task["common_feedback"],
        "new_physical_executions": 0,
    }
    path = OUTPUT / "e2" / phase / sut / f"{kind}.json.gz"
    if path.exists():
        saved = read_json(path)
        if saved["specification"] != spec:
            raise ValueError("Saved E2 replay has another frozen configuration")
        return str(path.relative_to(OUTPUT))
    bank = ExecutionBank(DATA / "targets/fixed" / sut / kind)
    if [asdict(s) for s in bank.candidates] != original["specification"]["candidates"]:
        raise ValueError("E2 must preserve the original candidate coordinates")
    ledger = QueryLedger(bank.candidates, 200, bank, joint_feedback=True)
    public = {
        "candidates": bank.candidates,
        "pretrained": base,
        "sources": source,
        "initial_indices": original["result"]["selected_indices"][:20],
        "budget": 200,
        "snapshot_counts": CHECKPOINTS,
        "query_order": common["result"]["selected_indices"] if phase == "common" else None,
        "diagnose_rss": phase == "common",
        "batch_size": 128,
        "device": "cpu",
    }
    try:
        result = run_isolated(
            select,
            public,
            ledger,
            wall_limit_s=E2_CONFIGURATION["max_task_wall_s"],
            memory_limit_bytes=E2_CONFIGURATION["max_task_cpu_gib"] * 2**30,
        )
        labels = original["evaluation"]["candidate_collisions"]
        from .failure_metrics import summarize

        metrics = summarize(
            bank.candidates,
            labels,
            result["queries"],
            ledger.trajectories,
            initial_count=20,
            query_ledger=ledger.records,
        )
        for snapshot in result["prediction_snapshots"]:
            snapshot["evaluation"] = prediction_metrics(snapshot, labels)
            if phase == "common":
                counterfactual = {
                    **snapshot,
                    "probabilities": snapshot["collision_conditioned_probabilities"],
                }
                snapshot["collision_conditioned_evaluation"] = prediction_metrics(
                    counterfactual, labels
                )
        write_json(
            path,
            {
                "specification": spec,
                "result": compact_result(result, bank.root),
                "metrics": metrics,
            },
        )
    except Exception as error:
        write_json(
            path.with_name(f"{kind}_failure.json"),
            {
                "specification": spec,
                "error": repr(error),
                "completed_queries": len(ledger.records),
                "physical_executions": 0,
            },
        )
        raise
    return str(path.relative_to(OUTPUT))


def run():
    register_e2()
    build_progress = OUTPUT / "e2/build_progress.json"
    if not build_progress.exists():
        build_models()
    elif read_json(build_progress)["status"] != "complete":
        raise RuntimeError(
            "E2 construction is already registered; inspect its live workers "
            "and saved evidence before any restart"
        )
    completed = []
    for phase in ("common", "adaptive"):
        with ProcessPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(replay_task, task, phase) for task in task_inputs()]
            for future in as_completed(pending):
                path = future.result()
                completed.append(path)
                write_json(
                    OUTPUT / "e2/progress.json",
                    {
                        "status": "running",
                        "phase": phase,
                        "completed": completed,
                    },
                )
                print("E2", phase, len(completed), path, flush=True)
    write_json(OUTPUT / "e2/progress.json", {"status": "complete", "completed": completed})


if __name__ == "__main__":
    run()
