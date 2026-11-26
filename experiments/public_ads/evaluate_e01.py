"""Isolated formal E01 extension for three admitted public-source ADSs."""

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import torch

from controllers.highway.public_agents import create, runtime_contract
from simulator.highway.collision_env import execute

from experiments.execution_events import analyze_events
from experiments.executions import ExecutionBank
from experiments.history_transfer_experiment import OUTPUT as ORIGINAL, registered_populations
from experiments.k_shot import CHECKPOINTS, split_candidates
from experiments.protocol import ROOT, episode_duration
from experiments.public_ads.qualify import verify_admitted
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.storage import compact_feedback, compact_result, read_json, write_json

OUTPUT = ROOT / "results/public_ads"
TARGETS = ("online_mc_ppo", "social_attention_ddqn", "decision_transformer")
ENVIRONMENT_SEED = 81001


def _initialize():
    torch.set_num_threads(1)


def _execute_one(sut_id, scenario, index, destination):
    started = time.perf_counter()
    policy = create(sut_id)
    result = execute(
        policy, scenario, environment_seed=ENVIRONMENT_SEED,
        duration_s=episode_duration(scenario.kind),
    )
    payload = {
        "call": index + 1,
        "scenario": asdict(scenario),
        "feedback": result.feedback.to_dict(),
        "trajectory": result.trajectory,
        "events": analyze_events(scenario, result.trajectory),
    }
    path = Path(destination) / "executions" / f"{index:06d}.json.gz"
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False)
    temporary.replace(path)
    return {"index": index, "wall_s": time.perf_counter() - started,
            "execution_s": result.feedback.execution_s,
            "collision": bool(result.feedback.collision)}


def _specification(sut_id, scenarios):
    acceptance = verify_admitted(sut_id)
    return {
        "format": "collision-execution-bank",
        "sut_id": sut_id,
        "environment_seed": ENVIRONMENT_SEED,
        "qualification": f"results/public_ads/qualification/{sut_id}/acceptance.json",
        "contract": {
            "public_ads": runtime_contract(sut_id),
            "acceptance_sha256": hashlib.sha256(json.dumps(
                acceptance, sort_keys=True, allow_nan=False
            ).encode()).hexdigest(),
            "scenario_kind": scenarios[0].kind,
            "episode_duration_s": episode_duration(scenarios[0].kind),
        },
        "scenarios": [asdict(s) for s in scenarios],
    }


def collect_bank(sut_id, scenarios, destination, *, workers=8):
    if sut_id not in TARGETS or len(scenarios) != 2000:
        raise ValueError("Public extension requires a registered ADS and 2,000 target points")
    destination = Path(destination)
    (destination / "executions").mkdir(parents=True, exist_ok=True)
    specification = _specification(sut_id, scenarios)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(old.get(key) != value for key, value in specification.items()):
            raise ValueError("Existing public ADS bank has a different frozen identity")
    pending = [i for i in range(len(scenarios)) if not (
        destination / "executions" / f"{i:06d}.json.gz"
    ).exists()]
    if not pending and manifest_path.exists() and old.get("status") == "complete":
        return ExecutionBank(destination)
    write_json(manifest_path, {**specification, "status": "running"})
    ledger = destination / "calls.jsonl"
    started = time.perf_counter()
    with ledger.open("a", encoding="utf-8", buffering=1) as stream:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_initialize
        ) as pool:
            tasks = {
                pool.submit(_execute_one, sut_id, scenarios[i], i, destination): i
                for i in pending
            }
            errors = []
            for completed, future in enumerate(as_completed(tasks), 1):
                index = tasks[future]
                try:
                    row = future.result()
                    stream.write(json.dumps({"status": "complete", **row}) + "\n")
                except Exception as error:
                    errors.append({"index": index, "error": repr(error)})
                    stream.write(json.dumps({"status": "failed", **errors[-1]}) + "\n")
                if completed % 100 == 0 or completed == len(pending):
                    print(sut_id, scenarios[0].kind, completed, "/", len(pending), flush=True)
    if errors:
        write_json(manifest_path, {**specification, "status": "failed", "errors": errors})
        raise RuntimeError(f"{len(errors)} public ADS physical executions failed")
    write_json(manifest_path, {
        **specification, "status": "complete", "candidate_count": len(scenarios),
        "last_collection_wall_s": time.perf_counter() - started,
    })
    return ExecutionBank(destination)


def collect(*, targets=TARGETS, kinds=FORMAL_SCENARIOS, workers=8):
    populations = registered_populations(ORIGINAL)
    if set(kinds) - set(FORMAL_SCENARIOS):
        raise ValueError("Unregistered public ADS scenario")
    for sut_id in targets:
        if sut_id not in TARGETS:
            raise ValueError(f"Unregistered public ADS: {sut_id}")
        verify_admitted(sut_id)
        for kind in kinds:
            scenarios = populations["target", kind]
            collect_bank(sut_id, scenarios, OUTPUT / "physical/target" / kind / sut_id,
                         workers=workers)


def _run_prediction_case(case, bank, models, pooled, sources, settings):
    """Use the frozen E01 selector with a separate admission for new targets."""
    from testing_methods.baselines.response_baselines.pooled_gp import select as pooled_select
    from testing_methods.history_transfer.selection import select
    from testing_methods.history_transfer.shared_private import SharedPrivateKernel

    from experiments import transfer_execution as execution
    from experiments.query import run_isolated
    from experiments.resume import ResumableLedger

    approval = read_json(execution.approval_path(OUTPUT, case))
    if approval != {"key": list(case.key), "specification": case.specification,
                    "scope": "public_extension"}:
        raise ValueError("Public extension task was not admitted before querying")
    path, checkpoint = execution._paths(OUTPUT, case)
    spec = case.specification
    ledger = ResumableLedger(
        bank.candidates, len(spec["split"]["query_order"]), bank,
        checkpoint_path=checkpoint, specification=spec,
        paid_execution=lambda index: bank(bank.candidates[index]),
        joint_feedback=True, domain=bank.domain, allow_registration=False,
    )
    if path.exists():
        saved = read_json(path)
        if (saved["specification"] != spec or
                saved["query_ledger"] != execution.canonical(compact_feedback(ledger.records)) or
                read_json(checkpoint)["status"] != "complete"):
            raise ValueError("Completed extension result changed")
        execution._validate_output(case, saved["result"], ledger.records)
        return saved
    public = {
        "candidates": bank.candidates,
        "query_order": spec["split"]["query_order"],
        "budget": len(spec["split"]["query_order"]),
        "snapshot_counts": spec["checkpoints"],
        "batch_size": 128,
        "device": spec["device"],
    }
    if case.method == "pooled_history":
        selector = pooled_select
        public.update(pretrained=models[0], history=pooled,
                      inference_backend="grouped", fit_settings=settings.pooled)
    else:
        selector = select
        public.update(initial_indices=spec["split"]["query_order"][:1],
                      feedback_mode="joint", pretrained=models[0], sources=sources,
                      online_settings=settings.online,
                      update_source_weights=spec["update_source_weights"])
        if case.method == "target_only":
            public["raw_kernel_type"] = SharedPrivateKernel
    started = time.perf_counter()
    try:
        result = run_isolated(
            selector, public, ledger,
            wall_limit_s=spec["execution_contract"]["limits"]["wall_s"],
            memory_limit_bytes=spec["execution_contract"]["limits"]["child_memory_bytes"],
        )
        execution._validate_output(case, result, ledger.records)
        ledger.finish()
        saved = execution._finalize(
            case, bank, result, ledger, scope="formal", construction_resources=None
        )
        saved["wall_s"] = time.perf_counter() - started
        write_json(path, saved)
        return saved
    except Exception as error:
        write_json(path.with_suffix(".failure.json"), {
            "specification": spec, "error": repr(error),
            "query_ledger": compact_feedback(ledger.records),
        })
        raise


def prepare_history(kind):
    from testing_methods.common.response import archive_fingerprints
    from testing_methods.history_transfer.shared_private import SharedPrivateKernel
    from testing_methods.history_transfer.training import validate_model

    from experiments.history import build_source
    from experiments.history_transfer_experiment import pooled_history, registered_history_views, scenario_history

    history = scenario_history(ORIGINAL, kind)
    model_root = ORIGINAL / "models/adaptive_mixture" / kind
    models = []
    for name, part in (("target_kernel.pt", "target"),
                       ("historical_encoder.pt", "historical_encoder")):
        saved = torch.load(model_root / name, weights_only=True, map_location="cpu")
        spec = saved["specification"]
        if (saved["format"] != "sequential-joint-collision-rss" or saved["part"] != part
                or saved["step"] != spec["config"]["steps"]
                or spec["scenario"] != kind
                or spec["history_fingerprints"] != archive_fingerprints(history)
                or read_json((model_root / name).with_suffix(".json"))["specification"] != spec):
            raise ValueError("Published historical weights differ from their training evidence")
        model = SharedPrivateKernel(saved["dimension"]).double()
        model.load_state_dict(saved["state_dict"], strict=True)
        model.construction_specification = spec
        validate_model(model)
        models.append(model.eval())
    models = tuple(models)
    views, view = registered_history_views(ORIGINAL, kind, history)
    population = registered_populations(ORIGINAL)["target", kind]
    sources = build_source(
        models[1], history, population,
        cache_directory=OUTPUT / "sources" / kind,
        view_record_ids=view["selected_record_ids"],
    )
    return {"models": models, "sources": sources,
            "pooled": pooled_history(views, view)}


def evaluate(*, targets=TARGETS, kinds=FORMAL_SCENARIOS, methods=(
    "target_only", "pooled_history", "full"
)):
    from experiments import transfer_execution as execution
    from experiments.history_transfer_experiment import freeze
    from experiments.transfer_admission import LIMITS
    from experiments.transfer_settings import native_settings

    if set(targets) - set(TARGETS) or set(kinds) - set(FORMAL_SCENARIOS) or set(methods) - {
        "target_only", "pooled_history", "full"
    }:
        raise ValueError("Unregistered extension evaluation task")
    settings = native_settings()
    limits = dict(LIMITS)
    freeze(OUTPUT / "reports/protocol.json", {
        "scope": "public_e01_extension", "targets": list(TARGETS),
        "scenarios": list(FORMAL_SCENARIOS),
        "methods": ["target_only", "pooled_history", "full"],
        "checkpoints": list(CHECKPOINTS), "feedback_seed": 11,
        "settings": settings.to_dict(), "limits": limits,
        "original_study": ORIGINAL.relative_to(ROOT).as_posix(),
    })
    for kind in kinds:
        context = prepare_history(kind)
        for sut_id in targets:
            verify_admitted(sut_id)
            bank = ExecutionBank(OUTPUT / "physical/target" / kind / sut_id)
            if (bank.candidates != registered_populations(ORIGINAL)["target", kind]
                    or bank.manifest["contract"] != _specification(sut_id, bank.candidates)["contract"]):
                raise ValueError("Extension bank differs from frozen target inputs")
            for method in methods:
                models = context["models"] if method != "target_only" else (None, None)
                pooled = context["pooled"] if method == "pooled_history" else None
                sources = context["sources"] if method == "full" else None
                case = execution.prediction_case(
                    bank, split_candidates(bank.candidates, 11), 11, method,
                    models, pooled, sources, settings=settings,
                    checkpoints=CHECKPOINTS, limits=limits,
                )
                execution.declare(OUTPUT, case, scope="public_extension")
                _run_prediction_case(case, bank, models, pooled, sources, settings)
                print("complete", sut_id, kind, method, flush=True)


def replay_control():
    """Recompute one sealed original E01 target with the extension's current code."""
    from experiments import transfer_execution as execution
    from experiments.transfer_admission import LIMITS
    from experiments.transfer_settings import native_settings
    import numpy as np

    kind, sut_id = FORMAL_SCENARIOS[0], "target_ppo"
    settings = native_settings()
    context = prepare_history(kind)
    bank = ExecutionBank(ORIGINAL / "physical/target" / kind / sut_id)
    rows = []
    for method in ("target_only", "pooled_history", "full"):
        models = context["models"] if method != "target_only" else (None, None)
        pooled = context["pooled"] if method == "pooled_history" else None
        sources = context["sources"] if method == "full" else None
        case = execution.prediction_case(
            bank, split_candidates(bank.candidates, 11), 11, method,
            models, pooled, sources, settings=settings,
            checkpoints=CHECKPOINTS, limits=dict(LIMITS),
        )
        execution.declare(OUTPUT, case, scope="public_extension")
        current = _run_prediction_case(case, bank, models, pooled, sources, settings)
        old = read_json(ORIGINAL / "k_shot" / sut_id / kind / "11" / f"{method}.json.gz")
        rows.append({
            "method": method, "same_query_order": (
                [r["index"] for r in current["query_ledger"]] ==
                [r["index"] for r in old["query_ledger"]]
            ),
            "prediction_max_abs_error": max(
                float(np.max(np.abs(np.asarray(a["probabilities"]) -
                                    np.asarray(b["probabilities"]))))
                for a, b in zip(current["result"]["prediction_snapshots"],
                                old["result"]["prediction_snapshots"], strict=True)
            ),
            "metrics": [
                {"K": a["K"], "original_nll": b["nll"], "current_nll": a["nll"],
                 "original_recall_100": b["recall_100"],
                 "current_recall_100": a["recall_100"]}
                for a, b in zip(current["metrics"], old["metrics"], strict=True)
            ],
        })
        print("control_complete", method, flush=True)
    write_json(OUTPUT / "reports/original_task_replay.json", {
        "sut_id": sut_id, "scenario": kind, "rows": rows,
        "purpose": "Check numerical drift before comparing new targets with sealed original E01",
    })
    return rows


def _collision_labels(bank):
    labels = []
    for index, scene in enumerate(bank.candidates):
        with gzip.open(bank.execution_path(index), "rt", encoding="utf-8") as stream:
            row = json.load(stream)
        if row["scenario"] != asdict(scene) or row["feedback"]["scenario_id"] != scene.scenario_id:
            raise ValueError("Physical feedback differs from its fixed coordinate")
        labels.append(bool(row["feedback"]["collision"]))
    return labels


def report():
    from experiments.k_shot import split_candidates

    revised_dqn = ROOT / "results/dqn_continual_learning/k_shot_adaptation"
    rows = []
    for kind in FORMAL_SCENARIOS:
        references = {
            "ppo": ExecutionBank(ORIGINAL / "physical/target" / kind / "target_ppo"),
            "dqn": ExecutionBank(revised_dqn / "physical/target" / kind / "target_dqn"),
        }
        reference_labels = {name: _collision_labels(bank)
                            for name, bank in references.items()}
        for sut_id in TARGETS:
            path = OUTPUT / "physical/target" / kind / sut_id
            if not (path / "manifest.json").exists() or read_json(path / "manifest.json")["status"] != "complete":
                continue
            bank = ExecutionBank(path)
            if any(other.candidates != bank.candidates for other in references.values()):
                raise ValueError("Original and new target coordinates do not match")
            labels = _collision_labels(bank)
            split = split_candidates(bank.candidates, 11)
            evaluation = split["evaluation_indices"]
            calls = [json.loads(line) for line in (path / "calls.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()]
            completed_calls = {entry["index"]: entry for entry in calls
                               if entry["status"] == "complete"}
            if len(completed_calls) != len(labels):
                raise ValueError("Public physical call ledger is incomplete")
            row = {
                "sut_id": sut_id, "scenario": kind, "candidates": len(labels),
                "collisions": sum(labels),
                "physical_execution_s": sum(entry["execution_s"] for entry in
                                            completed_calls.values()),
                "last_collection_wall_s": bank.manifest["last_collection_wall_s"],
                "evaluation_count": len(evaluation),
                "evaluation_collisions": sum(labels[i] for i in evaluation),
                "recall_100_ceiling": (
                    min(100, sum(labels[i] for i in evaluation)) /
                    sum(labels[i] for i in evaluation)
                    if any(labels[i] for i in evaluation) else None
                ),
                "paired_collisions": {}, "methods": {},
            }
            for name, other in reference_labels.items():
                row["paired_collisions"][name] = {
                    "both": sum(a and b for a, b in zip(labels, other)),
                    "only_public": sum(a and not b for a, b in zip(labels, other)),
                    "only_reference": sum(not a and b for a, b in zip(labels, other)),
                    "both_safe": sum(not a and not b for a, b in zip(labels, other)),
                }
            for method in ("target_only", "pooled_history", "full"):
                result = OUTPUT / "k_shot" / sut_id / kind / "11" / f"{method}.json.gz"
                if result.exists():
                    saved = read_json(result)
                    row["methods"][method] = {
                        "status": "complete", "metrics": saved["metrics"],
                        "paid_queries": len(saved["query_ledger"]),
                    }
                else:
                    row["methods"][method] = {"status": "pending"}
            rows.append(row)
    summary = {"scope": "public_source_e01_extension", "completed_banks": len(rows),
               "expected_banks": len(TARGETS) * len(FORMAL_SCENARIOS), "tasks": rows}
    write_json(OUTPUT / "reports/summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "evaluate", "control", "report"))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    parser.add_argument("--kinds", nargs="+", choices=FORMAL_SCENARIOS,
                        default=FORMAL_SCENARIOS)
    parser.add_argument("--methods", nargs="+", choices=(
        "target_only", "pooled_history", "full"
    ), default=("target_only", "pooled_history", "full"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "collect":
        collect(targets=args.targets, kinds=args.kinds, workers=args.workers)
    elif args.stage == "report":
        summary = report()
        print("complete banks", summary["completed_banks"], "/", summary["expected_banks"])
    elif args.stage == "control":
        replay_control()
    else:
        evaluate(targets=args.targets, kinds=args.kinds, methods=args.methods)


if __name__ == "__main__":
    main()
