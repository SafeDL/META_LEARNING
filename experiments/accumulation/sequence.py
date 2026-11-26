"""Chronological paid-feedback execution and model accumulation core.

This module implements task execution, archives, fixed-reference reuse and model
publication. Formal use requires a separately registered and admitted study.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.training import DEFAULT as TRAINING_CONFIG

from experiments import transfer_execution as execution
from experiments.costs import PeakMemory
from experiments.history import build_source, completed_archives, finalize_task, model_version, update_shared_model
from experiments.history_transfer_experiment import freeze
from experiments.query import QueryLedger
from experiments.resume import ReplayInterrupted
from experiments.storage import read_json, write_json

REGIMES = ("fixed_history", "append_sources", "update_shared")


class SequenceInterrupted(RuntimeError):
    """A test interruption between fully persisted tasks and model publications."""


def directory(root, sequence):
    return (
        Path(root)
        / "accumulation"
        / sequence["scenario"]
        / str(sequence["seed"])
        / Path(*sequence.get("namespace", []))
        / sequence["regime"]
    )


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _past(paths, sequence, ordinal, suts):
    allowed = {t["sut_id"]: t for t in sequence["tasks"][: ordinal - 1]}
    for path in paths:
        archive = read_json(path)
        task = allowed.get(archive["sut_id"])
        if (
            task is None
            or archive["branch_id"] != sequence["branch_id"]
            or archive["completed_order"] != task["ordinal"]
            or archive["sut_version"] != suts[archive["sut_id"]].version
            or len(archive["executions"]) > task["total_budget"]
        ):
            raise ValueError("Accumulation archive is outside this branch's actual past tasks")
    return completed_archives(
        paths,
        before_order=ordinal,
        branch_id=sequence["branch_id"],
        current_system=(
            sequence["tasks"][ordinal - 1]["sut_id"],
            suts[sequence["tasks"][ordinal - 1]["sut_id"]].version,
        ),
    )


def _archive(path, saved, bank, sut, sequence, task, models, scope):
    def forbidden(*args):
        raise AssertionError("Archiving may read paid responses, never request a new query")

    ledger = QueryLedger(
        bank.candidates,
        task["total_budget"],
        forbidden,
        joint_feedback=True,
        domain=getattr(bank, "domain", None),
        allow_registration=False,
    )
    ledger.records = saved["query_ledger"]
    successful = [row for row in ledger.records if row["status"] == "complete"]
    if not successful:
        return None
    ledger.trajectories = [bank(bank.candidates[row["index"]]).trajectory for row in successful]
    finalize_task(
        path,
        ledger,
        task_id=f"{sequence['branch_id']}:{task['ordinal']}:{sut.sut_id}",
        sut_id=sut.sut_id,
        sut_version=sut.version,
        family=sut.family,
        lineage=sut.lineage,
        environment_seed=bank.manifest["environment_seed"],
        completed_order=task["ordinal"],
        branch_id=sequence["branch_id"],
        original_role=sut.role,
        execution_contract=bank.manifest["contract"],
        model_snapshot={"base": model_version(models[0]), "encoder": model_version(models[1])},
        domain=getattr(bank, "domain", None),
        execution_bank=bank if scope == "formal" else None,
    )
    return path


def _reuse_main(main_root, output, case):
    """The fixed-history branch is exactly the already executed main Ours task."""
    source, source_paid = execution._paths(main_root, case)
    approval = read_json(execution.approval_path(main_root, case))
    if approval != {"key": list(case.key), "scope": "formal", "specification": case.specification}:
        raise ValueError("Fixed-history reference differs from the registered main Ours task")
    result_path, paid_path = execution._paths(output, case)
    main_failure = sorted(source.parent.glob("failed_attempts_*.json"))
    selected = source if source.exists() else main_failure[-1] if main_failure else None
    if selected is None:
        raise ValueError("Fixed-history reuse requires a terminal main result or failure")
    saved = read_json(selected)
    if saved["specification"] != case.specification:
        raise ValueError("Main reference uses another model, history view or feedback rule")
    copies = [(selected, result_path if source.exists() else result_path.parent / selected.name)]
    if source_paid.exists():
        copies.append((source_paid, paid_path))
    copies.extend(
        (p, result_path.parent / "attempts" / p.name)
        for p in sorted((source.parent / "attempts").glob(f"{case.method}_*.json"))
    )
    for original, destination in copies:
        if destination.exists():
            if _sha(destination) != _sha(original):
                raise ValueError("Existing fixed-history reference differs from its main origin")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
    freeze(
        result_path.parent / "main_reuse.json",
        {
            "specification": case.specification,
            "source_result": str(selected),
            "source_sha256": _sha(selected),
            "scope": "Same logical run reused as a fixed-history reference; no new selector execution or physical query. Source preparation and archive I/O are separate.",
        },
    )
    return saved, not source.exists()


def _prior(output, models, allowed, view_ids, candidates):
    existing = set((output / "sources").glob("*.npz"))
    with PeakMemory() as resources:
        prior = build_source(
            models[1],
            allowed,
            candidates,
            cache_directory=output / "sources",
            view_record_ids=view_ids,
        )
    key = hashlib.sha256(json.dumps(prior["specification"], sort_keys=True).encode()).hexdigest()
    receipt = output / "sources" / f"{key}.construction.json"
    if not receipt.exists():
        write_json(
            receipt,
            {
                "specification": prior["specification"],
                "resources": resources.result,
                "scope": (
                    "cache_load"
                    if output / "sources" / f"{key}.npz" in existing
                    else "source_construction"
                ),
            },
        )
    elif read_json(receipt)["specification"] != prior["specification"]:
        raise ValueError("Sequence source construction changed")
    return prior, receipt


def _reuse_prefix(origin, output, case, prior, bank):
    """Reuse an eligible paid path while preserving this branch's archive identity."""
    result_path, paid_path = execution._paths(origin, case)
    failed = not result_path.exists()
    if failed:
        failures = sorted(result_path.parent.glob("failed_attempts_*.json"))
        if not failures:
            raise ValueError("Shared prefix has no terminal path")
        result_path = failures[-1]
    saved = read_json(result_path)
    recorded = saved["specification"]
    if read_json(execution.approval_path(origin, case)) != {
        "key": list(case.key), "scope": "formal", "specification": recorded
    }:
        raise ValueError("Shared prefix is missing its original approval")
    source = case.specification["source_snapshot"]
    old_source = recorded["source_snapshot"]
    if recorded != {**case.specification, "source_snapshot": old_source}:
        raise ValueError("Shared prefix changed the target, candidate bank or selector")
    if source != old_source:
        # Independent branch archives may assign different record identifiers to
        # the same feedback. Require identical numerical priors in source order.
        provenance = {"archives", "record_ids", "view_record_ids"}
        if {k: v for k, v in source.items() if k not in provenance} != {
            k: v for k, v in old_source.items() if k not in provenance
        }:
            raise ValueError("Shared prefix changed the prior construction or source views")
        key = hashlib.sha256(json.dumps(old_source, sort_keys=True).encode()).hexdigest()
        with np.load(origin / "sources" / f"{key}.npz", allow_pickle=False) as old:
            if (
                json.loads(str(old["specification"])) != old_source
                or old["names"].tolist() != prior["names"]
                or not all(np.array_equal(old[name], prior[name])
                           for name in ("means", "blocks", "weights"))
            ):
                raise ValueError("Shared prefix changed the source posterior")
    records = saved["query_ledger"]
    if not failed:
        execution._validate_output(case, saved["result"], records)
        ledger = read_json(paid_path)
        if ledger["status"] != "complete" or ledger["query_ledger"] != records:
            raise ValueError("Shared prefix differs from its terminal paid ledger")
    for row in records:
        scene = bank.candidates[row["index"]]
        if row["scenario_id"] != scene.scenario_id:
            raise ValueError("Shared prefix changed the preassigned candidate")
    freeze(execution._paths(output, case)[0].parent / "prefix_reuse.json", {
        "origin": str(result_path),
        "sha256": _sha(result_path),
        "paid_queries": len(records),
    })
    return saved, failed, result_path


def execute_sequence(
    root,
    sequence,
    context,
    *,
    settings,
    limits,
    scope,
    main_root=None,
    fixed_reference=None,
    training_config=TRAINING_CONFIG,
    stop_after_tasks=None,
    reuse_from=None,
):
    """Run an already admitted branch, rebuilding only its own past on recovery.

    context supplies initial models/history/view IDs, target banks and registered
    SUT identities. The outer formal driver owns full study/scale admission.
    """
    if sequence["regime"] not in REGIMES or scope not in ("formal", "development"):
        raise ValueError("Unknown chronological regime or execution scope")
    fixed_reference = fixed_reference or ("reuse_main" if scope == "formal" else "execute")
    if fixed_reference not in ("reuse_main", "execute"):
        raise ValueError("Unknown fixed-history reference policy")
    reuse_main = (
        scope == "formal"
        and sequence["regime"] == "fixed_history"
        and fixed_reference == "reuse_main"
    )
    if reuse_main and main_root is None:
        raise ValueError("Formal fixed-history reference requires the admitted main root")
    if scope == "formal" and training_config != TRAINING_CONFIG:
        raise ValueError("Formal accumulation requires the registered full shared-training config")
    order = sequence["tasks"]
    if (
        [t["ordinal"] for t in order] != list(range(1, len(order) + 1))
        or len({t["sut_id"] for t in order}) != len(order)
        or any(
            t["sut_id"] not in context["banks"] or t["sut_id"] not in context["suts"] for t in order
        )
    ):
        raise ValueError("Sequence tasks must be unique, ordered, and backed by admitted banks")
    output = directory(root, sequence)
    models = context["models"]
    initial = context["history"]
    initial_ids = context["view_record_ids"]
    freeze(
        output / "protocol.json",
        {
            "sequence": sequence,
            "scope": scope,
            "settings": settings.to_dict(),
            "execution_contract": execution.execution_contract(limits),
            "initial_model_versions": [model_version(m) for m in models],
            "initial_source_ids": [a.source_id for a in initial],
            "initial_view_record_ids": initial_ids,
            "initial_history_fingerprints": archive_fingerprints(initial),
            "training_config": execution.canonical(training_config.__dict__),
            "driver_sha256": _sha(__file__),
            "fixed_reference": (
                "reuse exact main Ours run" if reuse_main else "independent branch execution"
            ),
        },
    )
    if reuse_from and scope != "formal":
        raise ValueError("Prefix reuse requires individually approved formal tasks")
    completed, rows, updates, learned_ids = [], [], [], set()
    for task in order:
        ordinal, sut_id = task["ordinal"], task["sut_id"]
        sut, bank = context["suts"][sut_id], context["banks"][sut_id]
        past = (
            []
            if sequence["regime"] == "fixed_history"
            else _past(completed, sequence, ordinal, context["suts"])
        )
        allowed = [*initial, *past]
        view_ids = {**initial_ids, **{a.source_id: a.provenance["record_ids"] for a in past}}
        prior, receipt = _prior(output, models, allowed, view_ids, bank.candidates)
        case = execution.budget_case(
            bank,
            task["initialization_seed"],
            "full",
            models,
            prior,
            settings=settings,
            initial_count=task["initial_queries"],
            budget=task["total_budget"],
            limits=limits,
        )
        freeze(
            output / "task_inputs" / f"{ordinal:02d}_{sut_id}.json",
            {
                "task": task,
                "specification": case.specification,
                "available_completed_archives": (
                    [
                        {"path": p.relative_to(output).as_posix(), "sha256": _sha(p)}
                        for p in completed
                    ]
                    if sequence["regime"] != "fixed_history"
                    else []
                ),
                "actual_source_ids": [a.source_id for a in allowed],
                "model_versions": [model_version(m) for m in models],
            },
        )
        execution.declare(output, case, scope=scope)
        failed = False
        result_path = execution._paths(output, case)[0]
        origin = (reuse_from or {}).get(ordinal)
        if origin is not None:
            saved, failed, result_path = _reuse_prefix(origin, output, case, prior, bank)
        elif reuse_main:
            saved, failed = _reuse_main(main_root, output, case)
        else:
            failures = sorted(result_path.parent.glob("failed_attempts_*.json"))
            attempts = sorted((result_path.parent / "attempts").glob(f"{case.method}_*.json"))
            terminal_failure = (
                not result_path.exists()
                and failures
                and attempts
                and read_json(attempts[-1]).get("status") == "failed"
            )
            if terminal_failure:
                saved = read_json(failures[-1])
                result_path = failures[-1]
                if saved["specification"] != case.specification:
                    raise ValueError("Terminal sequence failure belongs to another task")
                failed = True
            else:
                try:
                    saved = execution.run_budget(
                        output,
                        case,
                        bank,
                        models,
                        prior,
                        settings=settings,
                        paid_execution=lambda index: bank(bank.candidates[index]),
                        scope=scope,
                        construction_resources={
                            "shared_receipt": receipt.relative_to(output).as_posix(),
                            "scope": "sequence_source_constructor",
                        },
                    )
                except ReplayInterrupted:
                    raise
                except Exception:
                    path = execution._paths(output, case)[0]
                    failures = sorted(path.parent.glob("failed_attempts_*.json"))
                    if not failures:
                        raise
                    saved = read_json(failures[-1])
                    result_path = failures[-1]
                    if saved["specification"] != case.specification:
                        raise ValueError("Sequence failure belongs to another task")
                    failed = True
        archive = _archive(
            output / "completed" / f"{ordinal:02d}_{sut_id}.json.gz",
            saved,
            bank,
            sut,
            sequence,
            task,
            models,
            scope,
        )
        if archive is not None:
            completed.append(archive)
        rows.append(
            {
                "ordinal": ordinal,
                "sut_id": sut_id,
                "status": "failed" if failed else "complete",
                "successful_paid_feedback": sum(
                    r["status"] == "complete" for r in saved["query_ledger"]
                ),
                "actual_source_count": len(allowed),
                "source_ids": [a.source_id for a in allowed],
                "model_versions": [model_version(m) for m in models],
                "archive": None if archive is None else archive.relative_to(output).as_posix(),
                "reused_main_run": reuse_main,
                "result_path": str(result_path),
                "reused_prefix": origin is not None,
            }
        )
        if task["update_after_task"]:
            available = completed_archives(
                completed, before_order=ordinal + 1, branch_id=sequence["branch_id"]
            )
            # One paid response still defines a source posterior, but the
            # registered next-response trainer needs at least two records.
            new_ids = [
                a.source_id
                for a in available
                if len(a.scenarios) >= 2 and a.source_id not in learned_ids
            ]
            update_root = output / "models" / f"after_{ordinal:02d}_tasks"
            if new_ids:
                with PeakMemory() as update_cost:
                    models, publication = update_shared_model(
                        update_root,
                        [*initial, *available],
                        models,
                        new_source_ids=new_ids,
                        config=training_config,
                    )
                if publication["accepted"]:
                    learned_ids.update(new_ids)
                cost_path = update_root / "execution_cost.json"
                if not cost_path.exists():
                    write_json(cost_path, update_cost.result)
            else:
                publication = {"accepted": False, "reason": "no_new_trainable_source"}
            updates.append(
                {
                    "after_task": ordinal,
                    "publication": publication,
                    "learned_source_ids": sorted(learned_ids),
                    "model_versions": [model_version(m) for m in models],
                }
            )
            freeze(update_root / "sequence_event.json", updates[-1])
        progress = {
            "sequence": sequence,
            "completed": rows,
            "updates": updates,
            "status": "complete" if ordinal == len(order) else "running",
            "new_physical_executions": 0,
        }
        write_json(output / "progress.json", progress)
        if stop_after_tasks == ordinal and ordinal < len(order):
            raise SequenceInterrupted(f"Interrupted after {ordinal} persisted target tasks")
    return progress
