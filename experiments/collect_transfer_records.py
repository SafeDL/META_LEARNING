"""Read approved task specifications and independently audit saved result grids.

The caller supplies specifications from registered inputs, never metric rows.
Formal collection remains gated; development accepts only the old ADS cohort.
"""

import copy
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

from controllers.highway.catalog import FORMAL_TARGET_SUTS, INDEPENDENT_SUTS

from .executions import ExecutionBank
from .record_audit import (
    BankTruth,
    _canonical_feedback,
    _check_specification,
    budget_record,
    prediction_record,
)
from .resume import paid_feedback_hash, trajectory_hash
from .storage import compact_feedback, read_json, write_json

DEVELOPMENT_IDS = tuple(s.sut_id for s in INDEPENDENT_SUTS)


@dataclass(frozen=True)
class ApprovedCase:
    experiment: str
    sut_id: str
    scenario: str
    seed: int
    method: str
    specification: dict | None
    final_failure: bool = False

    @property
    def key(self):
        return self.experiment, self.sut_id, self.scenario, self.seed, self.method


def _files(case, root):
    if (
        type(case.seed) is not int
        or case.seed < 0
        or any(
            not isinstance(value, str)
            or value in ("", ".", "..")
            or any(c in value for c in "/\\:")
            for value in (case.sut_id, case.scenario, case.method)
        )
    ):
        raise ValueError("Registered case keys must be safe identifiers and an integer seed")
    if case.experiment == "prediction":
        directory = Path(root) / "k_shot" / case.sut_id / case.scenario / str(case.seed)
        result = directory / f"{case.method}.json.gz"
        failures = [result.with_suffix(".failure.json")]
        checkpoint = directory / f"{case.method}_paid_feedback.json.gz"
    elif case.experiment == "budget":
        directory = (
            Path(root) / "budget" / case.sut_id / case.scenario / str(case.seed) / case.method
        )
        result = directory / "result.json.gz"
        failures = sorted(directory.glob("failed_attempts_*.json"))
        checkpoint = directory / "paid_feedback.json.gz"
    else:
        raise ValueError("Unknown registered experiment")
    return result, [p for p in failures if p.exists()], checkpoint


def inventory(cases, root):
    if not cases or len({c.key for c in cases}) != len(cases):
        raise ValueError("A reader requires a nonempty unique registered case list")
    rows = []
    for case in cases:
        if type(case.final_failure) is not bool:
            raise ValueError("Final failure status must come from the terminal run manifest")
        result, failures, _ = _files(case, root)
        if result.exists() and case.final_failure:
            raise ValueError("Completed result conflicts with a finalized task failure")
        status = (
            "complete"
            if result.exists()
            else "failed" if failures and case.final_failure else "pending"
        )
        rows.append({"key": list(case.key), "status": status, "prior_failure_files": len(failures)})
    return rows


def _formal_cases(cases, root):
    from .history_transfer_experiment import (
        BUDGET_METHODS,
        PREDICTION_METHODS,
        SEEDS,
        scenarios_for,
    )

    modes = {c.experiment for c in cases}
    methods = {"prediction": PREDICTION_METHODS, "budget": BUDGET_METHODS}
    if not modes or not modes.issubset(methods):
        raise ValueError("Unknown formal report mode")
    expected = {
        (mode, sut.sut_id, scene, seed, method)
        for mode in modes
        for sut, scene, seed, method in product(
            FORMAL_TARGET_SUTS, scenarios_for(root), SEEDS, methods[mode]
        )
    }
    if {c.key for c in cases} != expected:
        raise ValueError("Formal reports require the complete registered ADS/method/seed cohort")


def _formal_context(root, cases):
    from . import transfer_settings
    from .method_contracts import method_contract
    from .baselines import BASELINES

    settings = transfer_settings.load_settings(root)
    context = {
        "settings": settings,
        "inputs": {},
        "method_contracts": {m: method_contract(m) for m in {c.method for c in cases}},
        "source_contract": method_contract("source_prior"),
        "baseline_configurations": {
            k: asdict(BASELINES[k].DEFAULT)
            for k in {c.method for c in cases}
            if k in BASELINES and k != "random"
        },
        "classifiers": {},
    }
    from .history import build_source, model_version
    from .history_transfer_experiment import (
        models_and_history,
        pooled_history,
        registered_history_views,
        registered_populations,
    )

    for kind in {case.scenario for case in cases}:
        models, history = models_and_history(root, kind)
        views, view = registered_history_views(root, kind, history)
        candidates = registered_populations(root)["target", kind]
        sources = build_source(
            models[1],
            history,
            candidates,
            cache_directory=Path(root) / "sources" / kind,
            view_record_ids=view["selected_record_ids"],
        )
        source = sources["specification"]
        context.setdefault("prediction_inputs", {})[kind] = {
            "models": models,
            "sources": sources,
            "pooled": pooled_history(views, view),
        }
        context["inputs"][kind] = {
            "full": {
                "model_versions": [model_version(m) for m in models],
                "history_view": view,
                "source_snapshot": source,
            },
            "pooled_history": {"model_versions": [model_version(models[0])], "history_view": view},
        }
    for kind in {c.scenario for c in cases if c.method == "scenario_fuzz"}:
        path = Path(root) / "models/baselines" / kind
        completion = read_json(path / "training_completion.json")
        weight = path / "scenario_fuzz/classifier.pt"
        if (
            completion["status"] != "complete"
            or completion["weights_sha256"] != hashlib.sha256(weight.read_bytes()).hexdigest()
            or completion["specification"]["history_fingerprints"]
            != context["inputs"][kind]["full"]["history_view"]["history_fingerprints"]
        ):
            raise ValueError("Historical baseline construction changed before report collection")
        context["classifiers"][kind] = completion
    return context


def _check_formal_specification(case, context):
    from .history_transfer_experiment import CHECKPOINTS, TARGET_COUNT
    from .parallel import prediction_device

    spec = case.specification
    if len(spec["candidates"]) != TARGET_COUNT:
        raise ValueError("Formal results require the registered complete candidate count")
    expected_kwargs = (
        context["settings"].prediction_kwargs(case.method)
        if case.experiment == "prediction"
        else (
            context["settings"].budget_kwargs(case.method)
            if case.method in ("full", "target_only")
            else {}
        )
    )
    for name, value in expected_kwargs.items():
        if spec.get(name) != value.to_dict():
            raise ValueError("Task did not use the registered fixed GP setting")
    contract = spec.get("method_contract", spec.get("task_snapshot", {}).get("method_contract"))
    if contract != context["method_contracts"][case.method]:
        raise ValueError("Formal result uses an unapproved method implementation")
    if case.experiment == "prediction" and spec["checkpoints"] != list(CHECKPOINTS):
        raise ValueError("Formal prediction checkpoints differ from the registered design")
    if case.experiment == "prediction" and spec.get("device") != prediction_device(case.method):
        raise ValueError("Formal prediction device differs from the registered design")
    if case.experiment == "budget" and (spec["budget"], spec["initial_count"]) != (200, 20):
        raise ValueError("Formal discovery budget or initialization changed")
    inputs = context["inputs"][case.scenario]
    if case.method in ("full", "target_only", "pooled_history"):
        if spec.get("update_source_weights") is not True:
            raise ValueError("Main GP results must use the registered mixture update setting")
        if case.experiment == "budget" and any(
            spec.get(k) != v
            for k, v in {
                "feedback_mode": "joint",
                "prior_mode": "sources",
                "historical_rss": True,
                "native_feedback": True,
            }.items()
        ):
            raise ValueError("Main GP query results changed their joint feedback or prior mode")
        if case.method == "target_only":
            fields = (
                ("model", "source_snapshot", "history_view", "model_construction")
                if case.experiment == "prediction"
                else ("target_model_version", "encoder_version", "source_snapshot")
            )
            if any(spec.get(k) is not None for k in fields):
                raise ValueError("Target-only GP accessed a historical model or source")
            if case.experiment == "budget" and spec.get("response_kernel") != "SharedPrivateKernel":
                raise ValueError("Target-only GP used a different response kernel")
        else:
            base_version = (
                spec["model"] if case.experiment == "prediction" else spec["target_model_version"]
            )
            if base_version != inputs[case.method]["model_versions"][0]:
                raise ValueError("Historical GP base differs from its registered historical model")
        if (
            case.method == "pooled_history"
            and spec["history_view"] != inputs["pooled_history"]["history_view"]
        ):
            raise ValueError("Pooled GP used different historical record IDs")
        if case.method == "full":
            expected_source = copy.deepcopy(inputs["full"]["source_snapshot"])
            expected_source.update(
                candidates=spec["candidates"], source_contract=context["source_contract"]
            )
            if spec["source_snapshot"] != expected_source:
                raise ValueError("Ours used different source records, model or public candidates")
            if (
                case.experiment == "budget"
                and spec["encoder_version"] != inputs["full"]["model_versions"][1]
            ):
                raise ValueError("Ours used a different historical encoder")
    elif case.experiment == "budget":
        configuration = (
            {} if case.method == "random" else context["baseline_configurations"][case.method]
        )
        if spec["configuration"] != configuration:
            raise ValueError("Baseline configuration differs from its registered default")
        expected = {}
        if case.method == "scenario_fuzz":
            expected = {
                "model_version": context["classifiers"][case.scenario]["model_version"],
                "history_fingerprints": inputs["full"]["history_view"]["history_fingerprints"],
            }
        if spec["construction"] != expected:
            raise ValueError(
                "Baseline history or trained classifier differs from its approved input"
            )


def _check_case_specification(case):
    spec = case.specification
    if case.experiment == "prediction":
        if any(
            spec.get(k) != v
            for k, v in {
                "sut_id": case.sut_id,
                "scenario": case.scenario,
                "feedback_seed": case.seed,
                "method": case.method,
            }.items()
        ):
            raise ValueError("Approved prediction specification belongs to another case")
    elif (
        spec["seed"] != case.seed
        or spec["task_snapshot"]["method_contract"]["method"] != case.method
    ):
        raise ValueError("Approved budget specification belongs to another case")


def _check_checkpoint(path, saved, expected, truth, *, complete):
    records = compact_feedback(saved["query_ledger"])
    if not path.exists():
        if complete or records:
            raise ValueError("Paid responses require their persisted recovery receipt")
        return
    checkpoint = read_json(path)
    declaration = checkpoint["specification"]
    budget = len(expected["split"]["query_order"]) if "split" in expected else expected["budget"]
    if declaration != {
        "task": expected,
        "budget": budget,
        "account": "target",
        "joint_feedback": True,
        "candidates": [asdict(s) for s in truth.candidates],
        "domain": None if truth.bank.domain is None else truth.bank.domain.to_dict(),
    }:
        raise ValueError("Recovery receipt uses another registered task or candidate library")
    if (
        checkpoint["query_ledger"] != records
        or checkpoint["paid_feedback_hash"] != paid_feedback_hash(records)
        or checkpoint["status"] not in ("running", "interrupted", "failed", "complete")
        or (complete and checkpoint["status"] != "complete")
    ):
        raise ValueError("Result/failure and persisted paid-feedback receipt disagree")
    successful = [r for r in records if r["status"] == "complete"]
    if (
        len(successful) != checkpoint["successful_paid_responses"]
        or len(successful) != len(checkpoint["trajectory_hashes"])
        or any(r["status"] not in ("complete", "failed") for r in records)
    ):
        raise ValueError("Incomplete or malformed paid recovery receipt")
    for record, fingerprint in zip(successful, checkpoint["trajectory_hashes"]):
        trace = truth.bank(truth.candidates[record["index"]]).trajectory
        if trajectory_hash(trace) != fingerprint:
            raise ValueError("Saved paid trace changed after execution")


def _failure_rows(case, saved, truth):
    from .initialization import initial_indices
    from .k_shot import METRICS

    spec = case.specification
    if saved["specification"] != spec or not saved.get("error"):
        raise ValueError("Failure record lacks its registered specification or reason")
    records = saved["query_ledger"]
    initial_count = 1 if case.experiment == "prediction" else spec["initial_count"]
    budget = (
        len(spec["split"]["query_order"]) if case.experiment == "prediction" else spec["budget"]
    )
    initial = (
        spec["split"]["query_order"]
        if case.experiment == "prediction"
        else initial_indices(len(truth.candidates), initial_count, case.seed)
    )
    if len(records) > budget or len({r["index"] for r in records}) != len(records):
        raise ValueError("Failed task exceeds its budget or repeats a candidate")
    for call, record in enumerate(records, 1):
        index = record["index"]
        if (
            type(index) is not int
            or not 0 <= index < len(truth.candidates)
            or record["call"] != call
            or record["account"] != "target"
            or record["purpose"] != ("initial" if call <= initial_count else "target")
            or record["scenario_id"] != truth.candidates[index].scenario_id
            or (call <= len(initial) and index != initial[call - 1])
        ):
            raise ValueError("Failed task has an invalid paid sequence or initialization")
        if record["status"] == "complete":
            if _canonical_feedback(record["feedback"]) != truth.paid_feedback(
                index, case.experiment == "budget"
            ):
                raise ValueError("Failed task paid feedback differs from its actual native trace")
        elif record["status"] != "failed" or call != len(records) or not record.get("error"):
            raise ValueError("Only the last execution in a failed task may lack feedback")
    row = {
        "sut_id": case.sut_id,
        "scenario": case.scenario,
        "seed": case.seed,
        "method": case.method,
        "status": "failed",
        "failure_reason": saved["error"],
        "candidate_count": len(truth.candidates),
        "pool_collisions": int(truth.labels.sum()),
    }
    if case.experiment == "budget":
        return [row]
    return [{**row, "K": k, **{m: None for m in METRICS}} for k in spec["checkpoints"]]


def _cost_row(case, saved):
    complete = [r for r in saved["query_ledger"] if r["status"] == "complete"]
    resources = saved.get("result", {}).get("resources", {})
    source = saved.get("source_construction_resources") or {}
    return {
        "experiment": case.experiment,
        "sut_id": case.sut_id,
        "scenario": case.scenario,
        "seed": case.seed,
        "method": case.method,
        "successful_logical_queries": len(complete),
        "failed_query_calls": len(saved["query_ledger"]) - len(complete),
        "equivalent_execution_s": sum(r["feedback"]["execution_s"] for r in complete),
        "paid_feedback_wall_s": sum(r["wall_s"] for r in saved["query_ledger"]),
        "selector_wall_s": resources.get("wall_s"),
        "selector_peak_cpu_bytes": resources.get("sampled_peak_cpu_bytes"),
        "selector_device": resources.get("device"),
        "selector_peak_cuda_allocated_bytes": resources.get("peak_cuda_allocated_bytes"),
        "source_construction_wall_s": source.get("wall_s"),
        "additional_physical_calls": 0,
        "cost_scope": "Stored-library replay; selector time covers the last completed attempt; unavailable earlier computation remains null",
    }


def collect_cases(cases, result_root, bank_root, *, scope="formal"):
    cases = list(cases)
    state = inventory(cases, result_root)
    context = None
    if scope == "formal":
        from .history_transfer_experiment import require_method_evaluation

        _formal_cases(cases, result_root)
        require_method_evaluation(result_root)
    elif scope != "development" or any(c.sut_id not in DEVELOPMENT_IDS for c in cases):
        raise ValueError("Development report readers accept only registered old ADS configurations")
    if any(r["status"] == "pending" for r in state):
        return {"scope": scope, "status": "incomplete", "inventory": state, "rows": [], "costs": []}
    if any(c.specification is None for c in cases):
        raise ValueError("Completed cases need independently approved specifications")
    if scope == "formal":
        context = _formal_context(result_root, cases)
    for case in cases:
        _check_case_specification(case)
        if context is not None:
            _check_formal_specification(case, context)
    return audit_records(cases, result_root, bank_root, state, context, scope)


def audit_records(cases, result_root, bank_root, state, context, scope):
    """Reconstruct forecasts and metrics after the study has verified the registered inputs."""
    rows, costs, banks, raw_snapshots = [], [], {}, {}
    forecast_checks, weight_checks = [], []
    for case, status in zip(cases, state):
        key = case.sut_id, case.scenario
        if key not in banks:
            bank = ExecutionBank(Path(bank_root) / case.scenario / case.sut_id)
            if bank.manifest["sut_id"] != case.sut_id:
                raise ValueError("Reader bank belongs to another ADS")
            banks[key] = BankTruth(bank)
        truth = banks[key]
        result_path, failure_paths, checkpoint = _files(case, result_root)
        saved = read_json(result_path if status["status"] == "complete" else failure_paths[-1])
        _check_specification(saved, case.specification, truth)
        _check_checkpoint(
            checkpoint, saved, case.specification, truth, complete=status["status"] == "complete"
        )
        if status["status"] == "failed":
            metrics = _failure_rows(case, saved, truth)
        elif case.experiment == "prediction":
            if context is not None:
                from .transfer_prediction_audit import verify_forecasts, verify_weight_updates

                inputs = context["prediction_inputs"][case.scenario]
                checked = verify_forecasts(
                    truth.candidates,
                    saved["result"]["prediction_snapshots"],
                    saved["query_ledger"],
                    [truth.margins[r["index"]] for r in saved["query_ledger"]],
                    method=case.method,
                    settings=context["settings"],
                    models=(None, None) if case.method == "target_only" else inputs["models"],
                    pooled=inputs["pooled"] if case.method == "pooled_history" else None,
                    sources=inputs["sources"] if case.method == "full" else None,
                )
                identity = {
                    "sut_id": case.sut_id, "scenario": case.scenario,
                    "seed": case.seed, "method": case.method,
                }
                forecast_checks.append({**identity, **checked})
                if case.method == "full":
                    checked = verify_weight_updates(
                        saved["result"], saved["query_ledger"], inputs["sources"]
                    )
                    weight_checks.append({**identity, **checked})
            metrics = prediction_record(
                saved, case.specification, truth, method=case.method, seed=case.seed
            )
        else:
            metrics = [
                budget_record(saved, case.specification, truth, method=case.method, seed=case.seed)
            ]
        rows.extend({"experiment": case.experiment, **r} for r in metrics)
        if (
            scope == "formal" and status["status"] == "complete"
            and case.experiment == "prediction" and case.method in ("target_only", "full")
        ):
            for snapshot in saved["result"]["prediction_snapshots"]:
                key = case.sut_id, case.scenario, case.seed, snapshot["paid_queries"]
                raw_snapshots.setdefault(key, {})[case.method] = snapshot[
                    "target_parameters" if case.method == "target_only" else "physical_parameters"
                ]
        costs.append(
            {
                "status": status["status"],
                "prior_failure_files": status["prior_failure_files"],
                **_cost_row(case, saved),
            }
        )
    raw_checks = []
    for (sut, scene, seed, count), parameters in raw_snapshots.items():
        if set(parameters) != {"target_only", "full"}:
            continue  # Failed pairs retain their explicit failure rows.
        own, reference = parameters["full"], parameters["target_only"]
        if own.keys() != reference.keys():
            raise ValueError("Raw target branches fit different parameter sets")
        error = max(abs(own[name] - reference[name]) for name in own)
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError("Ours raw target adaptation differs from the paired Target-only GP")
        raw_checks.append({
            "sut_id": sut, "scenario": scene, "seed": seed, "K": count,
            "maximum_parameter_error": error,
        })
    return {
        "scope": scope,
        "status": "complete",
        "inventory": state,
        "approved_cases": [
            {
                "key": list(c.key),
                "specification_sha256": hashlib.sha256(
                    json.dumps(c.specification, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
            }
            for c in cases
        ],
        "rows": rows,
        "prediction_forecast_checks": forecast_checks,
        "mixture_weight_checks": weight_checks,
        "raw_target_parameter_checks": raw_checks,
        "costs": costs,
        "native_bank_profiles": [
            {"sut_id": a, "scenario": s, **t.profile} for (a, s), t in banks.items()
        ],
        "physical_banks": [
            {
                "sut_id": a,
                "scenario": s,
                "library_candidates": len(t.candidates),
                **{
                    k: t.bank.manifest.get(k)
                    for k in (
                        "attempted_calls",
                        "successful_calls",
                        "failed_or_interrupted_calls",
                        "physical_execution_s",
                    )
                },
            }
            for (a, s), t in banks.items()
        ],
        "physical_bank_scope": "Each target/development library counted once; historical collection is separate",
        "new_physical_executions": 0,
        "reader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "record_auditor_sha256": hashlib.sha256(
            Path(__file__).with_name("record_audit.py").read_bytes()
        ).hexdigest(),
    }


def export_source_data(destination, report):
    if report["status"] != "complete":
        raise ValueError("Do not export final source data from an incomplete case inventory")
    destination = Path(destination)
    write_json(destination / "audited_records.json", report)
    scalar_rows, curves = [], []
    for row in report["rows"]:
        scalar = {k: v for k, v in row.items() if k not in ("calibration", "metrics")}
        metrics = row.get("metrics", {})
        if metrics is None:
            metrics = {}
        scalar.update(
            {
                f"budget_{k}": v
                for k, v in metrics.items()
                if v is None or isinstance(v, (bool, int, float, str))
            }
        )
        scalar_rows.append(scalar)
        for index, found in enumerate(metrics.get("discovery_curve", [])):
            curves.append(
                {
                    **{k: row[k] for k in ("sut_id", "scenario", "seed", "method")},
                    "query": index + 1,
                    "cumulative_collisions": found,
                    "collision_coverage": metrics["recall_curve"][index],
                    "post_initial_collisions": metrics["post_initial_curve"][index],
                    "equivalent_testing_s": metrics["equivalent_testing_curve_s"][index],
                }
            )
    for name, rows in (
        ("metrics", scalar_rows),
        ("costs", report["costs"]),
        ("discovery_curves", curves),
    ):
        fields = list(
            dict.fromkeys(k for row in rows for k in row if k not in ("calibration", "metrics"))
        )
        if not fields and name == "discovery_curves":
            fields = [
                "sut_id",
                "scenario",
                "seed",
                "method",
                "query",
                "cumulative_collisions",
                "collision_coverage",
                "post_initial_collisions",
                "equivalent_testing_s",
            ]
        with (destination / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
