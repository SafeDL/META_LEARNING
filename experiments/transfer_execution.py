"""Execute approved tasks with explicit settings, limits and paid-feedback recovery."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from controllers.highway.catalog import FORMAL_TARGET_SUTS, INDEPENDENT_SUTS
from testing_methods.baselines.response_baselines.pooled_gp import select as pooled_select
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel

from .collect_transfer_records import ApprovedCase
from .domain_risk import disclose_driving_quality, disclose_impact
from .failure_metrics import METRIC_CONTRACT, summarize
from .history import model_version
from .history_transfer_experiment import freeze, prediction_specification
from .initialization import initial_indices
from .k_shot import prediction_metrics, validate_result
from .method_contracts import method_contract
from .baselines import BASELINES, baseline_selector
from .query import run_isolated
from .resume import ReplayInterrupted, ResumableLedger
from .storage import compact_feedback, compact_result, read_json, write_json

DEVELOPMENT_IDS = tuple(s.sut_id for s in INDEPENDENT_SUTS)


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def execution_contract(limits):
    if (
        set(limits) != {"wall_s", "child_memory_bytes"}
        or type(limits["wall_s"]) is not int
        or limits["wall_s"] <= 0
        or type(limits["child_memory_bytes"]) is not int
        or limits["child_memory_bytes"] <= 0
    ):
        raise ValueError("Use explicit positive selector wall and memory limits")
    return {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": limits,
        "resources_scope": "Selector wall includes query waits; per-query adaptation/selection costs are saved separately",
    }


def prediction_case(
    bank,
    split,
    seed,
    method,
    models,
    pooled,
    sources,
    *,
    settings,
    checkpoints,
    limits,
    evaluate=True,
    update_source_weights=True,
):
    if method not in ("target_only", "pooled_history", "full"):
        raise ValueError("Only the three registered GP prediction methods are supported")
    base, encoder = models
    if method == "target_only" and (
        base is not None or encoder is not None or sources is not None or pooled is not None
    ):
        raise ValueError("Target-only GP must not receive historical inputs")
    kwargs = settings.prediction_kwargs(method)
    specification = prediction_specification(
        bank,
        split,
        seed,
        method,
        base,
        pooled,
        sources,
        checkpoints,
        update_source_weights=update_source_weights,
        **kwargs,
    )
    specification.update(
        execution_contract=execution_contract(limits), evaluate_complete_bank=evaluate
    )
    return ApprovedCase(
        "prediction",
        bank.manifest["sut_id"],
        bank.candidates[0].kind,
        seed,
        method,
        canonical(specification),
    )


def budget_case(
    bank,
    seed,
    method,
    models,
    sources,
    *,
    settings,
    initial_count,
    budget,
    limits,
    classifier=None,
    evaluate=True,
    update_source_weights=True,
):
    if method not in ("target_only", "full", "random", "scenario_fuzz"):
        raise ValueError("Pooled GP is not a registered active-query method")
    if not 0 < initial_count <= budget <= len(bank.candidates):
        raise ValueError("Budget and initialization must fit the fixed candidate library")
    base, encoder = models
    common = {
        "task_snapshot": {
            "sut_id": bank.manifest["sut_id"],
            "scenario": bank.candidates[0].kind,
            "method": method,
            "method_contract": method_contract(method),
            "bank_snapshot": bank.snapshot_fingerprint,
        },
        "bank_contract": bank.manifest["contract"],
        "environment_seed": bank.manifest["environment_seed"],
        "candidates": [asdict(s) for s in bank.candidates],
        "budget": budget,
        "initial_count": initial_count,
        "initial_indices": initial_indices(len(bank.candidates), initial_count, seed),
        "seed": seed,
        "metric_contract": METRIC_CONTRACT,
        "execution_contract": execution_contract(limits),
        "evaluate_complete_bank": evaluate,
    }
    if method in ("target_only", "full"):
        if method == "target_only" and (
            base is not None or encoder is not None or sources is not None
        ):
            raise ValueError("Target-only GP must not receive historical models or priors")
        if method == "full" and (base is None or encoder is None or sources is None):
            raise ValueError("Ours requires the trained model pair and admitted source posteriors")
        common.update(
            target_model_version=None if base is None else model_version(base),
            encoder_version=None if encoder is None else model_version(encoder),
            source_snapshot=None if sources is None else sources["specification"],
            feedback_mode="joint",
            prior_mode="sources",
            historical_rss=True,
            update_source_weights=update_source_weights,
            feedback_contract="paid-collision-rss-actual-trajectory",
            batch_size=128,
            device="cpu",
            prediction_snapshot_counts=sorted(
                {0, budget, *[n for n in (20, 50, 100) if n < budget]}
            ),
            native_feedback=True,
            online_settings=settings.online.to_dict(),
        )
        if method == "target_only":
            common["response_kernel"] = "SharedPrivateKernel"
    else:
        if base is not None or encoder is not None or sources is not None:
            raise ValueError("Baseline selection must use only its declared inputs")
        construction = {}
        if method == "scenario_fuzz":
            if classifier is None:
                raise ValueError("ScenarioFuzz requires its completed historical classifier")
            construction = classifier["construction"]
            if construction["model_version"] != model_version(classifier["model"]):
                raise ValueError("Baseline classifier differs from its declared construction")
        elif classifier is not None:
            raise ValueError("Only ScenarioFuzz receives a historical classifier")
        common.update(
            method=method,
            configuration={} if method == "random" else asdict(BASELINES[method].DEFAULT),
            construction=construction,
            feedback_contract="paid-collision-rss-native-risk-actual-trajectory",
        )
    domain = getattr(bank, "domain", None)
    if domain is not None:
        common["parameter_definition"] = domain.to_dict()
    return ApprovedCase(
        "budget", bank.manifest["sut_id"], bank.candidates[0].kind, seed, method, canonical(common)
    )


def _paths(root, case):
    if case.experiment == "prediction":
        directory = Path(root) / "k_shot" / case.sut_id / case.scenario / str(case.seed)
        return (
            directory / f"{case.method}.json.gz",
            directory / f"{case.method}_paid_feedback.json.gz",
        )
    directory = Path(root) / "budget" / case.sut_id / case.scenario / str(case.seed) / case.method
    return directory / "result.json.gz", directory / "paid_feedback.json.gz"


def approval_path(root, case):
    return (
        Path(root)
        / "approvals"
        / case.experiment
        / case.sut_id
        / case.scenario
        / str(case.seed)
        / f"{case.method}.json"
    )


def declare(root, case, *, scope):
    """Study-level admission writes this specification before execution."""
    return freeze(
        approval_path(root, case),
        {"key": list(case.key), "specification": case.specification, "scope": scope},
    )


def _approve(root, case, bank, *, scope):
    from .collect_transfer_records import _check_case_specification

    if scope == "formal":
        if case.sut_id not in {s.sut_id for s in FORMAL_TARGET_SUTS}:
            from .public_ads.accumulation import admit_case

            admit_case(root, case, bank)
        path = approval_path(root, case)
        if not path.exists() or read_json(path) != {
            "key": list(case.key),
            "specification": case.specification,
            "scope": scope,
        }:
            raise ValueError("Study-level admission must approve the exact task before querying")
    elif scope != "development" or case.sut_id not in DEVELOPMENT_IDS:
        raise ValueError("Development execution accepts only registered old ADSs")
    _check_case_specification(case)
    if (
        bank.manifest["sut_id"] != case.sut_id
        or bank.candidates[0].kind != case.scenario
        or case.specification["candidates"] != canonical([asdict(s) for s in bank.candidates])
        or case.specification["bank_contract"] != canonical(bank.manifest["contract"])
    ):
        raise ValueError("Approved case differs from its execution bank")
    expected = execution_contract(case.specification["execution_contract"]["limits"])
    if case.specification["execution_contract"] != expected:
        raise ValueError("Approved executor implementation or resource limit changed")
    if scope == "development":
        declare(root, case, scope=scope)


def _native(execution):
    return disclose_driving_quality(disclose_impact(execution))


def _validate_output(case, result, records):
    indices = [r["index"] for r in records]
    queries, paid = compact_feedback(result["queries"]), compact_feedback(records)
    if (
        result["selected_indices"] != indices
        or len(set(indices)) != len(indices)
        or len(result["queries"]) != len(records)
        or any(canonical(q["feedback"]) != canonical(r["feedback"]) for q, r in zip(queries, paid))
    ):
        raise ValueError("Completed algorithm output differs from its paid queries")
    spec = case.specification
    if case.experiment == "prediction":
        if result["resources"]["device"] != spec["device"]:
            raise ValueError("Prediction device differs from the registered task")
        validate_result(result, spec["split"], checkpoints=spec["checkpoints"])
    elif (
        len(records) != spec["budget"]
        or indices[: spec["initial_count"]] != spec["initial_indices"]
        or result["actual_queries"] != spec["budget"]
        or result["maximum_queries"] != spec["budget"]
        or result["termination"] != "budget_exhausted"
    ):
        raise ValueError("Budget method changed its common initialization or paid budget")


def _finalize(case, bank, result, ledger, *, scope, construction_resources):
    spec = case.specification
    labels = (
        [bank(s).feedback.collision for s in bank.candidates]
        if spec["evaluate_complete_bank"]
        else None
    )
    saved = {
        "specification": spec,
        "result": compact_result(result),
        "query_ledger": compact_feedback(ledger.records),
    }
    if case.experiment == "prediction":
        metrics = []
        for snapshot in result["prediction_snapshots"] if labels is not None else ():
            values, calibration = prediction_metrics(
                snapshot["probabilities"],
                labels,
                spec["split"]["candidate_ids"],
                spec["split"]["evaluation_indices"],
            )
            metrics.append({"K": snapshot["paid_queries"], **values, "calibration": calibration})
        saved.update(
            metrics=metrics,
            new_physical_calls=0 if scope == "formal" else None,
            evaluation_scope=(
                "complete_fixed_evaluation"
                if labels is not None
                else "paid_only_development_no_full_truth"
            ),
        )
    else:
        saved.update(
            metrics=(
                None
                if labels is None
                else summarize(
                    bank.candidates,
                    labels,
                    result["queries"],
                    ledger.trajectories,
                    initial_count=spec["initial_count"],
                    query_ledger=ledger.records,
                    domain=getattr(bank, "domain", None),
                )
            ),
            evaluation=(
                {"scope": "paid_only_development_no_full_truth"}
                if labels is None
                else {"candidate_collisions": labels}
            ),
            source_construction_resources=construction_resources,
        )
    return saved


def _execute(
    root,
    case,
    bank,
    selector,
    public,
    *,
    paid_execution,
    scope,
    stop_after=None,
    construction_resources=None,
):
    _approve(root, case, bank, scope=scope)
    path, checkpoint = _paths(root, case)
    native = case.experiment == "budget"
    ledger = ResumableLedger(
        bank.candidates,
        public["budget"],
        (lambda scene: _native(bank(scene))) if native else bank,
        checkpoint_path=checkpoint,
        specification=case.specification,
        paid_execution=(lambda index: _native(paid_execution(index))) if native else paid_execution,
        joint_feedback=True,
        domain=getattr(bank, "domain", None),
        allow_registration=False,
        stop_after=stop_after,
    )
    if path.exists():
        saved = read_json(path)
        if (
            saved["specification"] != case.specification
            or saved["query_ledger"] != canonical(compact_feedback(ledger.records))
            or read_json(checkpoint)["status"] != "complete"
            or saved["result"]["selected_indices"] != [r["index"] for r in ledger.records]
        ):
            raise ValueError("Completed task differs from its approval or paid receipts")
        _validate_output(case, saved["result"], ledger.records)
        return saved
    started = time.perf_counter()
    attempt = {
        "specification": case.specification,
        "saved_responses_before_run": ledger.saved_count,
    }
    attempt_path = path.parent / "attempts" / f"{case.method}_{time.time_ns()}.json"
    limits = case.specification["execution_contract"]["limits"]
    stage = "selector"
    try:
        result = run_isolated(
            selector,
            public,
            ledger,
            wall_limit_s=limits["wall_s"],
            memory_limit_bytes=limits["child_memory_bytes"],
        )
        attempt["selector_resources"] = result["resources"]
        stage = "output_validation"
        _validate_output(case, result, ledger.records)
        ledger.finish()
        stage = "evaluation"
        saved = _finalize(
            case, bank, result, ledger, scope=scope, construction_resources=construction_resources
        )
        saved["wall_s"] = time.perf_counter() - started
        write_json(path, saved)
        attempt.update(status="complete", selector_resources=result["resources"])
    except Exception as error:
        attempt.update(
            status="interrupted" if isinstance(error, ReplayInterrupted) else "failed",
            error=repr(error),
            failure_stage=stage,
            query_ledger=compact_feedback(ledger.records),
        )
        failure = {
            "specification": case.specification,
            "error": repr(error),
            "failure_stage": stage,
            "query_ledger": compact_feedback(ledger.records),
        }
        failure_path = (
            path.with_suffix(".failure.json")
            if case.experiment == "prediction"
            else path.parent / f"failed_attempts_{time.time_ns()}.json"
        )
        write_json(failure_path, failure)
        raise
    finally:
        attempt.update(
            wall_s=time.perf_counter() - started, saved_responses_after_run=len(ledger.records)
        )
        write_json(attempt_path, attempt)
    return read_json(path)


def run_prediction(
    root,
    case,
    bank,
    models,
    pooled,
    sources,
    *,
    settings,
    paid_execution,
    scope="formal",
    stop_after=None,
):
    spec = case.specification
    actual = prediction_case(
        bank,
        spec["split"],
        case.seed,
        case.method,
        models,
        pooled,
        sources,
        settings=settings,
        checkpoints=spec["checkpoints"],
        limits=spec["execution_contract"]["limits"],
        evaluate=spec["evaluate_complete_bank"],
        update_source_weights=spec["update_source_weights"],
    )
    if actual != case:
        raise ValueError("Actual prediction inputs differ from the pre-query task declaration")
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
        public.update(
            pretrained=models[0],
            history=pooled,
            inference_backend="grouped",
            fit_settings=settings.pooled,
        )
    else:
        selector = select
        public.update(
            initial_indices=spec["split"]["query_order"][:1],
            feedback_mode="joint",
            pretrained=models[0],
            sources=sources,
            online_settings=settings.online,
            update_source_weights=spec["update_source_weights"],
        )
        if case.method == "target_only":
            public["raw_kernel_type"] = SharedPrivateKernel
    return _execute(
        root,
        case,
        bank,
        selector,
        public,
        paid_execution=paid_execution,
        scope=scope,
        stop_after=stop_after,
    )


def run_budget(
    root,
    case,
    bank,
    models,
    sources,
    *,
    settings,
    paid_execution,
    classifier=None,
    construction_resources=None,
    scope="formal",
    stop_after=None,
):
    spec = case.specification
    actual = budget_case(
        bank,
        case.seed,
        case.method,
        models,
        sources,
        settings=settings,
        initial_count=spec["initial_count"],
        budget=spec["budget"],
        limits=spec["execution_contract"]["limits"],
        classifier=classifier,
        evaluate=spec["evaluate_complete_bank"],
        update_source_weights=spec.get("update_source_weights", True),
    )
    if actual != case:
        raise ValueError("Actual query inputs differ from the pre-query task declaration")
    public = {
        "candidates": bank.candidates,
        "initial_indices": spec["initial_indices"],
        "budget": spec["budget"],
    }
    if case.method in ("full", "target_only"):
        selector = select
        public.update(
            pretrained=models[0],
            sources=sources,
            batch_size=128,
            feedback_mode="joint",
            mode="H1F1",
            update_source_weights=spec["update_source_weights"],
            device="cpu",
            online_settings=settings.online,
            snapshot_counts=spec["prediction_snapshot_counts"],
        )
        if case.method == "target_only":
            public["raw_kernel_type"] = SharedPrivateKernel
        if getattr(bank, "domain", None) is not None:
            public["feature_fn"] = bank.domain.features
    else:
        selector = baseline_selector
        public.update(method=case.method, seed=case.seed)
        if case.method != "random":
            public["config"] = BASELINES[case.method].DEFAULT
        if classifier is not None:
            public["classifier"] = classifier["model"]
        if getattr(bank, "domain", None) is not None:
            public["domain"] = bank.domain
    return _execute(
        root,
        case,
        bank,
        selector,
        public,
        paid_execution=paid_execution,
        scope=scope,
        stop_after=stop_after,
        construction_resources=construction_resources,
    )
