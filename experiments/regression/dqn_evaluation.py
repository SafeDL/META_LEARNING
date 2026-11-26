"""Collect the revised target DQN's common paper suite after independent repair confirmation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from time import perf_counter

from controllers.highway.catalog import FORMAL_BY_ID
from controllers.highway.contracts import runtime_contract
from controllers.highway.learning import ACTION_INTERFACE, CHECKPOINT_ROOT

from experiments.regression.dqn_learning import ANCESTOR_HISTORY, PHYSICAL_ACTIONS, STUDY, retains_skills
from experiments.executions import ExecutionBank, execution_contract
from experiments.regression.fixed_inputs import SUITE_SIZE, fixed_bank
from experiments.formal_scenarios import FREE_BOUNDS
from experiments.history_transfer_experiment import CHECKPOINTS, PREDICTION_METHODS, freeze
from experiments.parallel import COLLECTION_WORKERS, initialize_worker, prepare_processes
from experiments.protocol import ROOT
from experiments.qualification import verify_course_evidence
from experiments.regression.execution import checkpoint_identity, execute_record
from experiments.regression.preflight import DQN_CHECKPOINTS
from experiments.storage import read_json
from experiments.transfer_settings import native_settings

PREDICTION = STUDY / "k_shot_adaptation"
TARGET = "target_dqn"
REVISED = replace(FORMAL_BY_ID[TARGET], training_run="target_dqn_continual")


def confirmed_training_root():
    base = STUDY / "base"
    checkpoint = base / "selected/policy.zip"
    identity = checkpoint_identity(checkpoint)
    report = read_json(base / "confirmation.json")
    if (not report["passed"] or report["selected_sha256"] != identity
            or report["capability"]["courses"] != verify_course_evidence(base / "qualification")["courses"]):
        raise ValueError("The new base requires independent driving and capability confirmation")
    directory = PHYSICAL_ACTIONS / "cut_in_onset/reference/R0"
    path = directory / "confirmation.json"
    if path.is_file() and (directory / "acceptance.json").exists():
        confirmed = read_json(path)
        selected = read_json(directory / "selection.json")
        skills = read_json(directory / "acceptance.json")
        if (skills["passed"] and skills["parent_sha256"] == identity
                and skills["selected_sha256"] == confirmed["selected_sha256"]
                and retains_skills(skills["before"], skills["after"])
                and confirmed["parent_sha256"] == identity and confirmed["passed"] and selected["step"]
                and selected["parent_sha256"] == identity and selected["published_sha256"] ==
                confirmed["selected_sha256"] == checkpoint_identity(directory / "selected/policy.zip")
                and confirmed["capability"]["courses"] == verify_course_evidence(
                    directory / "qualification")["courses"]):
            return PHYSICAL_ACTIONS
    raise ValueError("No paid repair from the new base has passed independent confirmation")


def admission():
    confirmed_training_root()
    return STUDY / "base/selected/policy.zip"


def register_candidate():
    training_root = confirmed_training_root()
    checkpoint = STUDY / "base/selected/policy.zip"
    identity = checkpoint_identity(checkpoint)
    candidate_dir = CHECKPOINT_ROOT / REVISED.training_run
    candidate_dir.mkdir(exist_ok=True)
    candidate = candidate_dir / "best.zip"
    if not candidate.exists():
        candidate.hardlink_to(checkpoint)
    if checkpoint_identity(candidate) != identity:
        raise ValueError("Revised registered controller and confirmed base differ")
    base = STUDY / "base"
    selection = read_json(base / "selection.json")
    parent = DQN_CHECKPOINTS["parent"]
    capability = read_json(base / "confirmation.json")["capability"]
    freeze(STUDY / "policy_regression/protocol.json", {
        "training_root": training_root.relative_to(ROOT).as_posix(),
        "parent_sha256": identity, "ancestor_sha256": checkpoint_identity(parent)})
    freeze(base / "qualification/qualification.json", {
        "sut_id": TARGET, "parameters": REVISED.parameters, "training_seed": REVISED.training_seed,
        "runtime_contract": runtime_contract(REVISED), **capability})
    return freeze(candidate_dir / "training.json", {
        "sut_id": TARGET, "training_seed": REVISED.training_seed, "object": asdict(REVISED),
        "status": "qualified", "configuration": {"action_interface": ACTION_INTERFACE,
        "algorithm": "DQN with successful parent demonstrations", "protocol": "experiments/regression/dqn_learning.py"},
        "initial_checkpoint": {"path": parent.relative_to(ROOT).as_posix(), "sha256": checkpoint_identity(parent)},
        "actual_timesteps": selection["actual_online_steps"], "selected_step": selection["step"],
        "checkpoint_sha256": identity, "selection": (base / "selection.json").relative_to(ROOT).as_posix(),
        "qualification": capability})


def execute_target(job):
    checkpoint, scene, file, ordinal = job
    row = execute_record(checkpoint, scene, file, purpose="target_bank", public_interface=True,
                         policy="dqn", bank_call=ordinal)
    return row["feedback"]["execution_s"], row["wall_s"]


def collect_target(kind, workers):
    checkpoint = admission()
    original = fixed_bank(kind, "dqn")
    identity = checkpoint_identity(checkpoint)
    candidate = CHECKPOINT_ROOT / REVISED.training_run / "best.zip"
    if checkpoint_identity(candidate) != identity:
        raise ValueError("Revised registered controller and confirmed base differ")
    root = PREDICTION / "physical/target" / kind / TARGET
    if len(original.candidates) != SUITE_SIZE:
        raise ValueError("Revised suite size differs from PPO")
    freeze(root / "protocol.json", {"checkpoint_sha256": identity,
        "coordinates": [asdict(s) for s in original.candidates], "gp_settings": native_settings().to_dict()})
    contract = execution_contract(REVISED, kind=kind)
    spec = {"format": "collision-execution-bank", "sut_id": TARGET, "contract": contract,
            "scenarios": [asdict(s) for s in original.candidates], "environment_seed": 81001,
            "qualification": (STUDY / "base/qualification/qualification.json").relative_to(ROOT).as_posix()}
    if (root / "manifest.json").exists():
        bank = ExecutionBank(root)
        if any(bank.manifest[key] != value for key, value in spec.items()):
            raise ValueError("Native DQN bank no longer matches its registration")
        return audit_target(bank)
    jobs = [(candidate, scene, root / "executions" / f"{i:06d}.json.gz", i + 1)
            for i, scene in enumerate(original.candidates)]
    prepare_processes()
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        costs = list(pool.map(execute_target, jobs, chunksize=8))
    if any((root / "executions").glob("*.attempt.json")):
        raise ValueError("The bank still contains unfinished native executions")
    freeze(root / "manifest.json", {**spec, "status": "complete", "candidate_count": len(costs),
           "successful_calls": len(costs), "attempted_calls": len(costs), "failed_or_interrupted_calls": 0,
           "last_collection_wall_s": perf_counter() - started, "physical_execution_s": sum(s for s, _ in costs),
           "total_successful_call_wall_s": sum(w for _, w in costs)})
    return audit_target(ExecutionBank(root))


def audit_target(bank):
    from experiments.audit_transfer_targets import audit_bank

    path = bank.root / "quality.json"
    digest = bank.snapshot_fingerprint
    if path.exists():
        if read_json(path)["bank_sha256"] != digest:
            raise ValueError("Target quality evidence belongs to a different physical bank")
    else:
        freeze(path, {"bank_sha256": digest, **audit_bank(bank)})
    return bank


def prepare_regression(kind, *, through=None):
    from experiments.regression.dqn_learning import certify_continuation
    from experiments.regression.fixed_inputs import AUC_WINDOW, METHODS, NEW_WINDOW, VERSIONS, initial_indices, paid_reference_records, reference_root, scene_root
    from experiments.protocol import episode_duration
    from experiments.regression.audit import verify_execution
    from experiments.regression.execution import response_archive
    from experiments.regression.preflight import DQN_CONTINUAL
    from experiments.regression.protocol import repair_subset
    from experiments.schema import Scenario

    parent = admission()
    root = reference_root(kind, DQN_CONTINUAL)
    protocol = read_json(root / "protocol.json")
    chain = None
    versions = VERSIONS[1:VERSIONS.index(through) + 1] if through else VERSIONS[1:]
    if through is None:
        chain = read_json(root / "chain.json")
        if chain["status"] != "complete" or [r["version"] for r in chain["rounds"]] != list(versions):
            raise ValueError("Four qualified DQN repair releases are required for this scene")
    bank = ExecutionBank(PREDICTION / "physical/target" / kind / TARGET)
    if (bank.manifest["contract"] != execution_contract(REVISED, kind=kind)
            or bank.candidates != fixed_bank(kind, "dqn").candidates):
        raise ValueError("Regression requires the same DQN bank and ordered coordinates as E01")
    archives = []
    for name, folder, digest in (
            ("ancestor", ANCESTOR_HISTORY / kind, checkpoint_identity(DQN_CHECKPOINTS["parent"])),
            ("parent", STUDY / kind / "reference", checkpoint_identity(parent))):
        records = []
        for i, scene in enumerate(protocol["history"]):
            row = read_json(folder / "history/parent" / f"{i:04d}.json.gz")
            verify_execution(row, scene, digest)
            if row["purpose"] != "history":
                raise ValueError("DQN sources require paid historical executions")
            records.append(row)
        archives.append(response_archive(name, records, "dqn"))
    failures = [s for s, hit in zip(archives[-1].scenarios, archives[-1].collision, strict=True) if hit]
    current, historical = repair_subset(failures), []
    models, paid, repairs = {"h0": parent}, [], []
    for i, version in enumerate(versions):
        folder = root / version
        selection = read_json(folder / "selection.json")
        spec = read_json(folder / "protocol.json")
        if (spec["current"] != [asdict(s) for s in current]
                or spec["historical"] != [asdict(s) for s in historical]
                or spec["parent_sha256"] != checkpoint_identity(parent)):
            raise ValueError("Repair did not use the preceding policy and its paid failure feedback")
        if selection["step"]:
            parent = folder / "selected/policy.zip"
            certify_continuation(parent.parent)
        identity = checkpoint_identity(parent)
        release = read_json(folder / ("confirmation.json" if i == 0 else "release.json"))
        if (identity != release["selected_sha256"] or identity != selection["published_sha256"]
                or (chain and identity != chain["rounds"][i]["sha256"])):
            raise ValueError("The released policy differs from the development selection")
        capability = verify_course_evidence(folder / "qualification")
        if capability["courses"] != release["capability"]["courses"]:
            raise ValueError("Released driving evidence differs from the official qualification")
        models[version] = parent
        repairs.append({"checkpoint_sha256": identity, "wall_s": selection["elapsed_s"],
                        "actual_environment_steps": selection["actual_online_steps"], "selected_step": selection["step"],
                        "qualification_executions": len(capability["courses"])})
        if i + 1 < len(versions):
            query_root = root
            if chain:
                query_root = ROOT / chain["rounds"][i]["query_root"]
            elif not (root / "random/queries" / f"result_{i + 1}.json").exists():
                query_root = STUDY / kind / "reference"
            records = paid_reference_records(query_root, protocol, i + 1, identity)
            paid.append(records)
            historical = repair_subset(failures)
            current = [Scenario(**r["scenario"]) for r in records if r["feedback"]["collision"]]
            failures.extend(current)
    registration = {
        "candidates": [asdict(s) for s in bank.candidates], "h0_bank": str(bank.root.relative_to(ROOT)),
        "bank_sha256": bank.snapshot_fingerprint, "environment_seed": bank.manifest["environment_seed"],
        "duration_s": episode_duration(kind), "initial_indices": initial_indices(SUITE_SIZE),
        "methods": list(METHODS), "auc_window": AUC_WINDOW, "new_window": NEW_WINDOW, "repairs": repairs,
        "checkpoints": {v: {"path": str(p.relative_to(ROOT)), "sha256": checkpoint_identity(p)}
                        for v, p in models.items()}}
    if through:
        return bank, archives, models, paid, registration
    freeze(scene_root(kind, DQN_CONTINUAL) / "registration.json", registration)
    return bank, archives, models, paid


def version_suite(kind, version, workers, *, methods=(), device="cpu", cpu_threads=1):
    from experiments.costs import PeakMemory
    from experiments.regression.fixed_inputs import scene_root, version_directory
    from experiments.regression.fixed_replay import model_paths, replay_job, source_snapshot, truth_rows
    from experiments.regression.execution import response_archive
    from experiments.regression.preflight import DQN_CONTINUAL
    from experiments.regression.run_fixed import collect_all

    bank, history, models, paid, registration = prepare_regression(kind, through=version)
    root = scene_root(kind, DQN_CONTINUAL)
    truth = version_directory(kind, "truth", version, DQN_CONTINUAL)
    if not methods:
        jobs = [(models[version], s, truth / f"{i:06d}.json.gz", "fixed_suite", DQN_CONTINUAL)
                for i, s in enumerate(bank.candidates)]
        collect_all(jobs, workers, f"{kind} {version}")
        return truth_rows(kind, version, bank, registration, DQN_CONTINUAL)
    sealed = read_json(truth / "sealed.json")
    if sealed["checkpoint_sha256"] != registration["checkpoints"][version]["sha256"]:
        raise ValueError("Paid replay requires the sealed responses of this exact qualified release")
    base = encoder = sources = None
    if "full" in methods:
        base, encoder = model_paths(kind, history, DQN_CONTINUAL)
    history.extend(response_archive(f"paid_policy_{i}", rows, "dqn") for i, rows in enumerate(paid, 1))
    if encoder is not None:
        with PeakMemory() as resources:
            sources = source_snapshot(encoder, history, bank.candidates, root / "sources" / f"{version}.npz")
        cost = root / "sources" / f"{version}.json"
        if not cost.exists():
            from experiments.storage import write_json
            write_json(cost, resources.result)
    jobs = [(kind, version, method, bank.candidates, registration["initial_indices"],
             [] if method == "birchler" else history, base if method == "full" else None,
             sources if method == "full" else None, device if method == "full" else "cpu", cpu_threads, DQN_CONTINUAL)
            for method in methods]
    prepare_processes()
    with ProcessPoolExecutor(max_workers=min(len(jobs), workers), initializer=initialize_worker) as pool:
        for result in pool.map(replay_job, jobs):
            print("replay complete", *result, flush=True)


def study_inputs(kind, mode):
    from experiments.history_transfer_experiment import OUTPUT, pooled_history, registered_history_views, scenario_history
    from experiments.weight_ablation.run import declaration, input_receipts, load_context, sealed_specification

    identity = checkpoint_identity(admission())
    bank = ExecutionBank(PREDICTION / "physical/target" / kind / TARGET)
    original = fixed_bank(kind, "dqn")
    if (bank.manifest["contract"] != execution_contract(REVISED, kind=kind)
            or bank.manifest["contract"]["runtime_contract"]["checkpoint_sha256"] != identity
            or bank.candidates != original.candidates):
        raise ValueError("Prediction requires the confirmed DQN on precisely the original coordinates")
    proof, approved = input_receipts()
    # E03's full-method compatibility checks replay the original sealed E01/E02 predictions.
    ablation = ROOT / "results/weight_adaptation"
    if read_json(ablation / "protocol.json") != declaration():
        raise ValueError("Prediction or historical source implementation changed since its compatibility replay")
    bridge = read_json(ablation / "compatibility" / kind / TARGET / f"{mode}.json")
    if (not bridge["selected_order_equal"] or bridge["new_paid_queries"] or bridge["paid_queries_replayed"] != 200
            or max(bridge["maximum_errors"].values()) > 1e-10):
        raise ValueError("The original paid-sequence compatibility check failed")
    old_case = sealed_specification(mode, original, approved)
    context = load_context(kind, proof, old_case)
    archives = scenario_history(OUTPUT, kind)
    context["pooled"] = pooled_history(*registered_history_views(OUTPUT, kind, archives))
    return bank, original, old_case, context


def revised_prediction_case(bank, old_case, context, method, frozen):
    from experiments import transfer_execution as execution
    from experiments.k_shot import split_candidates
    from experiments.weight_ablation.run import LIMITS, SEED
    settings = native_settings()
    models, sources, pooled = (None, None), None, None
    if method in ("full", "pooled_history"):
        models = context["models"]
    if method == "full":
        sources = context["sources"]
    if method == "pooled_history":
        pooled = context["pooled"]
    if frozen and method != "full":
        raise ValueError("Only the source-weight mechanism is ablated")
    case = execution.prediction_case(bank, split_candidates(bank.candidates, SEED), SEED, method,
            models, pooled, sources, settings=settings, checkpoints=CHECKPOINTS, limits=LIMITS,
            update_source_weights=not frozen)
    if case.specification["split"] != old_case.specification["split"]:
        raise ValueError("DQN revision changed the registered feedback/evaluation allocation")
    return case, models, pooled, sources


def prediction(kind, method, *, frozen=False):
    """Use the sealed original historical inputs and pay the revised target for its shared sequence."""
    from experiments import transfer_execution as execution
    from experiments.transfer_settings import register_settings

    bank, _, old_case, context = study_inputs(kind, "prediction")
    case, models, pooled, sources = revised_prediction_case(bank, old_case, context, method, frozen)
    directory = STUDY / "weight_adaptation" if frozen else PREDICTION
    register_settings(directory)
    execution.declare(directory, case, scope="formal")
    return execution.run_prediction(directory, case, bank, models, pooled, sources, settings=native_settings(),
                                   paid_execution=lambda index: bank(bank.candidates[index]))


def revised_budget_case(bank, original, context, method, frozen):
    import numpy as np
    from experiments.budget import protocol
    from experiments import transfer_execution as execution
    from experiments.budget.execution import make_case
    from experiments.collect_transfer_records import ApprovedCase
    control = ApprovedCase("prediction", TARGET, bank.candidates[0].kind, protocol.SEED, "target_only", None)
    source = execution._paths(PREDICTION, control)
    origin = read_json(source[0])["specification"]
    if origin["bank_snapshot"] != bank.snapshot_fingerprint:
        raise ValueError("E02 requires the matching revised E01 bank")
    if method == "static_prior":
        case, public = make_case(original, method, context)
        updated, _ = execution._paths(PREDICTION, replace(control, method="full"))
        saved = read_json(updated)
        snapshot, = [s for s in saved["result"]["prediction_snapshots"] if s["paid_queries"] == 0]
        if (snapshot["observed_indices"] or saved["specification"]["bank_snapshot"] != bank.snapshot_fingerprint
                or not np.array_equal(snapshot["probabilities"], public["probabilities"])):
            raise ValueError("Static ranking changed across identical feedback-free historical priors")
        spec = case.specification
        public.update(candidates=bank.candidates, probabilities=snapshot["probabilities"])
        case = replace(case, specification={**spec, "bank_contract": bank.manifest["contract"],
                       "task_snapshot": {**spec["task_snapshot"], "bank_snapshot": bank.snapshot_fingerprint},
                       "origin": {**spec["origin"], "result": updated.relative_to(ROOT).as_posix(),
                                  "sha256": protocol.sha(updated)}})
    else:
        case, public = make_case(bank, method, context)
    if frozen:
        if method != "full":
            raise ValueError("Only the source-weight mechanism is ablated")
        public["update_source_weights"] = False
        case = replace(case, specification={**case.specification, "update_source_weights": False})
    return case, public


def budget(kind, method, *, frozen=False):
    from experiments.budget import protocol
    from experiments import transfer_execution as execution
    from experiments.budget.audit import derive_random
    from experiments.budget.selectors import run
    from experiments.collect_transfer_records import ApprovedCase
    from experiments.record_audit import BankTruth

    bank, original, _, context = study_inputs(kind, "budget")
    directory = STUDY / ("weight_adaptation" if frozen else "budget_discovery")
    if method == "random":
        source = execution._paths(PREDICTION, ApprovedCase("prediction", TARGET, kind, protocol.SEED, "target_only", None))
        dest = directory / "budget" / TARGET / kind / str(protocol.SEED) / method / "result.json.gz"
        return derive_random(bank, BankTruth(bank), destination=dest, source=source)
    case, public = revised_budget_case(bank, original, context, method, frozen)
    execution.declare(directory, case, scope="formal")
    return execution._execute(directory, case, bank, run, public, scope="formal",
                              paid_execution=lambda i: bank(bank.candidates[i]))


if __name__ == "__main__":
    from experiments.budget.protocol import METHODS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "collect", "prediction", "frozen-prediction", "budget",
                                          "frozen-budget", "suite", "replay"))
    parser.add_argument("--version", choices=("R0", "R1", "R2", "R3"))
    parser.add_argument("--rankers", nargs="+", choices=("full", "detour", "birchler", "sdc_scissor"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--scene", choices=FREE_BOUNDS)
    parser.add_argument("--workers", type=int, default=COLLECTION_WORKERS)
    args = parser.parse_args()
    if args.stage == "register":
        register_candidate()
    else:
        initialize_worker()
        for kind in ([args.scene] if args.scene else FREE_BOUNDS):
            if args.stage in ("suite", "replay"):
                if not args.version or (args.stage == "replay" and (not args.rankers or args.version == "R3")):
                    parser.error("A suite needs a release; replay needs rankers and one of R0–R2")
                version_suite(kind, args.version, args.workers, methods=args.rankers if args.stage == "replay" else (),
                              device=args.device, cpu_threads=args.cpu_threads)
            elif args.stage == "collect":
                collect_target(kind, args.workers)
            else:
                fixed = args.stage.startswith("frozen-")
                is_budget = args.stage.endswith("budget")
                methods = ("full",) if fixed else METHODS if is_budget else PREDICTION_METHODS
                if args.method and args.method not in methods:
                    parser.error("Method is outside this registered experiment")
                for method in ([args.method] if args.method else methods):
                    (budget if is_budget else prediction)(kind, method, frozen=fixed)
