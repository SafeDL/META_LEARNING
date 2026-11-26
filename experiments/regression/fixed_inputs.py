"""Freeze the E01/E02 suite and verify each independent policy reference chain."""

from dataclasses import asdict, replace

import numpy as np

from controllers.highway.catalog import BY_ID
from simulator.highway.repair_env import physical_coordinate

from experiments.budget.protocol import OUTPUT as E02
from experiments.executions import ExecutionBank, execution_contract
from experiments.formal_scenarios import BRAKING_KIND, coordinate_stream, development_coordinates, verify_candidates
from experiments.history_transfer_experiment import OUTPUT as E01, freeze
from experiments.protocol import ROOT, episode_duration
from experiments.regression.audit import verify_capability_evidence, verify_execution, verify_optimizer_continuation, verify_repair
from experiments.regression.execution import ATTENTION, checkpoint_identity, response_archive
from experiments.regression.preflight import CHECKPOINTS, DQN_CHECKPOINTS, DQN_CONTINUAL, RUN_ROOT
from experiments.regression.protocol import INITIAL_COUNT, numeric_coordinate, repair_subset, sample_block, stop_reason
from experiments.resume import paid_feedback_hash, trajectory_hash
from experiments.schema import Scenario
from experiments.storage import read_json

OUTPUT = RUN_ROOT / "fixed_suite"
H0_COUNTS = {BRAKING_KIND: 173, "cut_in_onset": 484, "cut_out_rear_gap": 1836,
             "side_encroachment": 1281}
DQN_H0_COUNTS = {BRAKING_KIND: 173, "cut_in_onset": 1939, "cut_out_rear_gap": 1742,
                "side_encroachment": 1225}
VERSIONS = ("h0", "R0", "R1", "R2", "R3")
VERSION_DIRECTORIES = dict(zip(VERSIONS, ("original_policy", "prepared_policy", "first_repair_policy",
                                         "second_repair_policy", "final_repair_policy")))
ORIGINAL_METHODS = ("full", "detour", "birchler")
METHODS = (*ORIGINAL_METHODS, "sdc_scissor")
PPO_METHODS = (*METHODS, "spectre")
SUITE_SIZE = 2000
AUC_WINDOW = 500
NEW_WINDOW = 100
SEED = 81001


def methods_for(policy):
    return PPO_METHODS if policy in ("ppo", DQN_CONTINUAL, ATTENTION) else METHODS


def initial_indices(count):
    return np.random.default_rng(11).permutation(count)[:INITIAL_COUNT].tolist()


def output_root(policy):
    if policy == ATTENTION:
        return ROOT / "results/public_ads/policy_regression"
    if policy == DQN_CONTINUAL:
        from experiments.regression.dqn_learning import STUDY

        return STUDY / "policy_regression"
    return OUTPUT


def scene_root(kind, policy="ppo"):
    return output_root(policy) / kind


def version_directory(kind, purpose, version, policy="ppo"):
    return scene_root(kind, policy) / purpose / VERSION_DIRECTORIES[version]


def query_directory(kind, version, method, policy="ppo"):
    folder = "spectre_traffic_attributes" if method == "spectre" else method
    return version_directory(kind, "queries", version, policy) / folder


def fixed_bank(kind, policy="ppo"):
    target = "target_ppo" if policy == "ppo" else "target_dqn"
    sut = BY_ID[target]
    if policy != "ppo":
        # This bank supplies the original ordered suite and the pre-repair DQN identity.
        sut = replace(sut, training_run=DQN_CHECKPOINTS["parent"].parent.name)
    bank = ExecutionBank(E01 / "physical/target" / kind / target)
    expected = verify_candidates(E01 / "candidates/target" / kind, kind, SUITE_SIZE, "target")
    e02 = read_json(E02 / "budget" / target / kind / "11/full/result.json.gz")
    e02_random = read_json(E02 / "budget" / target / kind / "11/random/result.json.gz")
    if (bank.candidates != expected or len({physical_coordinate(s) for s in expected}) != SUITE_SIZE
            or bank.manifest["environment_seed"] != SEED
            or bank.manifest["contract"] != execution_contract(sut, kind=kind)
            or e02["specification"]["candidates"] != [asdict(s) for s in expected]
            or e02_random["specification"]["bank_snapshot"] != bank.snapshot_fingerprint):
        raise ValueError(f"{kind} does not match both frozen E01 and E02 inputs")
    if policy != "ppo" and read_json(scene_root(kind) / "registration.json")["candidates"] != [
            asdict(s) for s in expected]:
        raise ValueError("DQN and PPO must test identical ordered coordinates")
    return bank


def reference_protocol(kind, suite, policy="ppo"):
    if policy != "ppo":
        source = reference_root(kind) / "protocol.json"
        return freeze(reference_root(kind, policy) / "protocol.json", read_json(source))
    if kind != BRAKING_KIND:
        return read_json(RUN_ROOT / kind / "protocol.json")
    used = set(development_coordinates(kind))
    for split in ("history", "target"):
        used.update(tuple(r["independent"].values()) for r in coordinate_stream(kind, split))
    used.update(numeric_coordinate(s) for s in suite)
    history = sample_block(kind, 500, 72013, used, "history")
    candidates = sample_block(kind, 6000, 72017, used, "candidate")
    return freeze(scene_root(kind) / "reference/protocol.json", {
        "history": [asdict(s) for s in history],
        "candidates": [[asdict(s) for s in candidates[i:i + SUITE_SIZE]]
                       for i in range(0, len(candidates), SUITE_SIZE)],
        "initial_indices": initial_indices(SUITE_SIZE),
        "environment_seed": SEED,
        "duration_s": episode_duration(kind),
    })


def reference_root(kind, policy="ppo"):
    if policy == DQN_CONTINUAL:
        registration = read_json(output_root(policy) / "protocol.json")
        return ROOT / registration["training_root"] / kind / "reference"
    if policy != "ppo":
        return scene_root(kind, policy) / "reference"
    return scene_root(kind) / "reference" if kind == BRAKING_KIND else RUN_ROOT / kind


def model_root(kind, policy="ppo"):
    root = scene_root(kind, policy) if policy != "ppo" or kind == BRAKING_KIND else reference_root(kind)
    return root / "response_model"


def paths_for_version(root, version):
    if version == "R0":
        repair_root, name, checkpoint = root / "initialization", "repair", "policy.zip"
    else:
        repair_root, name = root / "random/repairs", f"repair_{version[1:]}"
        checkpoint = f"{name}.zip"
    return repair_root / checkpoint, repair_root / f"{name}.json", repair_root / f"{name}_qualification.json.gz"


def reference_history(kind, protocol, suite, policy="ppo"):
    root = reference_root(kind, policy)
    checkpoints = CHECKPOINTS if policy == "ppo" else DQN_CHECKPOINTS
    identities = {name: checkpoint_identity(path) for name, path in checkpoints.items()}
    excluded = {physical_coordinate(s) for s in suite}
    outside = [Scenario(**s) for s in (*protocol["history"],
                                      *(s for block in protocol["candidates"] for s in block))]
    if len({physical_coordinate(s) for s in outside}) != len(outside) or any(
            physical_coordinate(s) in excluded for s in outside):
        raise ValueError("Reference history or repair coordinates overlap the fixed suite")
    archives = []
    for name, digest in identities.items():
        rows = []
        for i, scene in enumerate(protocol["history"]):
            row = read_json(root / "history" / name / f"{i:04d}.json.gz")
            verify_execution(row, scene, digest)
            if row["purpose"] != "history":
                raise ValueError("Initial sources require actual historical executions")
            rows.append(row)
        archives.append(response_archive(name, rows, policy))
    return archives


def paid_reference_records(root, protocol, iteration, identity):
    path = root / "random/queries"
    saved = read_json(path / f"ledger_{iteration}.json")
    result = read_json(path / f"result_{iteration}.json")
    rows = saved["query_ledger"]
    pool = protocol["candidates"][iteration - 1]
    indices = [r["index"] for r in rows]
    order = np.random.default_rng(11).permutation(len(pool))[:len(rows)].tolist()
    if (saved["status"] != "complete" or saved["specification"]["task"]["checkpoint_sha256"] != identity
            or saved["specification"]["candidates"] != pool
            or saved["paid_feedback_hash"] != paid_feedback_hash(rows)
            or indices != result["result"]["selected_indices"] or indices != order
            or len(saved["trajectory_hashes"]) != len(rows)
            or indices[:INITIAL_COUNT] != protocol["initial_indices"]
            or result["actual_queries"] != len(rows)
            or result["collisions"] != sum(r["feedback"]["collision"] for r in rows)
            or result["stop_reason"] != stop_reason(result["collisions"], len(rows))
            or stop_reason(sum(r["feedback"]["collision"] for r in rows[:-1]), len(rows) - 1)
            is not None):
        raise ValueError("Reference ledger differs from its paid random sequence")
    records = []
    for call, (row, digest) in enumerate(zip(rows, saved["trajectory_hashes"]), 1):
        scene = pool[row["index"]]
        ordinal = scene["scenario_id"].rsplit(":", 1)[-1]
        record = read_json(root / "query_executions" / f"{identity}_{ordinal}.json.gz")
        verify_execution(record, scene, identity)
        if (row["call"] != call or row["status"] != "complete" or record["purpose"] != "query"
                or row["purpose"] != ("initial" if call <= INITIAL_COUNT else "target")
                or row["scenario_id"] != scene["scenario_id"] or trajectory_hash(record["trajectory"]) != digest
                or row["feedback"]["collision"] != record["feedback"]["collision"]
                or row["feedback"]["safety"]["rss_margin"] != record["feedback"]["safety"]["rss_margin"]):
            raise ValueError("Reference feedback differs from the paid native execution")
        records.append(record)
    return records


def verify_chain(kind, bank, protocol, archives, policy="ppo"):
    root = reference_root(kind, policy)
    chain = read_json(root / "random/chain.json")
    rejected = None
    if chain["status"] == "complete":
        accepted = VERSIONS
    elif policy != "ppo" and chain["status"] in ("terminated_unqualified", "terminated_without_new_failure",
                                                "terminated_development"):
        end = VERSIONS.index(chain["last_version"])
        if chain["status"] == "terminated_unqualified":
            rejected = VERSIONS[end]
            accepted = VERSIONS[:end]
            rejected_model, receipt, evidence = paths_for_version(root, rejected)
            failed = read_json(receipt)
            verify_capability_evidence(failed, evidence)
            if (failed["capability"]["qualified"]
                    or checkpoint_identity(rejected_model) != failed["checkpoint_sha256"]):
                raise ValueError("Rejected DQN checkpoint disagrees with its capability evidence")
        elif chain["status"] == "terminated_development":
            accepted = VERSIONS[:end]
        else:
            accepted = VERSIONS[:end + 1]
    else:
        raise RuntimeError(f"{kind} reference chain ended with {chain['status']}")
    checkpoints = CHECKPOINTS if policy == "ppo" else DQN_CHECKPOINTS
    models = {"h0": checkpoints["parent"]}
    failures = [s for s, hit in zip(archives[-1].scenarios, archives[-1].collision) if hit]
    excluded = {physical_coordinate(s) for s in bank.candidates}
    initial = repair_subset(failures)
    round_records, repairs = [], []
    parent_report = None
    for i, version in enumerate(accepted[1:]):
        checkpoint, receipt, evidence = paths_for_version(root, version)
        parent = checkpoint_identity(models[VERSIONS[i]])
        if i == 0:
            current, historical = initial, []
            registered = chain["common_checkpoint_sha256"]
        else:
            lineage = chain["rounds"][i - 1]
            if lineage["parent_sha256"] != parent or lineage["iteration"] != i:
                raise ValueError("Reference checkpoint ancestry changed")
            registered = lineage["checkpoint_sha256"]
            records = paid_reference_records(root, protocol, i, parent)
            current = [Scenario(**r["scenario"]) for r in records if r["feedback"]["collision"]]
            historical = repair_subset(failures)
            failures.extend(current)
            round_records.append(records)
        report = read_json(receipt)
        verify_repair(report, current, historical, parent, policy=policy)
        verify_capability_evidence(report, evidence)
        if (not report["capability"]["qualified"]
                or any(physical_coordinate(s) in excluded for s in current + historical)
                or checkpoint_identity(checkpoint) != registered or report["checkpoint_sha256"] != registered):
            raise ValueError(f"Reference checkpoint or repair inputs failed verification for {kind} {version}")
        verify_optimizer_continuation(models[VERSIONS[i]], checkpoint, policy=policy,
                                      state_path=checkpoint.parent / report["random_state_name"],
                                      report=report, parent_report=parent_report)
        parent_report = report
        repairs.append({key: report[key] for key in (
            "checkpoint_sha256", "wall_s", "actual_environment_steps", "qualification_executions")})
        models[version] = checkpoint
    if len(accepted) < len(VERSIONS) and len(accepted) > 1:
        iteration = len(accepted) - 1
        parent = checkpoint_identity(models[accepted[-1]])
        records = paid_reference_records(root, protocol, iteration, parent)
        round_records.append(records)
        current = [Scenario(**row["scenario"]) for row in records if row["feedback"]["collision"]]
        if rejected:
            verify_repair(failed, current, repair_subset(failures), parent, policy=policy)
        elif chain["status"] == "terminated_without_new_failure" and current:
            raise ValueError("Reference chain stopped despite acquiring paid collision reset points")
    elif rejected:
        verify_repair(failed, initial, [], checkpoint_identity(models["h0"]), policy=policy)
    if rejected:
        verify_optimizer_continuation(models[accepted[-1]], rejected_model, policy=policy,
                                      state_path=rejected_model.parent / failed["random_state_name"],
                                      report=failed, parent_report=parent_report)
    freeze(scene_root(kind, policy) / "registration.json", {
        "candidates": [asdict(s) for s in bank.candidates], "h0_bank": str(bank.root.relative_to(ROOT)),
        "bank_sha256": bank.snapshot_fingerprint, "environment_seed": SEED,
        "duration_s": episode_duration(kind), "initial_indices": initial_indices(SUITE_SIZE),
        "methods": list(ORIGINAL_METHODS if policy == "ppo" else METHODS),
        "auc_window": AUC_WINDOW, "new_window": NEW_WINDOW,
        "checkpoints": {v: {"path": str(p.relative_to(ROOT)), "sha256": checkpoint_identity(p)}
                        for v, p in models.items()}, "repairs": repairs,
    })
    (scene_root(kind, policy) / "status.json").unlink(missing_ok=True)
    return models, round_records
