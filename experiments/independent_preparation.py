"""Freeze and construct the new task and eight independent driving systems."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

from controllers.highway.catalog import BY_ID, INDEPENDENT_SUTS, SUTS
from controllers.highway.factory import create
from controllers.highway.learning import CHECKPOINT_ROOT
from controllers.highway.train import TRAINING, train, training_contract
from simulator.highway.interaction_env import GUARD_FRONT_GAP_M, GUARD_REAR_GAP_M
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer.training import DEFAULT as MODEL_TRAINING

from .discovery_config import read_json, write_json
from .executions import collect_bank
from .protocol import ROOT, TEST_RECORDS_ROOT
from .qualification import COURSES, QUALIFICATION_ROOT, qualify
from .scenarios import DOMAINS, sample
from .source_registration import specification_matches
from .storage import load_archive, save_archive

OUTPUT = ROOT / "experiment_outputs/benchmark/collision/discovery/independent_validation"


def preparation_design():
    paths = (
        "experiments/independent_preparation.py",
        "experiments/scenarios.py",
        "experiments/executions.py",
        "experiments/execution_events.py",
        "experiments/construction.py",
        "controllers/highway/catalog.py",
        "simulator/highway/interaction_env.py",
    )
    return json.loads(
        json.dumps(
            {
                "purpose": "Stage D construction before independent discovery outcomes",
                "objects": [asdict(s) for s in INDEPENDENT_SUTS],
                "scenarios": {k: asdict(v) for k, v in DOMAINS.items()},
                "interaction": {
                    "kind": "cut_in_interaction",
                    "paired_parameters": True,
                    "open": 256,
                    "restricted": 256,
                    "joint_budget": 128,
                    "guard_front_gap_m": GUARD_FRONT_GAP_M,
                    "guard_rear_gap_m": GUARD_REAR_GAP_M,
                    "guard_speed": "initial ego speed; constant and independent of ego control",
                    "guard_lane": "opposite the cut-in origin",
                    "ego_lane_centers_m": [0, 4, 8],
                },
                "qualification_courses": [asdict(c) for c in COURSES],
                "policy_training": TRAINING,
                "policy_training_source": training_contract(),
                "historical_model_training": asdict(MODEL_TRAINING),
                "data_roles": {
                    "history": [s.sut_id for s in SUTS if s.role == "history"],
                    "development": [s.sut_id for s in SUTS if s.role == "development"],
                },
                "source_conditions": ["all_sources", "strict_leave_family"],
                "baseline_coverage": "all six references, including Indago, in both source conditions",
                "main_runs": 2800,
                "mechanism_controls": ["H0F0", "H1F0", "H0F1"],
                "mechanism_scope": "all eight independent systems, five scenes, five seeds; all_sources",
                "additional_mechanism_runs": 600,
                "equivalent_queries": 435200,
                "new_interaction_history_development_calls": 6144,
                "new_target_pool_calls": 20480,
                "qualification_and_policy_training_calls": "metered separately; no discovery-based replacement",
                "source_files": {
                    p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
                },
            }
        )
    )


def freeze():
    decision = read_json(
        ROOT
        / "experiment_outputs/benchmark/collision/discovery/domain_comparison/diagonal_controls/finalization.json"
    )
    if (
        decision["status"] != "complete_finalized_development_method"
        or decision["gate"]["status"] != "passed"
    ):
        raise ValueError("Finish the admitted development method before independent construction")
    spec = preparation_design()
    path = OUTPUT / "preparation_protocol.json"
    if path.exists():
        recorded = read_json(path)
        if not specification_matches(recorded, spec):
            raise ValueError("Independent preparation was already frozen differently")
        return recorded
    write_json(path, spec)
    return spec


def validate_preparation():
    if not specification_matches(
        read_json(OUTPUT / "preparation_protocol.json"), preparation_design()
    ):
        raise ValueError("Independent task, objects or construction changed after registration")


def qualify_object(sut_id):
    sut = BY_ID[sut_id]
    root = QUALIFICATION_ROOT / sut.sut_id
    path = root / "qualification.json"
    if path.exists():
        report = read_json(path)
        if report["parameters"] != sut.parameters:
            raise ValueError("Independent qualification parameters changed")
        return report
    report = {
        "sut_id": sut.sut_id,
        "lineage": sut.lineage,
        "parameters": sut.parameters,
        **qualify(lambda: create(sut.sut_id), save_root=root),
    }
    write_json(path, report)
    return report


def qualify_nonlearning():
    validate_preparation()
    objects = [s for s in INDEPENDENT_SUTS if s.family != "learning"]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(qualify_object, s.sut_id) for s in objects]
        for future in as_completed(futures):
            report = future.result()
            rows.append(report)
            print(
                report["sut_id"],
                report["qualified"],
                [r["course"] for r in report["courses"] if not r["passed"]],
                flush=True,
            )
    result = {
        "status": "complete",
        "qualified": all(r["qualified"] for r in rows),
        "objects": rows,
        "physical_course_calls": 5 * len(rows),
    }
    write_json(OUTPUT / "nonlearning_qualification.json", result)
    return result


def train_policy(sut_id):
    validate_preparation()
    if not read_json(OUTPUT / "nonlearning_qualification.json")["qualified"]:
        raise ValueError("Independent nonlearning capabilities must pass first")
    sut = BY_ID[sut_id]
    path = CHECKPOINT_ROOT / sut.training_run / "training.json"
    if path.exists():
        saved = read_json(path)
        if saved["status"] not in ("qualified", "unqualified"):
            raise ValueError("An interrupted independent policy run needs inspection")
        return saved
    return train(sut_id)


def train_policies():
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(train_policy, s.sut_id) for s in INDEPENDENT_SUTS if s.family == "learning"
        ]
        for future in as_completed(futures):
            rows.append(future.result())
            write_json(
                OUTPUT / "policy_training.json",
                {"status": "complete" if len(rows) == 2 else "running", "runs": rows},
            )
    return rows


def collect_interaction():
    validate_preparation()
    kind = "cut_in_interaction"
    rows = []
    for role in ("history", "development"):
        scenarios = sample(kind, 512, role)
        for sut in SUTS:
            if sut.role != role:
                continue
            bank = collect_bank(
                sut.sut_id,
                scenarios,
                TEST_RECORDS_ROOT / "physical" / role / kind / sut.sut_id,
                workers=16,
            )
            executions = [bank(s) for s in scenarios]
            provenance = {
                "qualified": True,
                "bank": str(bank.root.relative_to(TEST_RECORDS_ROOT)),
                "contract": bank.manifest["contract"],
                "environment_seed": 81001,
                "physical_calls": bank.manifest["attempted_calls"],
            }
            archive = ResponseArchive(
                sut.sut_id,
                sut.family,
                sut.lineage,
                role,
                scenarios,
                np.asarray([e.feedback.collision for e in executions]),
                np.asarray(
                    [
                        [e.feedback.safety.min_clearance_m, e.feedback.safety.min_ttc_s]
                        for e in executions
                    ]
                ),
                provenance=provenance,
            )
            path = TEST_RECORDS_ROOT / role / kind / f"{sut.sut_id}.npz"
            if path.exists():
                existing = load_archive(path)
                if (
                    existing.provenance != provenance
                    or existing.scenarios != scenarios
                    or not np.array_equal(existing.collision, archive.collision)
                ):
                    raise ValueError("Interaction archive differs from its paid physical bank")
            else:
                save_archive(path, archive, provenance=provenance)
            rows.append({"source_id": sut.sut_id, "role": role, "manifest": bank.manifest})
            write_json(
                OUTPUT / "interaction_construction.json",
                {
                    "status": "complete" if len(rows) == 12 else "running",
                    "banks": rows,
                    "physical_calls": sum(r["manifest"]["attempted_calls"] for r in rows),
                },
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "qualify", "policies", "interaction"))
    stage = parser.parse_args().stage
    {
        "freeze": freeze,
        "qualify": qualify_nonlearning,
        "policies": train_policies,
        "interaction": collect_interaction,
    }[stage]()


if __name__ == "__main__":
    main()
