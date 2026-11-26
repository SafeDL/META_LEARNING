"""Admit K-shot from fixed defaults, native data quality and historical models.

No independent ADS development experiment or later experiment is required.
"""

import argparse
import ast
import hashlib
import inspect
import textwrap
from pathlib import Path

from controllers.highway.catalog import FORMAL_HISTORY_SUTS, FORMAL_SUTS, FORMAL_TARGET_SUTS
from controllers.highway.contracts import runtime_contract

from . import history_transfer_experiment as study
from .audit_transfer_history import audit as audit_history
from .audit_transfer_targets import verify as verify_targets
from experiments.scene_design.check_formal import verify_development
from .executions import ExecutionBank, require_qualified
from .history_transfer_summary import register_statistics
from .method_contracts import method_contract
from .protocol import ROOT
from .storage import read_json
from .transfer_batches import ACCEPTANCE
from .transfer_execution import execution_contract
from .transfer_settings import REGISTRY, load_settings


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(path):
    path = Path(path).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError("Admission evidence must stay within the workspace")
    return path.relative_to(ROOT).as_posix()


def _check_entry():
    # These wrappers have no defaults capable of bypassing the approved runner.
    for name, mode in (("run_k_shot", "prediction"),):
        function = getattr(study, name)
        parsed = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        expected = ast.parse(
            f"def entry(root):\n from .transfer_batches import run\n return run('{mode}', root)\n"
        ).body[0]
        if ast.dump(
            ast.Module(body=parsed.body, type_ignores=[]), include_attributes=False
        ) != ast.dump(ast.Module(body=expected.body, type_ignores=[]), include_attributes=False):
            raise ValueError("Formal commands have not migrated to the approved batch runner")
    nodes = list(ast.walk(ast.parse(textwrap.dedent(inspect.getsource(study.readiness)))))
    imports = [
        node
        for node in nodes
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "transfer_admission"
    ]
    names = [
        alias.asname or alias.name
        for node in imports
        for alias in node.names
        if alias.name == "readiness"
    ]
    if len(names) != 1 or not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == names[0]
        for node in nodes
    ):
        raise ValueError("Study readiness is not connected to evidence-based admission")


def _contracts():
    methods = (*study.PREDICTION_METHODS, "source_prior")
    return {method: method_contract(method) for method in methods}


def _implementation_files():
    # Include the paid risk transforms and numerical helpers imported below the
    # coarse method contracts; a change there must also invalidate admission.
    directories = (
        "controllers/highway",
        "simulator/highway",
        "testing_methods/common",
        "testing_methods/history_transfer",
        "testing_methods/baselines/response_baselines",
    )
    files = {path for name in directories for path in (ROOT / name).rglob("*.py")}
    files.add(ROOT / "testing_methods/selection.py")
    files.update(
        Path(__file__).with_name(name)
        for name in (
            "parallel.py",
            "domain.py",
            "domain_risk.py",
            "execution_events.py",
            "initialization.py",
            "costs.py",
            "failure_metrics.py",
            "k_shot.py",
            "paired_statistics.py",
            "source_registration.py",
        )
    )
    return files


# Operational ceilings; measured costs are saved by actual K-shot tasks.
LIMITS = {"wall_s": 7200, "child_memory_bytes": 8 * 1024**3}


def require_experiment(mode):
    if mode not in study.ACTIVE_EXPERIMENTS:
        raise ValueError("Only K-shot prediction is active; later experiments are deferred")


def collect_evidence(root=study.OUTPUT, *, recheck_paid=True):
    root = Path(root)
    load_settings(root)  # No formal target access before this succeeds.
    _check_entry()
    verify_development()
    for sut in FORMAL_SUTS:
        require_qualified(sut)
    files = {
        root / REGISTRY,
        root / "reports/protocol.json",
        root / "reports/execution_registration.json",
    }
    from .formal_scenarios import DEVELOPMENT_ROOT

    files.update(DEVELOPMENT_ROOT.rglob("*.json"))
    files.update(DEVELOPMENT_ROOT.rglob("*.json.gz"))
    files.add(ROOT / "experiments/scene_design/check_braking.py")
    from experiments.scene_design.check_formal import REPLAY_PATH, evidence

    files.add(ROOT / "experiments/scene_design/check_formal.py")
    files.add(REPLAY_PATH)
    for directory, _, _ in evidence().values():
        files.update(directory.rglob("*.json"))
        files.update(directory.rglob("*.json.gz"))
    files.update(
        Path(ROOT / name) for contract in _contracts().values() for name in contract["source_files"]
    )
    files.update(_implementation_files())
    files.update(
        Path(ROOT / name) for sut in FORMAL_SUTS for name in runtime_contract(sut)["source_files"]
    )
    files.add(ROOT / "controllers/highway/catalog.py")
    for sut in FORMAL_SUTS:
        from .qualification import QUALIFICATION_ROOT

        directory = QUALIFICATION_ROOT / sut.sut_id
        files.update(directory.glob("*.json"))
        files.update(directory.glob("*.json.gz"))
        if sut.family == "learning":
            from controllers.highway.learning import CHECKPOINT_ROOT

            files.update(
                (CHECKPOINT_ROOT / sut.training_run / name)
                for name in ("best.zip", "training.json")
            )
    for kind in study.scenarios_for(root):
        _, history = study.models_and_history(root, kind)
        study.registered_history_views(root, kind, history)
        directory = study.model_directory(root, kind)
        files.update(directory.glob("*.pt"))
        files.update(directory.glob("*.json"))
        files.update(
            [
                root / "reports/history_views" / f"{kind}.json",
                root / "reports/models/adaptive_mixture" / f"{kind}.json",
            ]
        )
        files.update((root / "data/history" / kind).glob("*.npz"))
        files.add(root / "data/history" / kind / "manifest.json")
        for split in ("history", "target"):
            files.add(root / "candidates" / split / kind / "manifest.json")
        for archive in history:
            bank = ExecutionBank(root / "physical/history" / kind / archive.source_id)
            files.add(bank.root / "manifest.json")
            files.update(bank.execution_path(i) for i in range(len(bank.candidates)))
    # Historical audit only reuses reports bound to the current native bank;
    # target quality must already have been executed by its explicit guarded CLI.
    historical = (
        audit_history(root)
        if recheck_paid
        else read_json(root / "reports/history_quality/summary.json")
    )
    if historical["status"] != "complete" or historical["records"] != len(
        FORMAL_HISTORY_SUTS
    ) * sum(study.HISTORY_COUNTS[k] for k in study.scenarios_for(root)):
        raise ValueError("Complete historical quality evidence is missing")
    files.update((root / "reports/history_quality").rglob("*.json"))
    target = verify_targets(root)
    if (
        target["summary"]["banks"] != len(FORMAL_TARGET_SUTS) * len(study.scenarios_for(root))
        or target["summary"]["records"]
        != len(FORMAL_TARGET_SUTS) * len(study.scenarios_for(root)) * study.TARGET_COUNT
    ):
        raise ValueError("Complete formal-target quality evidence is missing")
    files.update(Path(path) for path in target["files"])
    register_statistics(root)
    files.update(
        [
            root / "reports/statistics_protocol.json",
            root / "reports/history_view_protocol.json",
        ]
    )
    files.update(
        Path(__file__).with_name(name)
        for name in (
            "transfer_admission.py",
            "audit_transfer_targets.py",
            "transfer_batches.py",
            "transfer_prediction_audit.py",
            "transfer_settings.py",
            "formal_scenarios.py",
            "history_transfer_summary.py",
            "collect_transfer_records.py",
            "record_audit.py",
            "report_history_transfer.py",
        )
    )
    return {
        "files": sorted({_relative(path) for path in files}),
        "execution_contract": execution_contract(LIMITS),
    }


def publish(root=study.OUTPUT):
    root = Path(root)
    evidence = collect_evidence(root)
    declaration = {
        "status": "complete",
        "scope": "K-shot inputs and data quality; no separate development experiment",
        "allowed_experiments": list(study.ACTIVE_EXPERIMENTS),
        "study_root": _relative(root),
        "execution_contract": evidence["execution_contract"],
        "entry_source_sha256": sha(Path(__file__).with_name("transfer_batches.py")),
        "settings_sha256": sha(root / REGISTRY),
        "loader_sha256": sha(__file__),
        "method_contracts": _contracts(),
        "runtime_contracts": {sut.sut_id: runtime_contract(sut) for sut in FORMAL_SUTS},
        "proof_sha256": {name: sha(ROOT / name) for name in evidence["files"]},
        "formal_queries_executed_by_publication": 0,
        "note": "Quality truth remains evaluator-owned; this certificate never enters a selector",
    }
    return study.freeze(root / ACCEPTANCE, declaration)


def load_acceptance(root=study.OUTPUT):
    root = Path(root)
    load_settings(root)
    _check_entry()
    certificate = read_json(root / ACCEPTANCE)
    if (
        certificate["status"] != "complete"
        or certificate["allowed_experiments"] != list(study.ACTIVE_EXPERIMENTS)
        or certificate["study_root"] != _relative(root)
        or certificate["loader_sha256"] != sha(__file__)
        or certificate["entry_source_sha256"]
        != sha(Path(__file__).with_name("transfer_batches.py"))
        or certificate["settings_sha256"] != sha(root / REGISTRY)
        or certificate["execution_contract"] != execution_contract(LIMITS)
        or certificate["method_contracts"] != _contracts()
        or certificate["runtime_contracts"]
        != {sut.sut_id: runtime_contract(sut) for sut in FORMAL_SUTS}
    ):
        raise ValueError("Admission differs from current settings, numerical code or ADS execution")
    mandatory = {
        _relative(root / REGISTRY),
        _relative(root / "reports/protocol.json"),
        _relative(root / "reports/target_quality/summary.json"),
        _relative(root / "reports/history_quality/summary.json"),
        _relative(root / "reports/statistics_protocol.json"),
        _relative(__file__),
    }
    if not mandatory.issubset(certificate["proof_sha256"]):
        raise ValueError("Admission certificate omits required stage evidence")
    for name, digest in certificate["proof_sha256"].items():
        path = ROOT / name
        if _relative(path) != name or sha(path) != digest:
            raise ValueError("Admission evidence changed after publication")
    expected = collect_evidence(root, recheck_paid=False)
    if (
        set(expected["files"]) != set(certificate["proof_sha256"])
        or expected["execution_contract"] != certificate["execution_contract"]
    ):
        raise ValueError("Admission omits or substitutes required stage evidence")
    for sut in FORMAL_SUTS:
        require_qualified(sut)
    return certificate


def readiness(root=study.OUTPUT):
    try:
        certificate = load_acceptance(root)
    except (FileNotFoundError, ValueError, KeyError) as error:
        return {"ready": False, "reasons": [str(error)]}
    return {
        "ready": True,
        "reasons": [],
        "certificate_sha256": sha(Path(root) / ACCEPTANCE),
        "execution_contract": certificate["execution_contract"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "publish"))
    args = parser.parse_args()
    if args.command == "publish":
        publish()
        print({"status": "published", "formal_queries": 0}, flush=True)
    else:
        print(readiness(), flush=True)
