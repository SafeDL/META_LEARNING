"""The fixed E02 matrix and implementation identities, reusing E01 inputs."""

import hashlib
from dataclasses import asdict

from controllers.highway.catalog import FORMAL_TARGET_SUTS

from experiments.history_transfer_experiment import OUTPUT as INPUT, RUN_SCENARIOS
from experiments.method_contracts import method_contract
from experiments.protocol import ROOT
from experiments.transfer_settings import native_settings

OUTPUT = ROOT / "results/budget_discovery"
METHODS = ("full", "target_only", "pooled_history", "random", "scenario_fuzz",
           "paired_system_active_evaluation", "static_prior")
SEED = 11
BUDGET = 200
INITIAL = 20
CHECKPOINTS = (20, 50, 100, 200)
# The E01 resource ceiling applies independently to each E02 task.
LIMITS = {"wall_s": 7200, "child_memory_bytes": 8 * 1024**3}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract(method):
    if method in ("full", "target_only", "pooled_history", "random", "scenario_fuzz"):
        identity = method_contract(method)
    elif method == "static_prior":
        identity = {**method_contract("full"), "method": method}
    elif method == "paired_system_active_evaluation":
        identity = {**method_contract("random"), "method": method}
        identity["source_files"].update({
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in (ROOT / "testing_methods/baselines/paired_system_active_evaluation").glob("*.py")
        })
        identity["source_files"]["testing_methods/common/response.py"] = sha(
            ROOT / "testing_methods/common/response.py"
        )
    else:
        raise ValueError("Method is outside the registered E02 matrix")
    identity["source_files"].update({
        f"experiments/budget/{name}": sha(ROOT / "experiments/budget" / name)
        for name in ("protocol.py", "execution.py", "selectors.py")
    })
    return identity


def declaration():
    return {
        "experiment": "budget_discovery", "input_root": str(INPUT.relative_to(ROOT)),
        "scenarios": list(RUN_SCENARIOS), "targets": [s.sut_id for s in FORMAL_TARGET_SUTS],
        "methods": list(METHODS), "seed": SEED, "budget": BUDGET, "initial_count": INITIAL,
        "checkpoints": list(CHECKPOINTS), "settings": native_settings().to_dict(),
        "limits": LIMITS, "main_metrics": ["unique_collisions", "discovery_auc"],
        "discovery_auc": "mean of cumulative unique collision counts at queries 1..200",
        "physical_executions": 0, "primary_comparison": "target_only",
        "bootstrap": {"unit": "ADS", "replicates": 10000, "seed": 1301},
        "stop_after": "budget_discovery", "later_experiments_authorized": False,
    }
