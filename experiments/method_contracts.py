"""Implementation identities for source caches and formal method result reuse."""

import hashlib
from importlib.metadata import version

from .protocol import ROOT


def method_contract(method):
    paths = {
        "experiments/method_contracts.py",
        "experiments/parallel.py",
        "experiments/scenarios.py",
        "experiments/schema.py",
        "experiments/rss.py",
        "experiments/query.py",
        "experiments/executions.py",
        "experiments/resume.py",
        "experiments/history_transfer_experiment.py",
    }
    if method in ("target_only", "full", "pooled_history", "source_prior"):
        paths.update(
            {
                "testing_methods/history_transfer/inference.py",
                "testing_methods/history_transfer/adaptation.py",
                "testing_methods/history_transfer/model.py",
                "testing_methods/history_transfer/shared_private.py",
                "testing_methods/common/gp_classifier.py",
                "testing_methods/common/gp_laplace.py",
                "testing_methods/common/classification.py",
            }
        )
        if method in ("target_only", "full"):
            paths.add("testing_methods/history_transfer/selection.py")
        if method in ("full", "source_prior"):
            paths.update({"experiments/history.py", "testing_methods/common/response.py"})
        if method == "pooled_history":
            paths.update(
                {
                    "testing_methods/baselines/response_baselines/pooled_gp.py",
                    "testing_methods/baselines/response_baselines/grouped_gp.py",
                }
            )
    else:
        directory = {
            "random": "response_baselines",
            "av_bo_rf_ei": "av_bo_rf_ei",
            "samota": "samota",
            "scenario_fuzz": "scenario_fuzz",
        }[method]
        if method == "random":
            paths.add("testing_methods/baselines/response_baselines/collision.py")
        else:
            paths.update(
                p.relative_to(ROOT).as_posix()
                for p in (ROOT / "testing_methods/baselines" / directory).glob("*.py")
            )
        paths.add("experiments/baselines.py")
    packages = ["numpy", "scipy", "torch", "scikit-learn"]
    if method == "scenario_fuzz":
        packages.append("torch-geometric")
    return {
        "method": method,
        "source_files": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(paths)
        },
        "packages": {name: version(name) for name in packages},
    }
