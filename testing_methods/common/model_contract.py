"""Construction provenance for models reused between development and final studies."""

import hashlib
from importlib.metadata import version
from pathlib import Path

from experiments.source_registration import source_contract_matches

ROOT = Path(__file__).resolve().parents[2]


def model_contract(method):
    common = (
        "experiments/schema.py",
        "experiments/scenarios.py",
        "testing_methods/common/response.py",
        "testing_methods/common/model_contract.py",
    )
    if method != "fsbo":
        raise ValueError("Unknown model construction contract")
    specific = (
        "testing_methods/common/fsbo/method.py",
        "testing_methods/common/gp_classifier.py",
        "testing_methods/common/classification.py",
    )
    return {
        "method": method,
        "source_files": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (*common, *specific)
        },
        "packages": {name: version(name) for name in ("numpy", "scipy", "torch")},
    }


def validate_model_contract(checkpoint, method, *, configuration=None):
    recorded = checkpoint["model_contract"]
    current = model_contract(method)
    if (
        recorded["method"] != current["method"]
        or recorded["packages"] != current["packages"]
        or not source_contract_matches(recorded["source_files"], current["source_files"])
    ):
        raise ValueError("Model construction code changed; rebuild the checkpoint before reuse")
    if configuration is not None and checkpoint["config"] != configuration:
        raise ValueError("Model configuration changed; rebuild the checkpoint before reuse")
