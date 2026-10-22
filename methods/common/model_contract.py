"""Construction provenance for models reused between development and final studies."""

import hashlib
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def model_contract(method):
    common = ("benchmark/schema.py", "benchmark/scenarios.py", "methods/common/response.py",
              "methods/common/model_contract.py")
    specific = {
        "history_response": ("methods/gcrt/train.py", "methods/gcrt/prior.py", "methods/gcrt/config.py",
                             "methods/common/classification.py"),
        "fsbo": ("methods/baselines/fsbo/method.py", "methods/common/gp_classifier.py",
                 "methods/common/classification.py"),
        "pooled_ft": ("methods/baselines/pooled_ft/method.py",),
    }
    if method not in specific:
        raise ValueError("Unknown model construction contract")
    return {"method": method, "source_files": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (*common, *specific[method])},
        "packages": {name: version(name) for name in ("numpy", "scipy", "torch")}}


def validate_model_contract(checkpoint, method, *, configuration=None):
    if checkpoint.get("model_contract") != model_contract(method):
        raise ValueError("Model construction code changed; rebuild the checkpoint before reuse")
    if configuration is not None and checkpoint["config"] != configuration:
        raise ValueError("Model configuration changed; rebuild the checkpoint before reuse")
