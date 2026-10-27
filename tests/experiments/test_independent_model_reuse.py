import copy

import pytest

from experiments import independent_models as construction
from testing_methods.history_transfer.domain_training import training_contract


@pytest.mark.parametrize(
    "change",
    (
        "source_weighting",
        "history_scope",
        "history_order",
        "target_computation",
        "numerical_dependency",
    ),
)
def test_target_reuse_distinguishes_weighting_changes_from_changed_training(
    tmp_path, monkeypatch, change
):
    contract = training_contract()
    files = {name: (construction.ROOT / name).read_bytes() for name in contract}
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    original = {
        "history_fingerprints": {"h1": "source-a", "h2": "source-b"},
        "development_fingerprints": {"dev": "source-c"},
        "training_contract": contract,
        "config": {"steps": 1500},
        "held_family": None,
    }
    expected = copy.deepcopy(original)
    monkeypatch.setattr(construction, "ROOT", tmp_path)
    if change == "source_weighting":
        path = tmp_path / "testing_methods/common/response.py"
        content = path.read_bytes().replace(
            b"return np.ones(len(archives), dtype=np.float64) / len(archives)",
            b"return np.arange(len(archives), dtype=np.float64)",
        )
        assert content != path.read_bytes()
        path.write_bytes(content)
        construction.check_target_reuse(original, expected, files)
    else:
        if change == "history_scope":
            expected["history_fingerprints"]["h2"] = "different-source-data"
        elif change == "history_order":
            expected["history_fingerprints"] = dict(
                reversed(original["history_fingerprints"].items())
            )
        elif change == "target_computation":
            path = tmp_path / "testing_methods/history_transfer/domain_training.py"
            path.write_bytes(path.read_bytes().replace(b"loss.backward()", b"pass", 1))
        else:
            path = tmp_path / "testing_methods/common/gp_classifier.py"
            path.write_bytes(path.read_bytes() + b"\n# numerical dependency changed\n")
        with pytest.raises(ValueError):
            construction.check_target_reuse(original, expected, files)
