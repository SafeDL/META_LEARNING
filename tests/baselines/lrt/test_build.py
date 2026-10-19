"""Cold imports and construction from historical data without pretrained artifacts."""

import subprocess
import sys

import pytest

from methods.baselines.hgt.io import read_json, write_json
from methods.baselines.lrt import build


@pytest.mark.parametrize(
    "modules",
    (
        (
            "methods.baselines.cg_brt.reference",
            "methods.baselines.dart",
            "methods.baselines.lrt.selection",
        ),
        (
            "methods.baselines.dart",
            "methods.baselines.lrt.selection",
            "methods.baselines.cg_brt",
        ),
    ),
)
def test_component_imports_do_not_depend_on_import_order(modules):
    code = "import importlib\n" + "\n".join(
        f"importlib.import_module({name!r})" for name in modules
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True)


def test_build_needs_history_but_no_existing_weights_or_experiment_snapshots(
    tmp_path, monkeypatch
):
    history = {"historical_source": {"risk": [0.2, 0.8]}}
    kernel = tmp_path / "kernel.json"
    encoder_root = tmp_path / "encoder"
    encoder = encoder_root / "models/decision_11.pt"
    value = tmp_path / "lrt/models/full_11.json"
    monkeypatch.setattr(build, "ROOT", tmp_path / "lrt")
    monkeypatch.setattr(build, "ENCODER_ROOT", encoder_root)
    monkeypatch.setattr(build, "SEEDS", (11,))
    monkeypatch.setattr(build, "load_history", lambda: history)

    def train_kernel(data, seed):
        assert data is history
        write_json(kernel, {"seed": seed})

    def prepare_cases(data, seed):
        assert data is history and read_json(kernel)["seed"] == seed
        return {"historical_cases": data}

    def train_encoder(seed, mode, cases):
        assert cases["historical_cases"] is history
        assert mode == "decision"
        write_json(encoder, {"seed": seed})
        write_json(
            encoder_root / "models/training_decision_11.json", {"status": "complete"}
        )

    def train_value(seed, mode):
        assert read_json(encoder)["seed"] == seed and mode == "full"
        write_json(value, {"seed": seed})

    monkeypatch.setattr(build, "train_kernel", train_kernel)
    monkeypatch.setattr(build, "prepare_cases", prepare_cases)
    monkeypatch.setattr(build, "train_encoder", train_encoder)
    monkeypatch.setattr(build, "train_value", train_value)
    build.main()
    assert read_json(value)["seed"] == 11
    assert read_json(tmp_path / "lrt/status.json")["status"] == "complete"

    def already_built(*args):
        pytest.fail("Existing encoder weights must avoid rebuilding training cases")

    monkeypatch.setattr(build, "prepare_cases", already_built)
    build.main()
