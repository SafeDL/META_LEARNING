"""Shared numerical features retain the archived computation formulas."""

import ast
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import torch


def test_shared_feature_formulas_match_recorded_sources():
    project = Path(__file__).resolve().parents[2]
    current = ast.parse(
        (project / "methods/common/features.py").read_text(encoding="utf-8")
    )
    functions = {
        node.name: node for node in current.body if isinstance(node, ast.FunctionDef)
    }
    with ZipFile(
        project / "results/baselines/lrt/by_family/frozen_inputs.zip"
    ) as archive:
        for module, function in (
            ("methods/lrt/data.py", "assemble"),
            ("methods/lrt/evaluation/by_family.py", "prepare_pool"),
        ):
            original = ast.parse(archive.read(module))
            expected = next(
                node
                for node in original.body
                if isinstance(node, ast.FunctionDef) and node.name == function
            )
            assert ast.dump(functions[function]) == ast.dump(expected)


def test_pool_cache_builds_without_target_feedback_and_reloads(tmp_path, monkeypatch):
    from methods.common import features
    from methods.baselines.hgt.history import load_history

    history = load_history()
    x = next(iter(history.values()))["x"]
    bank = tmp_path / "candidates.npz"
    np.savez(bank, x=np.concatenate((x[:18], x[1024:1030])))
    history = {
        name: {key: values[:24] for key, values in source.items()}
        for name, source in history.items()
    }
    workspace = SimpleNamespace(
        root=tmp_path / "cutin",
        family=0,
        history=history,
        kernel_root=tmp_path / "kernel",
        encoder_root=tmp_path / "encoder",
    )
    monkeypatch.setattr(features, "bank_path", lambda pool, sut: bank)
    monkeypatch.setattr(
        features,
        "load_kernel",
        lambda *args, **kwargs: SimpleNamespace(
            covariance=lambda x: torch.eye(len(x), dtype=torch.float64) * 0.25
        ),
    )
    monkeypatch.setattr(
        features,
        "load_encoder",
        lambda *args, **kwargs: SimpleNamespace(
            covariance=lambda x, context, base: torch.as_tensor(base)
        ),
    )
    monkeypatch.setattr(features, "fixed_context", lambda *args: {})
    built = features.prepare_pool(workspace, 0)
    np.testing.assert_array_equal(built["prior"].x, x[:18])
    assert np.isfinite(built["prior"].means).all()
    assert len(built["common"]["initial_indices"]) == 10
    assert (workspace.root / "pool_cache/pool_0.json").exists()
    bank.unlink()
    cached = features.prepare_pool(workspace, 0)
    np.testing.assert_array_equal(cached["prior"].means, built["prior"].means)
    assert cached["preparation_s"] == built["preparation_s"]
