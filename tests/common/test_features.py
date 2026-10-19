"""Shared numerical features retain the archived computation formulas."""

import ast
from pathlib import Path
from zipfile import ZipFile


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
