"""Audit retained sources after package moves and declared API extensions.

This mapping is used only by artifact verification. Runtime imports use the
current package names directly; the original archives remain unchanged.
"""

import ast
import re
from pathlib import Path
from zipfile import ZipFile


API_EXTENSIONS = {
    "methods/baselines/dart/encoder.py",
    "methods/baselines/dart/objectives.py",
    "methods/baselines/dart/selection.py",
    "methods/baselines/dart/train.py",
    "methods/baselines/hgt/baseline.py",
    "methods/baselines/hgt/search.py",
    "methods/baselines/hgt/train.py",
    "methods/lrt/data.py",
    "methods/lrt/prior.py",
    "methods/lrt/selection.py",
    "methods/lrt/teacher.py",
    "methods/lrt/train.py",
    "methods/lrt/cg_brt/reference.py",
    "methods/lrt/cg_brt/selection.py",
}
ADDED_SOURCES = {
    "methods/driving_benchmark/initialization.py",
    "methods/driving_benchmark/timing.py",
    "methods/lrt/workspace.py",
}
ADAPTIVE_INTERFACES = {
    "methods/lrt/selection.py",
    "methods/driving_benchmark/timing.py",
    "methods/driving_benchmark/risk_channel.py",
}


def verify_source(project, relative, original):
    """Check original bytes or the declared API revision frozen before evaluation.

    The original archives are retained. Only listed interface extensions may use
    the newer source snapshot; weights, data and unlisted code stay exact checks.
    Default query behavior is checked separately against the retained selectors.
    """
    relative = current_path(relative)
    actual = (Path(project) / relative).read_bytes()
    if original is not None and source_equal(actual, original, relative):
        return False
    if relative not in API_EXTENSIONS | ADDED_SOURCES | ADAPTIVE_INTERFACES or (
        original is None and relative not in ADDED_SOURCES
    ):
        raise ValueError(f"Frozen source or input changed: {relative}")
    snapshot = Path(project) / "results/baselines/lrt/by_family/frozen_inputs.zip"
    adaptive = Path(project) / "results/baselines/lrt/adaptive/reviewed_interfaces.zip"
    for name in {relative, *ADDED_SOURCES}:
        selected = (
            adaptive
            if adaptive.exists() and name in ADAPTIVE_INTERFACES | ADDED_SOURCES
            else snapshot
        )
        with ZipFile(selected) as archive:
            entries = {current_path(entry): entry for entry in archive.namelist()}
            entry = entries[name]
            expected = current_source(archive.read(entry), entry)
            if not source_equal((Path(project) / name).read_bytes(), expected, name):
                raise ValueError(f"Reviewed interface source changed: {name}")
    return True


RENAMES = (
    ("decision_aware_risk_transfer/selective", "lrt"),
    ("decision_aware_risk_transfer.selective", "lrt"),
    ("concordance_gated_bayesian_risk_transfer", "cg_brt"),
    ("decision_aware_risk_transfer", "dart"),
    ("history_guided_testing", "hgt"),
    ("historical_expert_mixture", "hem"),
    ("scale_adaptive_expert_mixture", "saem"),
)

BASELINES = ("cg_brt", "dart", "hem", "hgt", "ras_frt_uq", "response_baselines", "saem")


ORGANIZATION = (
    ("methods/lrt/cg_brt", "methods/baselines/cg_brt"),
    ("methods/lrt/workspace.py", "methods/common/workspace.py"),
    ("methods/lrt/prior.py", "methods/common/prior.py"),
    ("methods/lrt/student.py", "methods/common/student.py"),
    ("methods/counts.py", "methods/common/counts.py"),
    ("methods/catalog.py", "methods/common/catalog.py"),
    ("methods/snapshots.py", "methods/common/snapshots.py"),
    ("methods/lrt", "methods/baselines/lrt"),
    ("methods/driving_benchmark", "benchmark"),
    ("highway_sim_env", "simulation/highway"),
    ("metadrive_sim_env", "simulation/metadrive"),
    ("sut_algorithms/highway_env", "controllers/highway"),
    ("sut_algorithms", "controllers"),
    ("results/lrt/continual", "results/gcrt/full_horizon"),
    ("results/lrt", "results/baselines/lrt"),
    ("results/cg_brt", "results/baselines/cg_brt"),
    ("results/dart/independent_bank", "data/scenarios"),
    ("results/dart", "results/baselines/dart"),
    ("results/hgt/history", "data/history"),
    ("results/hgt", "results/baselines/hgt"),
    ("results/response_baselines", "results/baselines/response_baselines"),
    ("results/driving_benchmark", "results/benchmark"),
)


def current_path(name):
    for old, new in RENAMES:
        name = name.replace(old, new)
    for package in BASELINES:
        for area in ("methods", "tests"):
            name = name.replace(f"{area}/{package}", f"{area}/baselines/{package}")
            name = name.replace(f"{area}.{package}", f"{area}.baselines.{package}")
    for area in ("methods", "tests"):
        name = name.replace(f"{area}/baselines/cg_brt", f"{area}/lrt/cg_brt")
        name = name.replace(f"{area}.baselines.cg_brt", f"{area}.lrt.cg_brt")
    name = name.replace(
        "methods/baselines/dart/reference.py", "methods/lrt/cg_brt/reference.py"
    ).replace("methods.baselines.dart.reference", "methods.lrt.cg_brt.reference")
    for old, new in ORGANIZATION:
        if "/" in old:
            name = name.replace(old, new)
        else:
            name = name.replace(old + "/", new + "/")
        old_module = old.removesuffix(".py").replace("/", ".")
        new_module = new.removesuffix(".py").replace("/", ".")
        name = name.replace(old_module, new_module)
    return name


def current_source(data, name):
    """Translate recorded package moves and shared-function extraction only."""
    if not name.endswith(".py"):
        return data
    relative = current_path(name)
    source = current_path(data.decode("utf-8").replace("\r\n", "\n"))
    if relative == "methods/baselines/lrt/__init__.py":
        source = source.splitlines()[0] + "\n"
    if relative == "methods/baselines/dart/selection.py":
        source = source.replace(
            "from .reference import ReferencePolicy",
            "from methods.baselines.cg_brt.reference import ReferencePolicy",
        )
    depth = len(relative.split("/")) - 1
    source = re.sub(
        r"Path\(__file__\)\.resolve\(\)\.parents\[\d+\]",
        f"Path(__file__).resolve().parents[{depth}]",
        source,
    )
    if relative.startswith("methods/baselines/lrt/"):
        for old, new in (
            ("from .prior import", "from methods.common.prior import"),
            ("from ..prior import", "from methods.common.prior import"),
            ("from .student import", "from methods.common.student import"),
            ("from ..workspace import", "from methods.common.workspace import"),
            ("from ..cg_brt", "from methods.baselines.cg_brt"),
        ):
            source = source.replace(old, new)
    if relative == "methods/baselines/lrt/config.py":
        source = source.replace("NOISE = 0.0025\nJITTER = 1e-8\n", "")
        if "from methods.common.config import JITTER, NOISE" not in source:
            source = source.replace(
                "from methods.baselines.dart.config import SEEDS",
                "from methods.baselines.dart.config import SEEDS\n"
                "from methods.common.config import JITTER, NOISE",
            )
    if relative == "methods/baselines/lrt/data.py":
        lines = source.splitlines(keepends=True)
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == "assemble":
                del lines[node.lineno - 1 : node.end_lineno]
                break
        source = "".join(lines)
        if "from methods.common.features import assemble" not in source:
            source = source.replace(
                "from .config import (",
                "from methods.common.features import assemble\n\nfrom .config import (",
            )
    if relative == "methods/baselines/lrt/evaluation/by_family.py":
        lines = source.splitlines(keepends=True)
        moved = {"bank_path", "measured_seconds", "prepare_pool"}
        nodes = [
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name in moved
        ]
        for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
            del lines[node.lineno - 1 : node.end_lineno]
        source = "".join(lines)
        if "from benchmark.data import bank_path, measured_seconds" not in source:
            source = source.replace(
                "from ..config import ROOT as LRT_ROOT",
                "from benchmark.data import bank_path, measured_seconds\n"
                "from methods.common.features import prepare_pool\n"
                "from ..config import ROOT as LRT_ROOT",
            )
        if "from methods.common.snapshots import current_path" not in source:
            source = source.replace(
                "from methods.common.features import prepare_pool",
                "from methods.common.features import prepare_pool\nfrom methods.common.snapshots import current_path",
            )
        source = source.replace(
            "with ZipFile(destination) as archive:\n            for path in weights:",
            "with ZipFile(destination) as archive:\n            entries = {current_path(name): name for name in archive.namelist()}\n            for path in weights:",
        ).replace(
            "archive.read(path.relative_to(PROJECT).as_posix())",
            "archive.read(entries[path.relative_to(PROJECT).as_posix()])",
        )
    if relative in {
        "methods/baselines/hgt/history.py",
        "methods/baselines/ras_frt_uq/unified.py",
        "methods/baselines/response_baselines/calibration.py",
    }:
        source = source.replace(
            'ROOT / "history/responses.npz"', 'HISTORY / "responses.npz"'
        )
        source = source.replace(
            'BASELINE / "history/responses.npz"', 'HISTORY / "responses.npz"'
        )
        if "from benchmark.data import HISTORY" not in source:
            position = source.find("\nimport numpy")
            if position < 0:
                position = source.find("\nfrom ")
            source = (
                source[:position]
                + "\nfrom benchmark.data import HISTORY\n"
                + source[position:]
            )
    return source.encode("utf-8")


def source_equal(actual, expected, name):
    if actual == expected:
        return True
    if not name.endswith(".py"):
        return False
    try:
        return ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
    except SyntaxError:
        return False


API_EXTENSIONS = {current_path(name) for name in API_EXTENSIONS}
ADDED_SOURCES = {current_path(name) for name in ADDED_SOURCES}
ADAPTIVE_INTERFACES = {current_path(name) for name in ADAPTIVE_INTERFACES}
