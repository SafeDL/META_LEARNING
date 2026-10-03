from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPLICATIONS = ROOT / "replications"
RESULTS = ROOT / "results" / "highway_replications"
SUT_ALGORITHMS = ROOT / "sut_algorithms"
METHODS = (
    "adate_highway_env",
    "detour_highway_env",
    "fst_highway_env",
    "highway_sut_selection",
    "scenariofuzz_highway_env",
)


def test_method_packages_reuse_the_project_highway_environment() -> None:
    for name in METHODS:
        package = REPLICATIONS / name
        source = "\n".join(
            path.read_text(encoding="utf-8") for path in package.rglob("*.py")
        )
        assert "highway_sim_env" in source
        assert not (package / "results").exists()
        assert not list(package.rglob("__pycache__"))


def test_results_have_one_semantic_root_per_method() -> None:
    expected = {
        "shared",
        "adate",
        "detour",
        "fst",
        "scenariofuzz",
        "evaluation",
        "sut_selection",
    }
    assert {path.name for path in RESULTS.iterdir() if path.is_dir()} == expected
    assert not (ROOT / "results" / "highway_replications_v1").exists()
    version_name = re.compile(r"(?:^|[_-])v\d+(?:$|[_.-])", re.IGNORECASE)
    versioned = [
        item
        for root in (REPLICATIONS, RESULTS)
        for item in root.rglob("*")
        if version_name.search(item.name)
    ]
    assert not versioned


def test_sut_algorithms_have_one_root_outside_simulator_packages() -> None:
    assert {path.name for path in SUT_ALGORITHMS.iterdir() if path.is_dir()} == {
        "highway_env",
        "metadrive",
    }
    assert not (ROOT / "highway_sim_env" / "sut").exists()
    assert not (ROOT / "metadrive_sim_env" / "sut").exists()
    assert (SUT_ALGORITHMS / "highway_env" / "registry.py").is_file()
    assert (SUT_ALGORITHMS / "metadrive" / "registry.py").is_file()
