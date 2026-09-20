from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPLICATIONS = ROOT / "replications"
RESULTS = ROOT / "results" / "highway_replications"
METHODS = (
    "adate_highway_env",
    "detour_highway_env",
    "fst_highway_env",
    "scenariofuzz_highway_env",
)


def test_method_packages_reuse_the_project_highway_environment() -> None:
    for name in METHODS:
        package = REPLICATIONS / name
        source = "\n".join(
            path.read_text(encoding="utf-8") for path in package.rglob("*.py")
        )
        assert "diva_highway_env" in source
        assert not (package / "results").exists()
        assert not list(package.rglob("__pycache__"))


def test_results_have_one_semantic_root_per_method() -> None:
    expected = {"shared", "adate", "detour", "fst", "scenariofuzz", "evaluation"}
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
