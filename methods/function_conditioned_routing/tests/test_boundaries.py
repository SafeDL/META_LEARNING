from pathlib import Path


def test_base_packages_do_not_import_the_method_chain() -> None:
    root = Path(__file__).resolve().parents[3]
    forbidden = "methods.function_conditioned_routing"
    for package in ("highway_sim_env", "replications"):
        for path in (root / package).rglob("*.py"):
            assert forbidden not in path.read_text(encoding="utf-8")
