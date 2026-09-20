from pathlib import Path


def test_base_packages_do_not_import_the_method_chain() -> None:
    root = Path(__file__).resolve().parents[3]
    forbidden = "method_chains.diva_function_conditioned_routing"
    for package in ("diva_highway_env", "replications"):
        for path in (root / package).rglob("*.py"):
            assert forbidden not in path.read_text(encoding="utf-8")
