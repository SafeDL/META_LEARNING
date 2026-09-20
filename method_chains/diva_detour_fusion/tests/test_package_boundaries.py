from __future__ import annotations

from pathlib import Path

from method_chains.diva_detour_fusion.experiment import OUTPUT_DIR


CHAIN = Path(__file__).resolve().parents[1]
REPOSITORY = CHAIN.parents[1]


def test_fusion_chain_owns_source_and_uses_canonical_results():
    assert OUTPUT_DIR == REPOSITORY / "results" / "method_chains" / CHAIN.name
    assert (CHAIN / "fusion.py").is_file()
    assert (CHAIN / "config.py").is_file()
    assert (CHAIN / "experiment.py").is_file()
    assert (CHAIN / "replay.py").is_file()
    assert (CHAIN / "ownership_manifest.json").is_file()
    assert (OUTPUT_DIR / "summary.json").is_file()
    assert (OUTPUT_DIR / "response_bank.npz").is_file()
    assert not (CHAIN / "results").exists()


def test_base_and_standalone_replication_do_not_own_fusion_modules():
    forbidden = (
        REPOSITORY / "mvr",
        REPOSITORY / "diva_highway_env" / "config.py",
        REPOSITORY / "diva_highway_env" / "diva" / "detour_transfer.py",
        REPOSITORY / "diva_highway_env" / "experiments" / "run_multifunction_detour_fusion.py",
        REPOSITORY / "diva_highway_env" / "scripts" / "render_multifunction_gifs.py",
        REPOSITORY / "results" / "diva_highway" / "detour_multifunction",
    )
    assert not [path for path in forbidden if path.exists()]
    assert (REPOSITORY / "replications" / "detour_highway_env").is_dir()


def test_dependency_direction_is_one_way_into_the_fusion_chain():
    base = REPOSITORY / "diva_highway_env"
    reverse_imports = [
        path
        for path in base.rglob("*.py")
        if "method_chains.diva_detour_fusion" in path.read_text(encoding="utf-8")
    ]
    assert not reverse_imports

    fusion_source = (CHAIN / "fusion.py").read_text(encoding="utf-8")
    assert "diva_highway_env" in fusion_source
    assert "replications.detour_highway_env" in fusion_source
