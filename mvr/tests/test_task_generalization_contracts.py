from __future__ import annotations

from mvr.scenario.taskbook import load_taskbook


def test_taskbook_has_disjoint_sut_geometry_and_logical_domain_axes() -> None:
    tasks = load_taskbook("mvr/configs/taskbook.json")
    train_suts = {task.sut_ref for task in tasks if task.sut_split == "train"}
    test_suts = {task.sut_ref for task in tasks if task.sut_split == "test"}
    train_geometry = {task.geometry_hash for task in tasks if task.geometry_split == "train"}
    test_geometry = {task.geometry_hash for task in tasks if task.geometry_split == "test"}
    train_domains = {task.logical_domain_id for task in tasks if task.logical_split == "train"}
    test_domains = {task.logical_domain_id for task in tasks if task.logical_split == "test"}
    assert train_suts.isdisjoint(test_suts)
    assert train_geometry.isdisjoint(test_geometry)
    assert train_domains.isdisjoint(test_domains)
    assert {task.functional_scenario for task in tasks} == {"merge", "cutin", "roundabout"}
