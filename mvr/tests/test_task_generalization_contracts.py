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


def test_logical_domain_intervals_are_numerically_disjoint_on_active_dimensions() -> None:
    tasks = load_taskbook("mvr/configs/taskbook.json")
    representative = {}
    for task in tasks:
        representative.setdefault((task.functional_scenario, task.logical_split), task)
    for family in {task.functional_scenario for task in tasks}:
        train = representative[family, "train"]
        validation = representative[family, "validation"]
        test = representative[family, "test"]
        for left, right in ((train, validation), (train, test), (validation, test)):
            for index, name in enumerate(left.logical_domain_bounds):
                if left.logical_parameter_mask[index] and right.logical_parameter_mask[index]:
                    left_bounds = left.logical_domain_bounds[name]
                    right_bounds = right.logical_domain_bounds[name]
                    assert max(left_bounds[0], right_bounds[0]) > min(left_bounds[1], right_bounds[1])
        assert len(train.logical_parameter_mask) == (6 if family == "cutin" else 5)
