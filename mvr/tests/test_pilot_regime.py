from mvr.evaluation.regimes import PILOT_VALIDATION_REGIME, select_regime_tasks
from mvr.model import TransferableScenarioMiner
from mvr.scenario.catalog import mvr_parameter_spaces
from mvr.scenario.taskbook import load_taskbook


def test_framework_pilot_uses_only_validation_sut_and_geometry() -> None:
    tasks = select_regime_tasks(load_taskbook("mvr/configs/taskbook.json"), PILOT_VALIDATION_REGIME)
    assert {task.functional_scenario for task in tasks} == {"merge", "cutin", "roundabout"}
    assert all(task.sut_split == "validation" for task in tasks)
    assert all(task.geometry_split == "validation" for task in tasks)
    assert all(task.functional_split == "train" for task in tasks)


def test_default_model_option_head_matches_the_shared_option_contract() -> None:
    model = TransferableScenarioMiner(state_dim=11, map_dim=8)
    option_count = next(iter(mvr_parameter_spaces().values())).options
    assert model.universal_scene_policy.option_head[-1].out_features == len(option_count)
