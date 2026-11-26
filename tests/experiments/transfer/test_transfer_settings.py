"""Fixed default registration does not depend on an ADS development experiment."""

import pytest

from experiments import transfer_settings as module
from experiments.storage import write_json
from testing_methods.history_transfer.inference import TARGET_PARAMETERS


def test_native_settings_keep_shared_raw_defaults_and_ten_steps():
    settings = module.native_settings()
    assert settings.online.raw.steps == settings.online.learned.steps == settings.pooled.steps == 10
    assert settings.online.raw.learning_rate == 0.01
    assert settings.online.learned.learning_rate == settings.pooled.learning_rate == 0.003
    assert settings.online.raw.parameters == TARGET_PARAMETERS
    assert len(TARGET_PARAMETERS) == 6
    assert settings.online.raw_initial_mean == -1.4


def test_settings_register_without_data_models_or_development(tmp_path):
    registered = module.register_settings(tmp_path)
    assert module.load_settings(tmp_path) == module.native_settings()
    assert module.register_settings(tmp_path) == registered
    assert not registered["formal_target_labels_read"]
    assert not registered["opens_formal_evaluation"]
    assert [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()] == [
        module.REGISTRY
    ]


@pytest.mark.parametrize("field", ["steps", "prior", "opens", "implementation"])
def test_changed_defaults_cannot_be_loaded_as_registered_native_settings(tmp_path, field):
    saved = module.register_settings(tmp_path)
    if field == "steps":
        saved["settings"]["online"]["raw"]["steps"] = 400
    elif field == "prior":
        saved["settings"]["online"]["raw_initial_mean"] = 0
    elif field == "opens":
        saved["opens_formal_evaluation"] = True
    else:
        saved["loader_sha256"] = "changed"
    write_json(tmp_path / module.REGISTRY, saved)
    with pytest.raises(ValueError, match="fixed defaults"):
        module.load_settings(tmp_path)


def test_settings_alone_do_not_open_k_shot(tmp_path):
    from experiments.transfer_admission import readiness

    module.register_settings(tmp_path)
    assert not readiness(tmp_path)["ready"]
