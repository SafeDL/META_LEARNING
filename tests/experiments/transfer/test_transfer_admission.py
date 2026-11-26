"""Admission envelope and routing tests; stage verifiers are explicit unit stubs."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import transfer_admission as module
from experiments.storage import write_json


def prediction_route(root):
    from .transfer_batches import run

    return run("prediction", root)


def budget_route(root):
    from .transfer_batches import run

    return run("budget", root)


def readiness_route(root):
    from .transfer_admission import readiness as method_readiness

    return method_readiness(root)


def legacy_route(root):
    return {"ready": True}


def test_missing_sealed_settings_closes_admission_without_target_access(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module, "verify_targets", lambda root: pytest.fail("No target access before settings")
    )
    root = tmp_path / "not_created"
    assert not module.readiness(root)["ready"]
    with pytest.raises(FileNotFoundError):
        module.publish(root)
    assert not root.exists()


def test_admission_also_binds_paid_risk_transforms_and_shared_numerics():
    files = module._implementation_files()
    names = {path.relative_to(module.ROOT).as_posix() for path in files}
    assert {
        "experiments/domain_risk.py",
        "experiments/initialization.py",
        "experiments/failure_metrics.py",
        "testing_methods/selection.py",
        "testing_methods/history_transfer/inference.py",
    }.issubset(names)
    assert all(path.is_file() for path in files)


@pytest.mark.parametrize("bad", [None, "prediction", "readiness"])
def test_only_the_approved_actual_routes_and_guard_are_accepted(monkeypatch, bad):
    routes = {
        "run_k_shot": prediction_route,
        "run_budget": budget_route,
        "readiness": readiness_route,
    }
    if bad is not None:
        routes[
            {"prediction": "run_k_shot", "budget": "run_budget", "readiness": "readiness"}[bad]
        ] = legacy_route
    monkeypatch.setattr(module, "study", SimpleNamespace(**routes))
    if bad is None:
        module._check_entry()
    else:
        with pytest.raises(ValueError, match="migrated|not connected"):
            module._check_entry()


@pytest.fixture
def envelope(tmp_path, monkeypatch):
    # No actual stage is represented as scientifically complete here. These
    # dependency stubs exercise only the certificate's exact file/identity rules.
    original_source = Path(module.__file__)
    source = tmp_path / "transfer_admission.py"
    source.write_bytes(original_source.read_bytes())
    (tmp_path / "transfer_batches.py").write_text(
        "# Explicit unit entry identity\n", encoding="utf-8"
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "__file__", str(source))
    monkeypatch.setattr(module, "load_settings", lambda root: None)
    monkeypatch.setattr(module, "_check_entry", lambda: None)
    monkeypatch.setattr(module, "_contracts", lambda: {"scope": "unit dependencies"})
    monkeypatch.setattr(module, "FORMAL_SUTS", ())
    study = tmp_path / "unit_study"
    scale = tmp_path / "unit_scale"
    names = [
        study / module.REGISTRY,
        study / "reports/protocol.json",
        study / "reports/target_quality/summary.json",
        study / "reports/history_quality/summary.json",
        study / "reports/control_design.json",
        study / "reports/statistics_protocol.json",
        study / "unit_extra_required.json",
    ]
    for path in names:
        write_json(path, {"scope": "unit envelope placeholder; not experimental evidence"})
    proof_names = [module._relative(path) for path in (*names, source)]
    contract = module.execution_contract(module.LIMITS)
    calls = []

    def dependencies(root, *, recheck_paid=True):
        calls.append(recheck_paid)
        return {
            "files": proof_names,
            "execution_contract": contract,
        }

    monkeypatch.setattr(module, "collect_evidence", dependencies)
    return study, scale, calls


def test_published_envelope_is_loaded_only_with_its_complete_identity(envelope):
    study, scale, calls = envelope
    certificate = module.publish(study)
    assert certificate == module.load_acceptance(study)
    assert calls == [True, False]
    assert module.readiness(study)["ready"]
    assert certificate["formal_queries_executed_by_publication"] == 0


def test_batch_resource_limits_require_the_complete_unchanged_certificate(envelope):
    from experiments import transfer_batches

    study, scale, _ = envelope
    certificate = module.publish(study)
    assert (
        transfer_batches.limits_from_acceptance(study)
        == certificate["execution_contract"]["limits"]
    )
    write_json(study / "unit_extra_required.json", {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        transfer_batches.limits_from_acceptance(study)


@pytest.mark.parametrize("collection_ready,method_ready", [(False, True), (True, False)])
def test_study_gate_requires_both_collection_and_method_evidence(
    tmp_path, monkeypatch, collection_ready, method_ready
):
    monkeypatch.setattr(
        module.study,
        "readiness",
        lambda _: {
            "collection": {"ready": collection_ready},
            "method_evaluation": {"ready": method_ready},
        },
    )
    with pytest.raises(ValueError, match="Formal method evaluation prerequisites"):
        module.study.require_method_evaluation(tmp_path)


@pytest.mark.parametrize("command", ["run_k_shot"])
def test_root_commands_stop_before_settings_or_target_access(tmp_path, monkeypatch, command):
    from experiments import transfer_batches

    def reject(_):
        raise ValueError("Formal evidence missing")

    monkeypatch.setattr(module.study, "require_method_evaluation", reject)
    monkeypatch.setattr(
        transfer_batches, "load_settings", lambda _: pytest.fail("Read settings before admission")
    )
    monkeypatch.setattr(
        transfer_batches,
        "prepare_context",
        lambda *args: pytest.fail("Read targets before admission"),
    )
    with pytest.raises(ValueError, match="evidence missing"):
        getattr(module.study, command)(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "change", ["data", "settings", "omitted_extra", "omitted_mandatory", "loader", "entry"]
)
def test_tampering_or_omission_cannot_be_hidden_by_a_complete_flag(envelope, change):
    study, scale, _ = envelope
    certificate = module.publish(study)
    if change == "data":
        write_json(study / "unit_extra_required.json", {"changed": True})
    elif change == "settings":
        write_json(study / module.REGISTRY, {"changed": True})
    elif change in ("omitted_extra", "omitted_mandatory"):
        key = module._relative(
            study
            / (
                "unit_extra_required.json"
                if change == "omitted_extra"
                else "reports/statistics_protocol.json"
            )
        )
        certificate["proof_sha256"].pop(key)
        write_json(study / module.ACCEPTANCE, certificate)
    elif change == "loader":
        certificate["loader_sha256"] = "changed"
        write_json(study / module.ACCEPTANCE, certificate)
    else:
        (module.ROOT / "transfer_batches.py").write_text("# Changed unit entry\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed|omits|differs"):
        module.load_acceptance(study)
    assert not module.readiness(study)["ready"]


def test_publication_cannot_claim_success_when_a_stage_verifier_fails(envelope, monkeypatch):
    study, scale, _ = envelope

    def failed(*args, **kwargs):
        raise ValueError("unit incomplete native data evidence")

    monkeypatch.setattr(module, "collect_evidence", failed)
    with pytest.raises(ValueError, match="incomplete native data"):
        module.publish(study)
    assert not (study / module.ACCEPTANCE).exists()


@pytest.mark.parametrize("mode", ["budget", "controls", "accumulation", "all"])
def test_later_experiments_are_deferred_before_any_target_access(tmp_path, monkeypatch, mode):
    from experiments import transfer_batches

    monkeypatch.setattr(
        module.study,
        "require_method_evaluation",
        lambda root: pytest.fail("Deferred experiment read data"),
    )
    with pytest.raises(ValueError, match="deferred"):
        transfer_batches.run(mode, tmp_path)
    assert not list(tmp_path.iterdir())
