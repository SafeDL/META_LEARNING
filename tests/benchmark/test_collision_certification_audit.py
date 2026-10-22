from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from benchmark.audit_collision_certification import CertificationEvidence, derive_metrics, validate_certificate
from benchmark import audit_collision_certification as auditor
from benchmark.protocol import DEFAULT
from benchmark.scenarios import sample
from controllers.highway.catalog import BY_ID


def test_independent_labels_include_reference_outside_allowed_history_and_quantized_duplicates():
    scenes = sample("front_brake", 3, "new", replicate=19)
    scenes[0] = replace(scenes[0], parameters={**scenes[0].parameters, "ego_speed_mps": 28.2})
    duplicate_bin = replace(scenes[0], scenario_id="same-quantized-bin",
                            parameters={**scenes[0].parameters, "ego_speed_mps": 28.3})
    candidates = list(map(asdict, [*scenes, duplicate_bin]))
    certificate = {"reference_id": "reference", "condition": "new", "actual_additional_source_calls": 37,
                   "outcomes": {scenes[0].scenario_id: {"reference": False, "allowed": True},
                                scenes[1].scenario_id: {"reference": True, "allowed": False},
                                scenes[2].scenario_id: {"reference": False, "allowed": False},
                                duplicate_bin.scenario_id: {"reference": False, "allowed": True}}}
    queries = [{"index": index, "feedback": {"collision": True}} for index in range(4)]
    result = derive_metrics(candidates, queries, certificate, ("allowed",))
    assert result["version_new_curve"] == [1, 1, 2, 2]
    assert result["historical_set_specific_curve"] == [0, 1, 2, 2]
    assert result["version_new_collisions"] == result["historical_set_specific_collisions"] == 2
    # Uncached certification must include the lineage reference even when its
    # family is held out of the method's historical set; shared calls are a reference.
    assert result["required_additional_source_calls_if_uncached"] == 8
    assert result["actual_shared_certification_calls"] == 37
    certificate["condition"] = "reuse"
    assert derive_metrics(candidates, queries, certificate, ("allowed",))["required_additional_source_calls_if_uncached"] == 0


@pytest.mark.parametrize("corruption", ["prediction", "wrong_reference", "numeric_label", "unpaid_calls"])
def test_certificate_rejects_false_provenance_wrong_reference_labels_or_cost(corruption):
    expected = {"label_provenance": "real_execution", "reference_id": "reference",
                "outcomes": {"scene": {"reference": False}}, "actual_additional_source_calls": 8}
    actual = {**deepcopy(expected), "cost_accounting": "Shared source-bank references are not additive."}
    validate_certificate(actual, expected)
    if corruption == "prediction":
        actual["label_provenance"] = "model_prediction"
    elif corruption == "wrong_reference":
        actual["reference_id"] = "another_lineage"
    elif corruption == "numeric_label":
        actual["outcomes"]["scene"]["reference"] = 0
    else:
        actual["actual_additional_source_calls"] = 0
    with pytest.raises(ValueError):
        validate_certificate(actual, expected)


def test_metric_reconstruction_rejects_query_that_disagrees_with_real_target_trace():
    scene = asdict(sample("front_brake", 1, "new")[0])
    certificate = {"reference_id": "a", "condition": "new", "actual_additional_source_calls": 0, "outcomes": {}}
    with pytest.raises(ValueError, match="actual target trace"):
        derive_metrics([scene], [{"index": 0, "feedback": {"collision": True}}], certificate, ("a",))
    with pytest.raises(ValueError, match="numeric"):
        derive_metrics([scene], [{"index": 0, "feedback": {"collision": 0}}], certificate, ("a",))


def test_certification_bank_rejects_other_source_before_reading_its_labels():
    class RawBank:
        def bank(self, path):
            return ({"sut_id": "wrong_source", "environment_seed": 81001},)

    evidence = CertificationEvidence({"sut_contracts": {}}, RawBank())
    with pytest.raises(ValueError, match="another physical source"):
        evidence.bank("not-read", "allowed_source", 81001)


@pytest.fixture
def shared_world(tmp_path, monkeypatch):
    sources = (BY_ID["history_idm_mobil"], BY_ID["history_fvdm_lane"])
    targets = (BY_ID["target_idm_mobil"], BY_ID["target_fvdm_lane"])
    monkeypatch.setattr(auditor, "SUTS", (*sources, *targets))
    monkeypatch.setattr(auditor, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(auditor, "DEFAULT", replace(DEFAULT, candidate_count=3, budget=3, initial_count=1,
                                                 horizon=2, checkpoints=(1, 2, 3)))
    candidates = [asdict(s) for s in sample("front_brake", 3, "new")]
    contracts = {sut.sut_id: {"fixture_contract": sut.sut_id} for sut in (*sources, *targets)}

    class Raw:
        def __init__(self):
            self.target_failures = {targets[0].sut_id: {0}, targets[1].sut_id: {2}}
            self.corrupt_source = False
            self.source_reads = 0

        def bank(self, root):
            root = auditor.Path(root)
            target = "targets" in root.parts
            name = root.parts[root.parts.index("targets") + 1] if target else root.name
            coordinates = deepcopy(candidates if target else [candidates[0], candidates[2]])
            if not target:
                self.source_reads += 1
                if self.corrupt_source:
                    coordinates[1] = candidates[1]
            manifest = {"sut_id": name, "environment_seed": 81001, "contract": contracts[name],
                        "candidate_count": len(coordinates), "scenarios": coordinates,
                        "attempted_calls": len(coordinates), "physical_execution_s": 5.0,
                        "failed_or_interrupted_calls": 0, "interrupted_call_duration_unknown": 0}
            root.mkdir(parents=True, exist_ok=True)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return (manifest,)

        def execution(self, root, index):
            root = auditor.Path(root)
            if "targets" in root.parts:
                name = root.parts[root.parts.index("targets") + 1]
                label = index in self.target_failures[name]
            else:
                label = (index == 1) if root.name == sources[0].sut_id else (index == 0)
            return {"facts": {"collision": label}}

    return {"sut_contracts": contracts}, Raw(), sources, targets


def test_shared_union_is_metered_once_and_wrong_union_is_rejected(shared_world):
    frozen, raw, sources, targets = shared_world
    evidence = CertificationEvidence(frozen, raw)
    first = evidence.certificate(targets[0].sut_id, "new", "front_brake", 81001)
    second = evidence.certificate(targets[1].sut_id, "new", "front_brake", 81001)
    assert first["actual_additional_source_calls"] == second["actual_additional_source_calls"] == 4
    assert sum(row["actual_physical_calls"] for row in evidence.paid_source_banks.values()) == 4
    assert raw.source_reads == 2
    assert first["source_bank_paths"] == second["source_bank_paths"]
    assert next(iter(first["outcomes"].values())) == {sources[0].sut_id: False, sources[1].sut_id: True}
    raw.corrupt_source = True
    with pytest.raises(ValueError, match="collision union"):
        CertificationEvidence(frozen, raw).certificate(targets[0].sut_id, "new", "front_brake", 81001)


def test_zero_target_collisions_need_no_source_bank(shared_world):
    frozen, raw, _, targets = shared_world
    raw.target_failures = {target.sut_id: set() for target in targets}
    evidence = CertificationEvidence(frozen, raw)
    report = evidence.certificate(targets[0].sut_id, "new", "front_brake", 81001)
    assert report["actual_additional_source_calls"] == 0 and not report["outcomes"] and not report["source_bank_paths"]
    assert not evidence.paid_source_banks and raw.source_reads == 0
