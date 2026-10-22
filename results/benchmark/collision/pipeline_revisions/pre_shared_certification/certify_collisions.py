"""Post-selection version/new-history labels use real replays and separate accounts."""

import json
from pathlib import Path

from controllers.highway.catalog import BY_ID, SUTS

from .executions import ExecutionBank, collect_bank
from .protocol import DATA_ROOT, DEFAULT, RESULT_ROOT
from .scenarios import DOMAINS, dedup_key
from .studies import SCOPES


def certificate_path(sut_id, condition, kind, environment_seed):
    return RESULT_ROOT / "certificates" / sut_id / condition / kind / f"env_{environment_seed}.json"


def certify_bank(sut_id, condition, kind, environment_seed):
    target = BY_ID[sut_id]
    if target.role != "target" or not target.reference_id:
        raise ValueError("Version-new labels require a registered reference/target lineage pair")
    target_bank = ExecutionBank(DATA_ROOT / "targets" / sut_id / condition / kind / f"env_{environment_seed}")
    collided = [s for s in target_bank.candidates if target_bank(s).feedback.collision]
    source_outcomes, calls, physical_s = {}, 0, 0.0
    for source in SUTS:
        if source.role != "history":
            continue
        if condition == "reuse":
            bank = ExecutionBank(DATA_ROOT / "physical/history" / kind / source.sut_id)
            coordinates = {s.coordinate(): s for s in bank.candidates}
            if any(s.coordinate() not in coordinates for s in collided):
                raise ValueError("Reused historical coordinates require real matching source executions")
            source_outcomes[source.sut_id] = {s.scenario_id: bank(coordinates[s.coordinate()]).feedback.collision
                                             for s in collided}
        elif collided:
            bank = collect_bank(source.sut_id, collided,
                                DATA_ROOT / "certification" / sut_id / condition / kind
                                / f"env_{environment_seed}" / source.sut_id,
                                environment_seed=environment_seed)
            source_outcomes[source.sut_id] = {s.scenario_id: bank(s).feedback.collision for s in collided}
            calls += bank.manifest["attempted_calls"]
            physical_s += bank.manifest["physical_execution_s"]
        else:
            source_outcomes[source.sut_id] = {}
    report = {"label_provenance": "real_execution", "target_id": sut_id,
              "reference_id": target.reference_id, "reference_lineage": BY_ID[target.reference_id].lineage,
              "change_type": "independent_training_run" if target.family == "learning" else "registered_configuration",
              "condition": condition, "scenario": kind, "environment_seed": environment_seed,
              "outcomes": {s.scenario_id: {source_id: values[s.scenario_id]
                                           for source_id, values in source_outcomes.items()} for s in collided},
              "actual_additional_source_calls": calls, "additional_source_execution_s": physical_s,
              "use": "independent_evaluation_only"}
    path = certificate_path(sut_id, condition, kind, environment_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def new_collision_metrics(candidates, queries, certificate, history_ids):
    if certificate["label_provenance"] != "real_execution":
        raise ValueError("Predicted historical passing cannot certify historical-set-specific collisions")
    if not history_ids:
        raise ValueError("The available historical set is required even for no-history baselines")
    version, historical, version_curve, historical_curve, found_ids = set(), set(), [], [], set()
    reference = certificate["reference_id"]
    for query in queries:
        if query["feedback"]["collision"]:
            scene = candidates[query["index"]]
            outcomes = certificate["outcomes"][scene.scenario_id]
            if any(type(outcomes[name]) is not bool for name in (*history_ids, reference)):
                raise TypeError("Certification must use actual Boolean collision labels")
            found_ids.add(scene.scenario_id)
            key = dedup_key(scene)
            if not outcomes[reference]:
                version.add(key)
            if all(not outcomes[name] for name in history_ids):
                historical.add(key)
        version_curve.append(len(version))
        historical_curve.append(len(historical))
    additional = (0 if certificate["condition"] == "reuse" else
                  len(found_ids) * len(set(history_ids) | {reference}))
    return {"version_new_collisions": len(version),
            "historical_set_specific_collisions": len(historical),
            "version_new_curve": version_curve, "historical_set_specific_curve": historical_curve,
            "certified_history_set": list(history_ids),
            "required_additional_source_calls_if_uncached": additional,
            "actual_shared_certification_calls": certificate["actual_additional_source_calls"]}


def main():
    frozen = DATA_ROOT / "frozen_protocol.json"
    if not frozen.is_file():
        raise ValueError("Certification follows frozen final comparisons")
    index = json.loads((RESULT_ROOT / "run_index.json").read_text(encoding="utf-8"))
    if index["status"] != "complete":
        raise ValueError("All fixed-budget trajectories must finish before certification")
    for sut in SUTS:
        if sut.role != "target":
            continue
        for condition in DEFAULT.conditions:
            for kind in DOMAINS:
                for environment_seed in DEFAULT.environment_seeds:
                    certify_bank(sut.sut_id, condition, kind, environment_seed)
    reports = []
    for row in index["runs"]:
        result = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
        certificate = json.loads(certificate_path(row["target_id"], row["condition"], row["scenario"],
                                                  row["environment_seed"]).read_text(encoding="utf-8"))
        bank = ExecutionBank(result["cost"]["physical_bank"])
        reports.append({"path": row["path"], **new_collision_metrics(
            bank.candidates, result["result"]["queries"], certificate, SCOPES[row["scope"]].history_ids)})
    (RESULT_ROOT / "new_collision_metrics.json").write_text(
        json.dumps({"status": "complete", "runs": reports}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
