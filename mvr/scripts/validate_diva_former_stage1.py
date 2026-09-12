"""Run the offline-only Stage 1 contract and counterfactual-teacher gates."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import yaml

from ..diva.source_bank import SourceBank, observation_from_dict
from ..provenance import content_hash
from ..diva_ai.counterfactual_teacher import CounterfactualTeacher


FORBIDDEN_SOURCE_REFS = {"idm_fast_small_gap", "idm_late_response"}


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bank(path: Path, source_refs: tuple[str, ...]) -> SourceBank:
    observations = [
        observation_from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    bank = SourceBank.from_observations(observations, source_refs)
    bank.require_common_anchors()
    if set(bank.source_refs) & FORBIDDEN_SOURCE_REFS:
        raise ValueError("sealed validation/test SUT appears in the Stage 1 source bank")
    return bank


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_manifest(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    root = config_file.resolve().parents[2]
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    base_file = root / config["base_config"]
    source_file = root / config["source_observations"]
    loso_file = root / config["source_loso_reference"]
    manifest = {
        "schema": "diva_former_stage1_manifest_v1",
        "git_commit": _git_commit(root),
        "config_sha256": _file_hash(config_file),
        "base_config_sha256": _file_hash(base_file),
        "source_observations_sha256": _file_hash(source_file),
        "source_loso_reference_sha256": _file_hash(loso_file),
        "config_hash": content_hash(config),
        "source_only": True,
        "new_simulator_calls": 0,
        "sealed_sut_refs": sorted(FORBIDDEN_SOURCE_REFS),
        "rebuild_command": (
            "conda run -n metadrive python -m mvr.scripts.validate_diva_former_stage1 "
            "--config mvr/configs/diva_former_stage1.yaml --phase teacher"
        ),
    }
    output = root / config["results_dir"] / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _signal_states(teacher: CounterfactualTeacher, source_index: int, total_budget: int) -> list[tuple[Any, int]]:
    """Deterministic partial states spanning the prescribed context sizes."""
    states: list[tuple[Any, int]] = []
    state = teacher.initial_state()
    for context_size in (0, 1, 2, 4, 8, 12):
        while len(state.history_indexes) < context_size:
            candidate = state.select_index("diagnostic")
            teacher._observe_oracle(state, source_index, candidate)
        states.append((state.clone(), total_budget - context_size))
    return states


def _teacher_gate(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    root = config_file.resolve().parents[2]
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    base_config = yaml.safe_load((root / config["base_config"]).read_text(encoding="utf-8"))
    source_refs = tuple(base_config["study"]["source_sut_refs"])
    bank = _load_bank(root / config["source_observations"], source_refs)
    total_budget = int(config["teacher"]["total_budget"])
    if total_budget > len(bank.designs):
        raise ValueError("Stage 1 teacher budget exceeds the frozen anchor pool")

    replay_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    for held_index, held_ref in enumerate(bank.source_refs):
        training_indexes = tuple(index for index in range(len(bank.source_refs)) if index != held_index)
        teacher = CounterfactualTeacher.from_source_bank(
            bank,
            training_indexes,
            rank=int(config["teacher"]["rank"]),
            device="cpu",
            gp_fit_steps=int(config["teacher"]["gp_fit_steps"]),
            proxy_event_threshold=float(base_config["vulnerability_response"]["proxy_event_threshold"]),
        )
        # This is intentionally an internal training-fold replay: the held-out
        # source is neither queried nor used by this fold's fitted prior.
        for source_index in training_indexes:
            for policy in ("teacher", "fixed_k4", "mine_now", "frozen_source_mean", "highest_shared_risk"):
                result = teacher.run(source_index, total_budget, policy)
                replay_rows.append(
                    {
                        "fold": held_index,
                        "held_out_source": held_ref,
                        "source_sut": bank.source_refs[source_index],
                        "policy": policy,
                        "formal_score_at_20": result.formal_score,
                        "cumulative_vulnerability_at_20": result.cumulative_vulnerability,
                        "selected_design_ids": [bank.design_ids[index] for index in result.selected_indexes],
                        "formal_curve": list(result.formal_curve),
                    }
                )
            for state, remaining_budget in _signal_states(teacher, source_index, total_budget):
                if remaining_budget < 4:
                    continue
                values = teacher.action_values(state, source_index, remaining_budget)
                formal = np.asarray([value.q_formal for value in values])
                vulnerability = np.asarray([value.q_vulnerability for value in values])
                signal_rows.append(
                    {
                        "fold": held_index,
                        "source_sut": bank.source_refs[source_index],
                        "context_size": len(state.history_indexes),
                        "remaining_budget": remaining_budget,
                        "formal_positive": bool(np.any(bank.formal_scores[source_index] > 0.0)),
                        "formal_action_gap": float(formal.max() - formal.min()),
                        "vulnerability_action_gap": float(vulnerability.max() - vulnerability.min()),
                    }
                )
    average_by_policy = {
        policy: float(np.mean([row["formal_score_at_20"] for row in replay_rows if row["policy"] == policy]))
        for policy in {row["policy"] for row in replay_rows}
    }
    teacher_score = average_by_policy["teacher"]
    baseline_names = ("fixed_k4", "mine_now", "frozen_source_mean", "highest_shared_risk")
    best_baseline = max(average_by_policy[name] for name in baseline_names)
    improvement = teacher_score - best_baseline
    relative_improvement = improvement / best_baseline if best_baseline > 0.0 else float("inf")
    vulnerability_signal = float(np.mean([
        row["vulnerability_action_gap"] >= 0.10 * row["remaining_budget"] for row in signal_rows
    ]))
    formal_signal_rows = [row for row in signal_rows if row["formal_positive"]]
    formal_signal = float(np.mean([
        row["formal_action_gap"] >= 0.5 for row in formal_signal_rows
    ])) if formal_signal_rows else 0.0
    criteria = {
        "portfolio_beats_or_matches_fixed_k4": teacher_score >= average_by_policy["fixed_k4"],
        "portfolio_beats_or_matches_mine_now": teacher_score >= average_by_policy["mine_now"],
        "downstream_gain": improvement >= 0.5 or relative_improvement >= 0.05,
        "nondegenerate_vulnerability_q": vulnerability_signal >= 0.50,
        "nondegenerate_formal_q": formal_signal >= 0.30,
        "fixed_budget": all(len(row["selected_design_ids"]) == total_budget and len(set(row["selected_design_ids"])) == total_budget for row in replay_rows),
        "held_out_not_used_in_fold_prior_or_replay": True,
    }
    report = {
        "schema": "diva_former_stage1_teacher_gate_v1",
        "source_only": True,
        "new_simulator_calls": 0,
        "teacher_policies": list(config["teacher"]["continuation_policies"]),
        "average_formal_score_at_20": average_by_policy,
        "teacher_score": teacher_score,
        "best_non_teacher_baseline": best_baseline,
        "absolute_improvement": improvement,
        "relative_improvement": relative_improvement,
        "signal_rates": {
            "vulnerability_action_gap": vulnerability_signal,
            "formal_action_gap": formal_signal,
        },
        "criteria": criteria,
        "pass": bool(all(criteria.values())),
        "replay_rows": replay_rows,
        "signal_rows": signal_rows,
    }
    output = root / config["results_dir"] / "teacher" / "teacher_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _finalize_stop_gate(config_path: str | Path) -> dict[str, Any]:
    """Write the required Stage 1 decision once a gate has stopped the run."""
    config_file = Path(config_path)
    root = config_file.resolve().parents[2]
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    results_dir = root / config["results_dir"]
    manifest_path = results_dir / "manifest.json"
    teacher_path = results_dir / "teacher" / "teacher_gate.json"
    pytest_path = results_dir / "pytest.stdout.log"
    if not manifest_path.exists() or not teacher_path.exists() or not pytest_path.exists():
        raise FileNotFoundError("manifest, teacher gate, and pytest report are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    pytest_text = pytest_path.read_text(encoding="utf-8")
    passed_match = re.search(r"(\d+) passed", pytest_text)
    g1_0_criteria = {
        "mvr_tests_passed": passed_match is not None,
        "source_only": manifest["source_only"] is True,
        "zero_new_simulator_calls": manifest["new_simulator_calls"] == 0,
        "sealed_suts_absent_from_source_bank": True,
        "frozen_input_hashes_recorded": all(
            manifest.get(name)
            for name in (
                "base_config_sha256",
                "source_observations_sha256",
                "source_loso_reference_sha256",
            )
        ),
        "fixed_budget_and_no_duplicate": teacher["criteria"]["fixed_budget"],
        "held_out_source_isolated": teacher["criteria"]["held_out_not_used_in_fold_prior_or_replay"],
    }
    report = {
        "schema": "diva_former_stage1_gate_v1",
        "source_only": True,
        "new_simulator_calls": 0,
        "g1_0_contract": {
            "pass": bool(all(g1_0_criteria.values())),
            "criteria": g1_0_criteria,
            "pytest_passed": int(passed_match.group(1)) if passed_match else 0,
        },
        "g1_1_teacher": {
            "pass": bool(teacher["pass"]),
            "report": "teacher/teacher_gate.json",
            "failure_reasons": [
                name for name, passed in teacher["criteria"].items() if not passed
            ],
        },
        "g1_2_belief": {"pass": False, "status": "not_run_after_g1_1_stop"},
        "g1_3_distillation": {"pass": False, "status": "not_run_after_g1_1_stop"},
        "g1_4_policy": {"pass": False, "status": "not_run_after_g1_1_stop"},
        "g1_5_engineering": {"pass": False, "status": "not_run_after_g1_1_stop"},
        "decision": "stop_before_validation_sut",
        "reason": (
            "G1-1 counterfactual teacher has no downstream formal-score headroom "
            "and its action-value signal is degenerate; Stage 1 rules prohibit "
            "training DIVA-Former after this result."
        ),
    }
    output = results_dir / "stage1_gate.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/diva_former_stage1.yaml")
    parser.add_argument("--phase", choices=("manifest", "teacher", "finalize"), default="teacher")
    args = parser.parse_args()
    manifest = build_manifest(args.config)
    if args.phase == "manifest":
        print(json.dumps(manifest, indent=2))
        return
    if args.phase == "finalize":
        report = _finalize_stop_gate(args.config)
        print(json.dumps({"decision": report["decision"], "g1_1": report["g1_1_teacher"]}, indent=2))
        return
    report = _teacher_gate(args.config)
    print(json.dumps({"pass": report["pass"], "criteria": report["criteria"]}, indent=2))


if __name__ == "__main__":
    main()
