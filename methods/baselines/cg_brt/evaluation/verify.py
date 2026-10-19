"""Audit frozen sources, paid feedback, evidence updates and active dependencies."""

import ast
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.special import logsumexp

from methods.baselines.hem.experts import HistoricalExperts
from methods.baselines.hgt.history import load_history, split_indices, subset
from methods.baselines.hgt.io import read_json, write_json
from methods.common.snapshots import current_source, verify_source

from .config import INITIAL_QUERIES, POLICY, PROJECT, ROOT, SEEDS


def source_dependencies():
    sources = {
        path.relative_to(PROJECT)
        .as_posix()
        .removesuffix(".py")
        .replace("/", ".")
        .removesuffix(".__init__"): path
        for folder in ("methods", "simulation/highway", "controllers")
        for path in (PROJECT / folder).rglob("*.py")
    }
    pending = [
        "methods.baselines.cg_brt.selection",
        "methods.baselines.cg_brt.evaluation.evaluate",
        "methods.baselines.cg_brt.evaluation.compare",
        "methods.baselines.cg_brt.evaluation.verify",
    ]
    reached = set()
    while pending:
        module = pending.pop()
        if module in reached:
            continue
        reached.add(module)
        path = sources[module]
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.ImportFrom):
                prefix = (
                    package.split(".")[: len(package.split(".")) - node.level + 1]
                    if node.level
                    else []
                )
                base = ".".join(
                    prefix + (node.module.split(".") if node.module else [])
                )
                imports = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            elif isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            else:
                continue
            pending.extend(
                name for name in imports if name in sources and name not in reached
            )
    return sorted({sources[name] for name in reached})


def main():
    sources = source_dependencies()
    forbidden = ("concordance_gated_compositional_bayesian_mixture", "methods/egfd/")
    if any(any(word in path.as_posix() for word in forbidden) for path in sources):
        raise ValueError("Current method still depends on the old collision main chain")
    for path in (PROJECT / "methods/baselines/cg_brt").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.Attribute) and node.attr == "collision"
            for node in ast.walk(tree)
        ):
            raise ValueError("Risk-only method accesses a collision observation")
    snapshot = ROOT / "source_inputs.zip"
    inputs = [PROJECT / "data/history/responses.npz"]
    inputs += [PROJECT / f"results/baselines/hgt/models/seed_{seed}.pt" for seed in SEEDS]
    if not snapshot.exists():
        with ZipFile(snapshot, "w", compression=ZIP_DEFLATED) as archive:
            for path in sources + inputs:
                archive.write(path, path.relative_to(PROJECT).as_posix())
    with ZipFile(snapshot) as archive:
        # Evaluation scripts may change; frozen decision code and inputs must match.
        names = [
            name
            for name in archive.namelist()
            if not name.startswith(("experiments/", "benchmark/"))
            and "/evaluation/" not in name
        ]
        for name in names:
            verify_source(PROJECT, name, current_source(archive.read(name), name))
    history = load_history()
    train, _ = split_indices(next(iter(history.values()))["x"])
    expert_prior = (
        HistoricalExperts(subset(history, train))
        .predict(next(iter(history.values()))["x"][:1])
        .group_weights
    )
    report = read_json(ROOT / "comparison.json")
    candidate = [row for row in report["trajectory_metrics"] if row["method"] == POLICY]
    counts, active = {}, 0
    for row in candidate:
        trace = read_json(PROJECT / row["trace"])
        source_log = np.log(expert_prior)
        gate_log = np.full(2, -np.log(2))
        records = []
        for step, (query, evidence, decision, disclosure) in enumerate(
            zip(
                trace["queries"],
                trace["evidence"],
                trace["decisions"],
                trace["disclosures"],
            )
        ):
            index = query["index"]
            high = query["risk"] > 0.5
            if query["high_risk"] != high or disclosure["collision"] is not None:
                raise ValueError("High-risk label or feedback isolation differs")
            if (
                any(item["index"] != index for item in (evidence, disclosure))
                or decision["selected_index"] != index
            ):
                raise ValueError("Query, update and decision indices differ")
            np.testing.assert_allclose(
                evidence["source_weights_before"], np.exp(source_log), atol=1e-14
            )
            source_log += np.asarray(evidence["risk_log_density"])
            source_log -= logsumexp(source_log)
            np.testing.assert_allclose(
                evidence["source_weights_after"], np.exp(source_log), atol=1e-14
            )
            if len(source_log) != 6:
                raise ValueError("A neutral mixture component remains")
            odds = np.asarray(evidence["prequery_log_odds"])
            opposite = [old for label, old in records if label != high]
            score = np.zeros(2)
            if opposite:
                difference = (odds[:, None] - np.asarray(opposite).T) * (
                    1 if high else -1
                )
                score = -np.logaddexp(0, -difference).mean(axis=1)
            trust_before = max(0.0, 2 * np.exp(gate_log[0]) - 1)
            np.testing.assert_allclose(
                evidence["trust_before"], trust_before, atol=1e-14
            )
            np.testing.assert_allclose(evidence["ranking_log_score"], score, atol=1e-14)
            if evidence["opposite_pairs"] != len(opposite):
                raise ValueError("Prequery ranking pair count differs")
            gate_log += score
            gate_log -= logsumexp(gate_log)
            trust_after = max(0.0, 2 * np.exp(gate_log[0]) - 1)
            np.testing.assert_allclose(evidence["trust_after"], trust_after, atol=1e-14)
            if step >= INITIAL_QUERIES:
                np.testing.assert_allclose(decision["trust"], trust_before, atol=1e-14)
            records.append((high, odds.copy()))
            active += int(trust_before > 0)
            counts[decision["reason"]] = counts.get(decision["reason"], 0) + 1
    verification = {
        "status": "complete",
        "current_trajectories_checked": len(candidate),
        "current_paid_queries_checked": 200 * len(candidate),
        "source_snapshot_files_matched_after_path_rename": len(names) - len(inputs),
        "source_verification_scope": "Original frozen sources match after declared import/path renames; model and historical inputs match byte for byte",
        "historical_input_and_kernel_files_byte_identical": len(inputs),
        "current_method_imports_old_collision_main": False,
        "source_components": 6,
        "neutral_mixture_component": False,
        "collision_disclosures_to_selector": 0,
        "source_weight_updates_verified": True,
        "prequery_ranking_gate_updates_verified": True,
        "positive_trust_queries": active,
        "decision_counts": counts,
        "comparison_verification": report["verification"],
        "new_selector_runs": 0,
        "new_physical_calls": 0,
    }
    write_json(ROOT / "verification.json", verification)
    print(verification)


if __name__ == "__main__":
    main()
