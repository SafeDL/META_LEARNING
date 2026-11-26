"""Score the two E03 probability forecasts on matched, sealed target feedback."""

from pathlib import Path

import numpy as np

from controllers.highway.catalog import FORMAL_TARGET_SUTS

from experiments.weight_ablation import run as study
from experiments.collect_transfer_records import ApprovedCase
from experiments.k_shot import validate_result
from experiments.paired_statistics import json_ready, paired_summary
from experiments.budget.report import write_csv
from experiments.weight_ablation.report import FEW_SHOT, METHODS
from experiments.storage import read_json, write_json
from experiments.transfer_execution import _paths

OUTPUT = study.OUTPUT / "reports"
METRICS = ("heldout_nll", "heldout_brier", "sequential_nll", "sequential_brier")


def read_sealed(path, checksum):
    if study.sha(path) != checksum:
        raise ValueError(f"Sealed E03 input changed: {path}")
    return read_json(path)


def probability_scores(probability, labels):
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid collision probability")
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    nll = -np.where(labels, np.log(clipped), np.log1p(-clipped)).mean()
    return nll, np.square(probability - labels).mean()


def sealed_result(root, mode, scene, target, audit):
    path, _ = _paths(root, ApprovedCase(mode, target, scene, study.SEED, "full", None))
    return read_sealed(path, audit[mode]["files"][study.relative(path)])


def paired_scores(scene, target, checksum, *, frozen_root=None, prediction_root=None, budget_root=None):
    frozen_root = study.OUTPUT if frozen_root is None else frozen_root
    prediction_root = study.INPUT if prediction_root is None else prediction_root
    budget_root = study.protocol.OUTPUT if budget_root is None else budget_root
    audit = read_sealed(frozen_root / "audits" / scene / f"{target}.json", checksum)
    if audit["prediction"]["shared_feedback_verified"] != study.protocol.BUDGET:
        raise ValueError("Prediction ablation lacks the complete shared feedback sequence")
    control = sealed_result(budget_root, "budget", scene, target, audit)
    labels = np.asarray(control["evaluation"]["candidate_collisions"])
    if labels.shape != (audit["native_records"],) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Incomplete native collision labels")
    labels = labels.astype(bool)
    row = {"scene": scene, "target": target, "seed": study.SEED}
    common = None
    for method, root in zip(METHODS, (prediction_root, frozen_root), strict=True):
        saved = sealed_result(root, "prediction", scene, target, audit)
        spec, result = saved["specification"], saved["result"]
        split = spec["split"]
        validate_result(result, split)
        order = split["query_order"]
        queries = result["queries"]
        indices = np.asarray(split["evaluation_indices"])
        if (spec["candidates"] != control["specification"]["candidates"]
                or spec["bank_snapshot"] != audit["bank_snapshot"]
                or result["source_weight_updates"] != (method == "full")
                or result["target_feedback_updates"] is not True or result["feedback_used"] != "joint"
                or len(indices) != 1800 or len(order) != study.protocol.BUDGET
                or [q["index"] for q in queries] != order
                or [q["conditioning_queries"] for q in queries] != list(range(len(order)))
                or [q["feedback"]["collision"] for q in queries] != labels[order].tolist()):
            raise ValueError(f"Unmatched candidates, settings or prospective forecasts: {scene}/{target}")
        components = np.array([q["prequery_component_probabilities"] for q in queries])
        weights = np.array([q["prequery_component_weights"] for q in queries])
        probability = np.array([q["prequery_probability"] for q in queries])
        np.testing.assert_allclose((components * weights).sum(axis=1), probability, atol=1e-10, rtol=0)
        if common is None:
            common = split, components, [q["feedback"] for q in queries], result["component_names"]
        else:
            if (common[0] != split or common[2] != [q["feedback"] for q in queries]
                    or common[3] != result["component_names"]):
                raise ValueError("The two methods used different splits, feedback or branches")
            np.testing.assert_allclose(components, common[1], atol=1e-10, rtol=0)

        y = labels[indices]
        slices = []
        for snapshot, reference in zip(result["prediction_snapshots"], saved["metrics"], strict=True):
            if snapshot["paid_queries"] not in FEW_SHOT:
                continue
            scores = probability_scores(np.asarray(snapshot["probabilities"])[indices], y)
            np.testing.assert_allclose(scores, [reference["nll"], reference["brier"]], atol=1e-12, rtol=0)
            slices.append(scores)
        if len(slices) != len(FEW_SHOT):
            raise ValueError("Incomplete few-shot checkpoints")
        scores = (*np.mean(slices, axis=0), *probability_scores(probability, labels[order]))
        row["evaluation_collisions"] = int(y.sum())
        row.update({f"{method}_{key}": float(val)
                    for key, val in zip(METRICS, scores, strict=True)})
    return row


def report():
    source = OUTPUT / "completion.json"
    checksum = study.sha(source)
    receipt = read_json(source)
    if receipt["status"] != "complete" or receipt["paired_targets"] != len(study.SCENES) * len(study.TARGETS):
        raise ValueError("Missing E03 paired records")
    data = {}
    for scene in study.SCENES:
        for target in study.TARGETS:
            key = f"{scene}/{target}.json"
            data[scene, target] = paired_scores(scene, target, receipt["audit_files"][key])
    return summarize_scores(data, checksum, OUTPUT)


def score_scopes(data, metrics):
    scopes = {}
    for scope in ("overall", *study.SCENES):
        scenes = study.SCENES if scope == "overall" else [scope]
        values = np.array([[[[data[s, t][f"{m}_{k}"] for k in metrics]
                             for m in METHODS] for s in scenes] for t in study.TARGETS], dtype=float)
        paired = paired_summary(
            values[:, :, None], ads_ids=study.TARGETS, ads_families=[s.family for s in FORMAL_TARGET_SUTS],
            methods=METHODS, baseline="frozen", lower_better=False, replicates=10000, seed=1301,
        )["paired"]
        summary = dict(scenes=scenes, pairs=len(study.TARGETS) * len(scenes),
                       full=paired["method_mean"][0], frozen=paired["reference_mean"][0],
                       delta=paired["gain"][0], low=paired["gain_ci_low"][0], high=paired["gain_ci_high"][0])
        difference = values[:, :, 0] - values[:, :, 1]
        summary["lower_equal_higher"] = np.stack([(difference < 0).sum(axis=(0, 1)),
                                                  (difference == 0).sum(axis=(0, 1)),
                                                  (difference > 0).sum(axis=(0, 1))], axis=1)
        scopes[scope] = summary
    return json_ready(scopes)


def summarize_scores(data, checksum, directory):
    scopes = score_scopes(data, METRICS)
    output = json_ready({
        "status": "complete", "metrics": METRICS, "difference": "dynamic minus frozen",
        "registered_metrics_in_this_table": ["heldout_nll"],
        "supplementary_metrics_in_this_table": ["heldout_brier", "sequential_nll", "sequential_brier"],
        "few_shot_checkpoints": FEW_SHOT, "sequential_queries": [1, study.protocol.BUDGET],
        "bootstrap": {"unit": "ADS", "replicates": 10000, "seed": 1301, "interval": "pointwise percentile 95%"},
        "original_completion_sha256": checksum,
        "report_source_sha256": study.sha(Path(__file__)), "scopes": scopes,
    })
    table = directory / "probability_scores.csv"
    temporary = table.with_suffix(".tmp")
    write_csv(temporary, list(data.values()))
    temporary.replace(table)
    output["probability_scores_sha256"] = study.sha(table)
    write_json(directory / "probability_summary.json", output)
    return output


if __name__ == "__main__":
    print(report()["scopes"]["overall"])
