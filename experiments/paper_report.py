"""Audit supplemental paper evidence and export compact tables and academic figures."""

import copy
import csv
import math
import time
from collections import Counter
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.special import logsumexp

from simulator.highway.observation import VehicleState
from testing_methods.history_transfer.inference import (
    initial_weights,
    mixture_weights,
    predict_components,
    source_tensors,
)

from .attribution import prediction_metrics, read_source
from .audit_joint import verify_metrics
from .history import build_source, completed_archives, finalize_task, model_version
from .joint_figures import STYLE, STYLES, save_figure
from .paper_validation import (
    CONTROLS,
    DATA,
    EXPLORATION,
    OLD,
    OUTPUT,
    ROOT,
    SEED,
    load_models,
    relative,
    task_bank,
)
from .pilot import PRIMARY_BASELINES
from .query import QueryLedger
from .render_joint import render_execution
from .risk_figures import candidate_projection
from .rss import measurement_contract, trajectory_margin
from .scenarios import TEST_SCENARIOS, features
from .storage import read_json, write_json

TITLES = {
    "car_following": "Car following (3D)",
    "front_brake": "Lead braking (6D)",
    "cut_out": "Cut-out (6D)",
    "lane_change_rear": "Lane change / rear conflict (6D)",
}


def audit_supplement():
    """Reuse old audit evidence and inspect all 8,000 newly executed trajectories."""
    protocol = read_json(OUTPUT / "reports/protocol.json")
    original_audit = read_json(OLD / "reports/truth_collision_check.json")
    if original_audit["status"] != "complete":
        raise ValueError("Previously admitted physical evidence is not complete")
    rows = []
    for sut in protocol["systems"]:
        sut_id = sut["sut_id"]
        bank = task_bank(sut_id, "car_following")
        if len(bank.candidates) != 2000 or len({s.coordinate() for s in bank.candidates}) != 2000:
            raise ValueError("Expanded truth bank lost candidate identity or uniqueness")
        for kind in TEST_SCENARIOS:
            old_key = f"fixed/{sut_id}/{kind}"
            expected = 1000 if kind == "car_following" else 2000
            if original_audit["banks"][old_key]["records"] != expected:
                raise ValueError("Reused physical bank differs from its previous audit scope")
        extra_root = DATA / "targets/paper_supplement" / sut_id / "car_following"
        attempts = [read_json(p) for p in (extra_root / "attempts").glob("*.json")]
        calls = {a["call"]: a for a in attempts}
        frequencies = Counter(a["scenario_id"] for a in attempts)
        if any(n > 2 for n in frequencies.values()):
            raise ValueError("A supplemented physical scenario exceeded one retry")
        frames, collisions = 0, 0
        for index, scene in enumerate(bank.candidates[1000:], 1000):
            payload = read_json(bank.execution_path(index))
            trace, feedback = payload["trajectory"], payload["feedback"]
            if payload["scenario"] != asdict(scene) or feedback["scenario_id"] != scene.scenario_id:
                raise ValueError("Supplement response is attached to another coordinate")
            attempt = calls[payload["call"]]
            if attempt["status"] != "complete" or attempt["scenario_id"] != scene.scenario_id:
                raise ValueError("Supplement has no successful physical execution receipt")
            times = np.array([f["time_s"] for f in trace])
            if (
                len(trace) < 2
                or times[0] != 0
                or times[-1] > 10
                or not np.allclose(np.diff(times), 0.05, rtol=0, atol=1e-10)
            ):
                raise ValueError("Supplement changed the physical cadence or horizon")
            if bool(any(f["collision"] for f in trace)) != feedback["collision"]:
                raise ValueError("Collision label differs from actual simulation events")
            ego, lead = [VehicleState(**a) for a in trace[0]["actors"]]
            if ego.polygon().intersects(lead.polygon()):
                raise ValueError("A supplemented candidate starts with overlapping geometry")
            p = scene.parameters
            if not np.allclose(
                [ego.speed, lead.speed, lead.x - ego.x - 5],
                [p["ego_speed_mps"], p["lead_speed_mps"], p["gap_m"]],
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError("Initial physical states do not match candidate parameters")
            if any(
                f["background_controls"]["lead"]["acceleration_mps2"] != 0
                for f in trace[1:]
                if not f["collision"]
            ):
                raise ValueError("Constant-speed lead received an undeclared event")
            frames += len(trace)
            collisions += feedback["collision"]
        rows.append(
            {
                "sut_id": sut_id,
                "new_records": 1000,
                "trajectory_frames": frames,
                "actual_collisions": collisions,
                "attempted_calls": len(attempts),
                "initial_overlap": False,
                "physical_hz": 20,
                "label_matches_events": True,
            }
        )
    result = {
        "status": "passed",
        "new_records_checked": 8000,
        "reused_records": 56000,
        "old_audit": relative(OLD / "reports/truth_collision_check.json"),
        "banks": rows,
        "new_physical_executions_in_audit": 0,
    }
    write_json(OUTPUT / "reports/data_audit.json", result)
    return result


def verify_weights(result, method):
    scores = np.log(np.asarray(result["group_prior_weights"]))
    raw = method == "no_history"
    update = method not in ("no_history", "fixed_weights")
    maximum = 0.0
    for query in result["queries"]:
        p = np.asarray(query["prequery_component_probabilities"])
        before = np.ones(1) if raw else np.r_[0.9 * np.exp(scores - logsumexp(scores)), 0.1]
        maximum = max(
            maximum,
            float(np.max(np.abs(before - query["prequery_component_weights"]))),
            abs(float(before @ p) - query["prequery_probability"]),
        )
        if update:
            likelihood = np.log(
                np.maximum(p[:-1] if query["feedback"]["collision"] else 1 - p[:-1], 1e-15)
            )
            maximum = max(
                maximum, float(np.max(np.abs(likelihood - query["prequery_log_likelihood"])))
            )
            scores += likelihood
            scores -= logsumexp(scores)
        elif query["prequery_log_likelihood"] is not None:
            raise ValueError("A fixed-source or raw-target control updated source likelihoods")
        after = np.ones(1) if raw else np.r_[0.9 * np.exp(scores - logsumexp(scores)), 0.1]
        maximum = max(
            maximum, float(np.max(np.abs(after - query["component_weights_after_query"])))
        )
    if not math.isfinite(maximum) or maximum > 1e-10:
        raise ValueError("Weights do not follow past collision predictive likelihoods")
    return maximum


def archive_reused_tasks():
    """Finalize the saved E2 paid records, without replaying their selectors."""
    protocol = read_json(OUTPUT / "reports/protocol.json")
    summary = read_json(OUTPUT / "reports/summary.json")
    selection = read_json(OUTPUT / "reports/model_selection.json")
    systems = {s["sut_id"]: s for s in protocol["systems"]}
    unique = {r["path"]: r for r in summary["runs"] + selection["runs"]}
    index = []
    for path, row in unique.items():
        completed = (ROOT / path).with_name("completed.json.gz")
        e2 = (row["operation"], row["kernel"], row["method"]) == (
            "reuse",
            "shared_private",
            "joint",
        )
        if not completed.exists() and e2:
            completed = (
                OUTPUT / "completed_reused" / row["sut_id"] / row["scenario"] / "completed.json.gz"
            )
        if not completed.exists():
            if not e2:
                raise ValueError(f"A previously registered paid archive is missing: {completed}")
            sut_id, kind = row["sut_id"], row["scenario"]
            saved = read_json(ROOT / path)
            bank = task_bank(sut_id, kind)
            ledger = QueryLedger(
                bank.candidates, 200, bank, joint_feedback=True, allow_registration=False
            )
            cumulative = saved["metrics"]["cumulative_costs_s"]
            replay = np.diff([0, *cumulative["replay"]])
            rss = np.diff([0, *cumulative["rss"]])
            for q, wall, margin_time in zip(saved["result"]["queries"], replay, rss):
                ledger.records.append(
                    {
                        "call": q["query"],
                        "index": q["index"],
                        "scenario_id": q["feedback"]["scenario_id"],
                        "purpose": "initial" if q["query"] <= 20 else "target",
                        "account": "target",
                        "status": "complete",
                        "wall_s": float(wall),
                        "rss_compute_s": float(margin_time),
                        "feedback": q["feedback"],
                    }
                )
                ledger.trajectories.append(bank(bank.candidates[q["index"]]).trajectory)
            sut = systems[sut_id]
            completed = OUTPUT / "completed_reused" / sut_id / kind / "completed.json.gz"
            finalize_task(
                completed,
                ledger,
                execution_bank=bank,
                task_id=f"paper:shared_private:joint:{sut_id}:{kind}:{SEED}",
                sut_id=sut_id,
                sut_version=sut["version"],
                family=sut["family"],
                lineage=sut["lineage"],
                original_role=sut["role"],
                environment_seed=protocol["config"]["environment_seed"],
                completed_order=1 + TEST_SCENARIOS.index(kind) * 8 + list(systems).index(sut_id),
                branch_id=f"paper:shared_private:joint:{SEED}",
                execution_contract=bank.manifest["contract"],
                model_snapshot={
                    "reused_result": path,
                    "original_specification": saved["specification"],
                    "timing_origin": "differences of saved cumulative paid costs",
                },
            )
        if row["operation"] == "new" or e2:
            payload = read_json(completed)
            if (
                payload["format"] != "completed-paid-task"
                or not payload["completed"]
                or len(payload["executions"]) != 200
                or [e["scenario"]["scenario_id"] for e in payload["executions"]]
                != [
                    q["feedback"]["scenario_id"]
                    for q in read_json(ROOT / path)["result"]["queries"]
                ]
            ):
                raise ValueError("Task archive does not preserve its 200 paid queries in order")
        index.append(
            {
                "result": path,
                "completed": relative(completed),
                "sut_id": row["sut_id"],
                "scenario": row["scenario"],
                "method": row["method"],
                "kernel": row["kernel"],
            }
        )
    write_json(
        OUTPUT / "reports/task_archives.json",
        {"status": "passed", "tasks": index, "new_method_queries": 0, "new_physical_executions": 0},
    )
    selected = summary["selected_kernel"]
    paths = [
        ROOT / r["completed"] for r in index if r["method"] == "joint" and r["kernel"] == selected
    ]
    archives = []
    for path in paths:
        task = read_json(path)
        if completed_archives(
            [path], before_order=task["completed_order"], branch_id=task["branch_id"]
        ):
            raise ValueError("The current task order was admitted as a historical source")
        if completed_archives(
            [path],
            before_order=task["completed_order"] + 1,
            branch_id=task["branch_id"],
            current_system=(task["sut_id"], task["sut_version"]),
        ):
            raise ValueError("A target admitted its own response archive as a source")
        archives.extend(
            completed_archives(
                [path], before_order=task["completed_order"] + 1, branch_id=task["branch_id"]
            )
        )
    if len(archives) != 32 or sum(len(a.scenarios) for a in archives) != 6400:
        raise ValueError("Selected method must admit only its 6,400 paid responses")
    first_sut = next(iter(systems))
    example = next(
        a
        for a in archives
        if a.scenarios[0].kind == "car_following" and a.provenance["sut_id"] == first_sut
    )
    _, encoder = load_models("car_following", selected)
    version = model_version(encoder)
    source_path = OUTPUT / "sources/completed/car_following.npz"
    source = build_source(
        encoder, [example], task_bank(first_sut, "car_following").candidates, cache_path=source_path
    )
    if (
        model_version(encoder) != version
        or len(source["specification"]["view_record_ids"][example.source_id]) != 200
    ):
        raise ValueError("Post-task source changed the shared model or included unpaid records")
    result = {
        "status": "passed",
        "selected_kernel": selected,
        "selected_tasks": 32,
        "paid_response_records": 6400,
        "new_queries": 0,
        "new_training_steps": 0,
        "source_example": relative(source_path),
        "source_records": 200,
        "shared_update_performed": False,
        "self_source_excluded": True,
        "current_task_order_excluded": True,
        "scope": "Archive admission and one frozen-model source construction; no accumulation-effect experiment and no changes to running task snapshots.",
    }
    write_json(OUTPUT / "reports/knowledge_archive_check.json", result)
    return result


def audit_results():
    started = time.perf_counter()
    protocol = read_json(OUTPUT / "reports/protocol.json")
    summary = read_json(OUTPUT / "reports/summary.json")
    selection = read_json(OUTPUT / "reports/model_selection.json")
    kernel_tasks = {
        (s["sut_id"], kind, kernel)
        for s in protocol["systems"]
        for kind in TEST_SCENARIOS
        for kernel in ("joint", "shared_private")
    }
    if (
        len(selection["runs"]) != 64
        or {(r["sut_id"], r["scenario"], r["kernel"]) for r in selection["runs"]} != kernel_tasks
    ):
        raise ValueError("The kernel comparison is not the complete registered 64-task matrix")
    if (
        summary["status"] != "complete"
        or read_json(OUTPUT / "reports/data_audit.json")["status"] != "passed"
    ):
        raise ValueError("The complete matrix and physical audit must precede result acceptance")
    expected = {
        (s["sut_id"], kind, method)
        for s in protocol["systems"]
        for kind in TEST_SCENARIOS
        for method in ("joint", *PRIMARY_BASELINES, *CONTROLS)
    }
    actual = [(r["sut_id"], r["scenario"], r["method"]) for r in summary["runs"]]
    if len(actual) != 320 or set(actual) != expected:
        raise ValueError("The paper matrix omits or duplicates registered tasks")
    original = read_json(OUTPUT / "reports/reuse.json")
    lookup = {(r["sut_id"], r["scenario"], r["method"], r["kernel"]): r for r in original}
    checked, truth = [], {}
    unique_runs = {r["path"]: r for r in summary["runs"] + selection["runs"]}
    if len(unique_runs) != 352:
        raise ValueError("Main and alternative-kernel evidence must contain 352 unique units")
    for path, row in unique_runs.items():
        saved = read_json(ROOT / path)
        result, metrics = saved["result"], saved["metrics"]
        sut_id, kind, method = (row[k] for k in ("sut_id", "scenario", "method"))
        bank = task_bank(sut_id, kind)
        key = sut_id, kind
        if key not in truth:
            previous = read_json(ROOT / lookup[sut_id, kind, "joint", "joint"]["path"])
            labels = previous["evaluation"]["candidate_collisions"]
            if kind == "car_following":
                labels = labels + [bank(s).feedback.collision for s in bank.candidates[1000:]]
            truth[key] = labels
        labels = truth[key]
        indices, queries = result["selected_indices"], result["queries"]
        if (
            len(indices) != 200
            or len(set(indices)) != 200
            or indices[:20] != protocol["initial_indices"][kind]
        ):
            raise ValueError("A method changed the common initialization or query budget")
        if row["operation"] == "new":
            specification = saved["specification"]
            if specification["candidates"] != [asdict(s) for s in bank.candidates]:
                raise ValueError("A new selector saw a different registered coordinate library")
            if method not in PRIMARY_BASELINES:
                if result["response_kernel"] != (
                    "JointResponseKernel" if row["kernel"] == "joint" else "SharedPrivateKernel"
                ):
                    raise ValueError("A control uses another response kernel")
                if method == "no_history":
                    if (
                        specification["target_model_version"] is not None
                        or specification["source_snapshot"] is not None
                        or result["representation"] != "normalized_parameters"
                    ):
                        raise ValueError(
                            "No-history control contains a historical model or representation"
                        )
                else:
                    expected_versions = protocol["models"][kind][row["kernel"]]
                    source = specification["source_snapshot"]
                    if (
                        specification["target_model_version"] != expected_versions[0]
                        or source["model_version"] != expected_versions[1]
                        or source["view_record_ids"] != protocol["source_view_record_ids"][kind]
                    ):
                        raise ValueError(
                            "A target task switched its model or historical view snapshot"
                        )
            if saved["evaluation"]["candidate_collisions"] != labels:
                raise ValueError("A new run used another complete truth denominator")
            verify_metrics(
                bank.candidates, labels, indices, queries, saved["query_ledger"], metrics, 20
            )
            completed = read_json((ROOT / path).with_name("completed.json.gz"))
            if len(completed["executions"]) != 200 or len(completed["attempts"]) != 200:
                raise ValueError("Completed archive includes missing or unpaid executions")
            for q, record in zip(queries, completed["executions"]):
                physical = bank(bank.candidates[q["index"]])
                if (
                    q["feedback"]["collision"] != physical.feedback.collision
                    or record["execution_ref"] != relative(bank.execution_path(q["index"]))
                    or record["scenario"] != asdict(bank.candidates[q["index"]])
                    or record["feedback"]["rss_contract"] != measurement_contract()
                ):
                    raise ValueError(
                        "Paid feedback/archive is not backed by its physical execution"
                    )
                if not np.isclose(
                    q["feedback"]["safety"]["rss_margin"],
                    trajectory_margin(physical.trajectory)["minimum_margin"],
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError("Paid RSS differs from its actual executed trajectory")
        else:
            counts = np.cumsum([labels[i] for i in indices])
            if (
                counts.tolist() != metrics["discovery_curve"]
                or counts[-1] != metrics["unique_collisions"]
                or float(counts.mean()) != metrics["discovery_auc"]
                or sum(labels) != metrics["pool_unique_collisions"]
            ):
                raise ValueError("Reused results no longer match candidate truth and discovery IDs")
        weight_error = None if method in PRIMARY_BASELINES else verify_weights(result, method)
        checked.append(
            {
                "sut_id": sut_id,
                "scenario": kind,
                "method": method,
                "kernel": row["kernel"],
                "path": path,
                "operation": row["operation"],
                "queries": 200,
                "weight_max_absolute_error": weight_error,
            }
        )
    report = {
        "status": "passed",
        "main_units": 320,
        "unique_units": len(checked),
        "unique_logical_queries": len(checked) * 200,
        "new_units": sum(r["operation"] == "new" for r in checked),
        "reused_units": sum(r["operation"] == "reuse" for r in checked),
        "runs": checked,
        "audit_wall_s": time.perf_counter() - started,
    }
    write_json(OUTPUT / "reports/results_audit.json", report)
    return report


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_defined(values):
    values = [v for v in values if v is not None]
    return None if not values else float(np.mean(values))


def aggregate(rows, methods):
    table = []
    for method in methods:
        measured = [r["metrics"] for r in rows if r["method"] == method]
        failures = sum(m["unique_collisions"] for m in measured)
        denominator = sum(m["pool_unique_collisions"] for m in measured)
        table.append(
            {
                "method": method,
                "tasks": len(measured),
                "failures": failures,
                "post_initial_failures": sum(m["post_initial_collisions"] for m in measured),
                "mean_auc": float(np.mean([m["discovery_auc"] for m in measured])),
                "pooled_coverage": failures / denominator if denominator else None,
                "macro_coverage": mean_defined([m["collision_recall"] for m in measured]),
                "coverage_defined_tasks": sum(m["collision_recall"] is not None for m in measured),
                "pair_distance": mean_defined([m["pair_distance"] for m in measured]),
                "nearest_neighbor_distance": mean_defined(
                    [m["nearest_neighbor_distance"] for m in measured]
                ),
                **{
                    f"found_{b}": sum(m["discovery_curve"][b - 1] for m in measured)
                    for b in (20, 50, 100, 200)
                },
            }
        )
    return table


def paired_distances(rows, methods):
    lookup = {(r["sut_id"], r["scenario"], r["method"]): r["metrics"] for r in rows}
    pairs = []
    for (sut, kind, method), a in lookup.items():
        if method != "joint":
            continue
        for comparator in methods:
            if comparator == "joint":
                continue
            b = lookup[sut, kind, comparator]
            k = min(a["unique_collisions"], b["unique_collisions"])
            pairs.append(
                {
                    "sut_id": sut,
                    "scenario": kind,
                    "comparator": comparator,
                    "delta_failures": a["unique_collisions"] - b["unique_collisions"],
                    "delta_auc": a["discovery_auc"] - b["discovery_auc"],
                    "matched_count": k,
                    **{
                        f"{prefix}_{metric}": values["distances_by_discovery_count"][metric][k - 1]
                        if k >= 2
                        else None
                        for prefix, values in (("joint", a), ("control", b))
                        for metric in ("pair_distance", "nearest_neighbor_distance")
                    },
                }
            )
    return pairs


@plt.rc_context(STYLE)
def plot_curves(rows, methods, path, *, coverage=False, labels=None):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for axis, kind in zip(axes.flat, TEST_SCENARIOS):
        for method in methods:
            measured = [
                r["metrics"] for r in rows if r["method"] == method and r["scenario"] == kind
            ]
            if coverage:
                measured = [m for m in measured if m["pool_unique_collisions"]]
            if not measured:
                continue
            metric = "recall_curve" if coverage else "discovery_curve"
            values = np.mean([m[metric] for m in measured], axis=0)
            title, color, style = STYLES[method]
            if labels is not None:
                title = labels.get(method, title)
            axis.step(
                range(1, 201),
                values,
                where="post",
                color=color,
                linestyle=style,
                label=title,
                linewidth=1.8 if method == "joint" else 1.1,
            )
        axis.axvline(20, color="#999999", linestyle=":", linewidth=0.7)
        title = TITLES[kind]
        if coverage:
            valid = sum(
                r["metrics"]["pool_unique_collisions"] > 0
                for r in rows
                if r["scenario"] == kind and r["method"] == "joint"
            )
            title += f" · n={valid}/8"
        axis.set_title(title, fontsize=10)
        axis.set_xlim(1, 200)
        axis.set_xlabel("Paid queries")
        axis.set_ylabel("Mean candidate recall" if coverage else "Mean failures discovered")
        if coverage:
            axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(3, len(methods)),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, path)


@plt.rc_context(STYLE)
def plot_diversity(pairs, path):
    methods = ("av_bo_rf_ei", "scenario_fuzz", "no_history")
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.8))
    for j, kind in enumerate(TEST_SCENARIOS):
        for i, metric in enumerate(("pair_distance", "nearest_neighbor_distance")):
            axis = axes[i, j]
            for x, method in enumerate(methods):
                values = [
                    p[f"joint_{metric}"] - p[f"control_{metric}"]
                    for p in pairs
                    if p["scenario"] == kind
                    and p["comparator"] == method
                    and p["matched_count"] >= 2
                ]
                if values:
                    axis.scatter(
                        np.full(len(values), x), values, color=STYLES[method][1], s=12, alpha=0.45
                    )
                    axis.scatter([x], [np.mean(values)], marker="D", color="black", s=23)
            axis.axhline(0, color="#777777", linewidth=0.7)
            axis.set_xticks(range(3), ["RF-EI", "Fuzz", "Raw GP"], rotation=30)
            title = (
                TITLES[kind] if kind != "lane_change_rear" else "Lane change / rear\nconflict (6D)"
            )
            axis.set_title(title, fontsize=9)
            axis.set_ylabel("Pairwise distance change" if i == 0 else "Nearest-neighbor change")
            axis.grid(axis="y", alpha=0.18)
    fig.suptitle("Full method minus comparator at matched discovery count", fontsize=11)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                color="#888888",
                linestyle="none",
                label="ADS task",
                markersize=3,
            ),
            Line2D(
                [], [], marker="D", color="black", linestyle="none", label="Task mean", markersize=4
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, path)


@plt.rc_context(STYLE)
def plot_space(rows, protocol, path):
    sut = protocol["systems"][0]["sut_id"]
    methods = ("joint", "av_bo_rf_ei", "scenario_fuzz")
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.2))
    for i, kind in enumerate(TEST_SCENARIOS):
        bank = task_bank(sut, kind)
        projection = candidate_projection(bank.candidates)
        points = np.asarray(projection["coordinates"])
        for j, method in enumerate(methods):
            row = next(
                r for r in rows if (r["sut_id"], r["scenario"], r["method"]) == (sut, kind, method)
            )
            selected = {
                s.scenario_id
                for s in bank.candidates
                if s.scenario_id in row["metrics"]["discovered_ids"]
            }
            found = np.array([s.scenario_id in selected for s in bank.candidates])
            axis = axes[i, j]
            axis.scatter(*points.T, s=1.5, color="#B8C0C8", alpha=0.6)
            axis.scatter(
                *points[found].T, s=12, color="#D55E00", edgecolors="white", linewidths=0.3
            )
            axis.set_title(f"{STYLES[method][0]}: {int(found.sum())}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel("Parameter PC1", fontsize=8)
            axis.set_ylabel(
                TITLES[kind] + "\nParameter PC2" if j == 0 else "Parameter PC2", fontsize=8
            )
    fig.suptitle("Discovered failures in public parameter coordinates · IDM/MOBIL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path)


@plt.rc_context(STYLE)
def plot_adaptation(rows, protocol, path):
    """Use the fixed first ADS and cut-out scene; projection sees coordinates only."""
    sut, kind = protocol["systems"][0]["sut_id"], "cut_out"
    row = next(r for r in rows if (r["sut_id"], r["scenario"], r["method"]) == (sut, kind, "joint"))
    saved = read_json(ROOT / row["path"])
    bank = task_bank(sut, kind)
    points = np.asarray(candidate_projection(bank.candidates)["coordinates"])
    snapshots = {s["paid_queries"]: s for s in saved["result"]["prediction_snapshots"]}
    if 0 not in snapshots:
        # The saved E2 run begins at 20. Reconstruct only the historical prior,
        # with no target feedback or selector execution.
        kernel = row["kernel"]
        base, _ = load_models(kind, kernel)
        source_path = (
            ROOT / "artifacts/runs/joint_response/attribution/sources" / kind / "joint.npz"
            if kernel == "joint"
            else EXPLORATION / "sources" / f"{kind}.npz"
        )
        source = read_source(source_path)
        if source["specification"]["view_record_ids"] != protocol["source_view_record_ids"][kind]:
            raise ValueError("The displayed historical prior uses a different source view")
        physical = type(base)(base.dimension, deep=False).double()
        with torch.no_grad():
            reference = base.point_mean(1)[0].detach()
            components, _ = predict_components(
                base,
                physical,
                features(bank.candidates),
                source_tensors(source, base),
                reference,
                [],
                [],
                [],
                batch_size=128,
            )
            weights = mixture_weights(initial_weights(source, reference).log())
            probability = (weights[:, None] * components).sum(0).numpy().tolist()
        snapshots[0] = {"paid_queries": 0, "observed_indices": [], "probabilities": probability}
        write_json(
            OUTPUT / "reports/zero_feedback_prediction.json",
            {
                "sut_id": sut,
                "scenario": kind,
                "kernel": kernel,
                "input": "frozen historical models, source priors and public coordinates only",
                "new_queries": 0,
                **snapshots[0],
            },
        )
    fig, axes = plt.subplots(1, 4, figsize=(8.5, 2.8), sharex=True, sharey=True)
    for axis, prefix in zip(axes, (0, 20, 50, 200)):
        snapshot = snapshots[prefix]
        plotted = axis.scatter(
            *points.T, c=snapshot["probabilities"], cmap="viridis", vmin=0, vmax=1, s=3
        )
        observed = snapshot["observed_indices"]
        failed = [
            i
            for i in observed
            if next(q for q in saved["result"]["queries"] if q["index"] == i)["feedback"][
                "collision"
            ]
        ]
        if failed:
            axis.scatter(
                *points[failed].T, s=16, facecolors="none", edgecolors="#D55E00", linewidths=0.7
            )
        axis.set_title(f"{prefix} paid queries", fontsize=9)
        axis.set_xlabel("Parameter PC1")
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0].set_ylabel("Parameter PC2")
    fig.subplots_adjust(bottom=0.18, top=0.78, right=0.90, wspace=0.15)
    color_axis = fig.add_axes((0.92, 0.20, 0.015, 0.54))
    fig.colorbar(plotted, cax=color_axis, label="Predicted collision probability")
    fig.suptitle("Historical prior and target adaptation · cut-out · IDM/MOBIL", fontsize=10)
    save_figure(fig, path)


def costs(rows, selection):
    unique = {r["path"]: r for r in rows + selection["runs"]}
    measured = []
    for path, row in unique.items():
        saved = read_json(ROOT / path)
        resource = saved["result"]["resources"]
        metrics = saved["metrics"]
        measured.append(
            {
                "path": path,
                "operation": row["operation"],
                "sut_id": row["sut_id"],
                "scenario": row["scenario"],
                "method": row["method"],
                "kernel": row["kernel"],
                "method_wall_s": resource["wall_s"],
                "sampled_peak_cpu_bytes": resource["sampled_peak_cpu_bytes"],
                "adaptation_s": metrics["adaptation_s"],
                "selection_s": metrics["selection_s"],
                "rss_compute_s": metrics["rss_compute_s"],
                "paid_replay_wall_s": metrics["replay_wall_s"],
            }
        )
    write_csv(OUTPUT / "reports/costs.csv", measured)
    collection = read_json(OUTPUT / "reports/collection.json")
    construction = []
    for kind in TEST_SCENARIOS:
        directory = (
            EXPLORATION / "models" / kind
            if selection["selected_kernel"] == "shared_private"
            else OLD / "models" / kind / "all_sources"
        )
        resource = read_json(directory / "resources.json")
        construction.append(
            {
                "scenario": kind,
                "record": relative(directory / "resources.json"),
                "wall_s": resource["wall_s"],
                "sampled_peak_cpu_bytes": resource["sampled_peak_cpu_bytes"],
            }
        )
    return {
        "new_physical_calls": sum(r["attempted_calls"] for r in collection["banks"]),
        "new_physical_successes": sum(r["new_records"] for r in collection["banks"]),
        "physical_execution_s": sum(r["physical_execution_s"] for r in collection["banks"]),
        "sum_physical_worker_wall_s": sum(r["wall_s"] for r in collection["banks"]),
        "new_method_wall_s": sum(r["method_wall_s"] for r in measured if r["operation"] == "new"),
        "largest_sampled_method_cpu_bytes": max(r["sampled_peak_cpu_bytes"] for r in measured),
        "new_training_steps": 0,
        "reused_model_construction": construction,
        "reused_model_construction_wall_s": sum(r["wall_s"] for r in construction),
        "timing_scope": "Summed per-run measurements, not parallel end-to-end elapsed time. RSS is not added again to replay wall time.",
        "follow_source_export": read_json(OUTPUT / "reports/source_export_timing.json"),
        "source_timing_note": "Initial expanded-cache export was not separately timed; a checked repeat is reported separately. Existing model construction costs remain in their original resource reports.",
    }


def follow_segments(rows):
    table = []
    for row in rows:
        if row["scenario"] != "car_following":
            continue
        saved = read_json(ROOT / row["path"])
        labels = saved["evaluation"]["candidate_collisions"]
        selected = saved["result"]["selected_indices"]
        table.append(
            {
                "sut_id": row["sut_id"],
                "method": row["method"],
                "grid_truth_failures": sum(labels[:1000]),
                "sobol_truth_failures": sum(labels[1000:]),
                "grid_discovered": sum(labels[i] for i in selected if i < 1000),
                "sobol_discovered": sum(labels[i] for i in selected if i >= 1000),
            }
        )
    write_csv(OUTPUT / "reports/follow_segments.csv", table)
    return table


def prediction_summary(rows):
    truth = {}
    for row in rows:
        if row["method"] == "no_history":
            truth[row["sut_id"], row["scenario"]] = read_json(ROOT / row["path"])["evaluation"][
                "candidate_collisions"
            ]
    values = []
    for row in rows:
        if row["method"] not in ("joint", *CONTROLS):
            continue
        saved = read_json(ROOT / row["path"])
        for snapshot in saved["result"]["prediction_snapshots"]:
            if snapshot["paid_queries"] not in (20, 50, 200):
                continue
            values.append(
                {
                    "sut_id": row["sut_id"],
                    "scenario": row["scenario"],
                    "method": row["method"],
                    "paid_queries": snapshot["paid_queries"],
                    **prediction_metrics(snapshot, truth[row["sut_id"], row["scenario"]]),
                }
            )
    write_csv(OUTPUT / "reports/prediction_metrics.csv", values)
    return {
        "scope": "Unqueried candidates after each method's own paid prefix; paths differ. Ranking hits are diagnostic, not additional discoveries.",
        "means": [
            {
                "method": method,
                "paid_queries": prefix,
                **{
                    metric: mean_defined(
                        [
                            r[metric]
                            for r in values
                            if r["method"] == method and r["paid_queries"] == prefix
                        ]
                    )
                    for metric in ("nll", "brier", "top_200_failures")
                },
            }
            for method in ("joint", *CONTROLS)
            for prefix in (20, 50, 200)
        ],
    }


@torch.no_grad()
def prefix_components(base, source, x, paid, state):
    """Restore the six saved online scalars while preserving the pretrained reference."""
    target = copy.deepcopy(base)
    physical = type(base)(base.dimension, deep=False).double()
    for name, value in state["target_parameters"].items():
        getattr(target, name).fill_(value)
    for name, value in state["physical_parameters"].items():
        getattr(physical, name).fill_(value)
    components, _ = predict_components(
        target,
        physical,
        x,
        source_tensors(source, base),
        base.point_mean(1)[0].detach(),
        state["observed_indices"],
        [q["feedback"]["collision"] for q in paid],
        [q["feedback"]["safety"]["rss_margin"] for q in paid],
        batch_size=128,
    )
    return components.numpy()


def diagnose_negative_transfers(rows, protocol):
    """Decompose existing common-feedback states without new feedback or optimization."""
    lookup = {(r["sut_id"], r["scenario"], r["method"]): r for r in rows}
    cases, output = [], []
    for (sut, kind, method), row in lookup.items():
        if method != "joint":
            continue
        delta = (
            row["metrics"]["unique_collisions"]
            - lookup[sut, kind, "no_history"]["metrics"]["unique_collisions"]
        )
        if delta >= 0:
            continue
        kernel = row["kernel"]
        common_path = (
            EXPLORATION / "common" / sut / f"{kind}.json.gz"
            if kernel == "shared_private"
            else ROOT
            / "artifacts/runs/joint_response/attribution/common_feedback"
            / sut
            / kind
            / "joint.json.gz"
        )
        common = read_json(common_path)
        if common["specification"]["target_model_version"] != protocol["models"][kind][kernel][0]:
            raise ValueError("The common-feedback diagnostic uses another target model")
        if kind == "car_following":
            paths = list((OUTPUT / "sources" / kernel / kind).glob("*.npz"))
            if len(paths) != 1:
                raise ValueError("The expanded source cache must identify one frozen snapshot")
            source_path = paths[0]
        else:
            source_path = (
                EXPLORATION / "sources" / f"{kind}.npz"
                if kernel == "shared_private"
                else ROOT / "artifacts/runs/joint_response/attribution/sources" / kind / "joint.npz"
            )
        source = read_source(source_path)
        if (
            source["specification"]["view_record_ids"] != protocol["source_view_record_ids"][kind]
            or source["specification"]["model_version"] != protocol["models"][kind][kernel][1]
        ):
            raise ValueError("The common-feedback diagnostic changed its historical view")
        base, _ = load_models(kind, kernel)
        bank = task_bank(sut, kind)
        x = features(bank.candidates)
        target_truth = read_json(ROOT / lookup[sut, kind, "no_history"]["path"])["evaluation"][
            "candidate_collisions"
        ]
        snapshots = {s["paid_queries"]: s for s in common["result"]["prediction_snapshots"]}
        maximum = 0.0
        for prefix in (20, 50, 100):
            state = snapshots[prefix]
            paid = common["result"]["queries"][:prefix]
            indices = [q["index"] for q in paid]
            if indices != state["observed_indices"] or any(
                q["feedback"]["scenario_id"] != bank.candidates[q["index"]].scenario_id
                for q in paid
            ):
                raise ValueError("The reused diagnostic prefix is not paid on these coordinates")
            probabilities = prefix_components(base, source, x, paid, state)
            weights = np.asarray(state["component_weights"])
            mixture = weights @ probabilities
            original = np.asarray(state["probabilities"])
            error = float(np.max(np.abs(mixture[: len(original)] - original)))
            maximum = max(maximum, error)
            if error > 1e-8:
                raise ValueError(
                    "Reconstructed common-prefix states disagree with saved predictions"
                )
            names = ["mixture", *common["result"]["component_names"]]
            for name, values, weight in zip(names, [mixture, *probabilities], [1.0, *weights]):
                measured = prediction_metrics(
                    {"probabilities": values, "observed_indices": indices}, target_truth
                )
                output.append(
                    {
                        "sut_id": sut,
                        "scenario": kind,
                        "delta_discoveries": delta,
                        "paid_prefix": prefix,
                        "component": name,
                        "weight": float(weight),
                        **measured,
                    }
                )
        cases.append(
            {
                "sut_id": sut,
                "scenario": kind,
                "delta_discoveries": delta,
                "delta_auc": row["metrics"]["discovery_auc"]
                - lookup[sut, kind, "no_history"]["metrics"]["discovery_auc"],
                "common_feedback": relative(common_path),
                "max_reconstruction_error": maximum,
                "prefix_100_history_mass": float(weights[:-2].sum()),
                "prefix_100_mixture_top_200": next(
                    r["top_200_failures"]
                    for r in output
                    if r["sut_id"] == sut
                    and r["scenario"] == kind
                    and r["paid_prefix"] == 100
                    and r["component"] == "mixture"
                ),
                "prefix_100_raw_top_200": next(
                    r["top_200_failures"]
                    for r in output
                    if r["sut_id"] == sut
                    and r["scenario"] == kind
                    and r["paid_prefix"] == 100
                    and r["component"] == "physical_target"
                ),
                "feedback_coordinate_scope": "original grid"
                if kind == "car_following"
                else "unchanged full candidate library",
            }
        )
    if output:
        write_csv(OUTPUT / "reports/negative_transfer_components.csv", output)
    result = {
        "status": "complete",
        "cases": cases,
        "new_queries": 0,
        "new_physical_executions": 0,
        "new_optimization_steps": 0,
        "selection": "every task with fewer final discoveries than its matched no-history control",
        "scope": "Existing Random-prefix states at 20/50/100; target truth is used only for scoring. Components share the saved paid state and historical representation.",
    }
    write_json(OUTPUT / "reports/negative_transfer_diagnosis.json", result)
    return result


def write_results(output, pairs):
    table = output["overall"]
    by_method = {r["method"]: r for r in table}
    negative = [p for p in pairs if p["comparator"] == "no_history" and p["delta_failures"] < 0]
    tradeoff = (
        f"最终数量较少的{len(negative)}个任务中，发现AUC均高于无历史对照，体现了更早发现与最终数量的取舍；这些任务仍完整保留。"
        if negative and all(p["delta_auc"] > 0 for p in negative)
        else "数量和发现速度分别呈现，全部不利任务保留。"
    )
    lines = [
        "# 旧平台统一候选后的论文补充验证结果",
        "",
        f"最终核：`{output['selected_kernel']}`。八目标、四功能场景，各2,000候选；共同初始化20次计入总预算200，种子11。",
        "",
        "跟车保留原1,000点网格，再增加1,000个独立范围上的扰动Sobol设计点。共同初始化沿用原网格中的20个ID，不是从扩展库重新抽取。其余三类坐标保持不变。该混合设计不称为纯Sobol或独立同分布抽样；参数包络为登记的工程设计范围。",
        "",
        f"复用22,528条历史和56,000条目标真值，新增8,000条跟车执行；离线训练0步。新增{output['new_method_runs']}次方法回放，合并为320个主比较单元。两核64单元一并保留，模型选择使用同一批目标，属于基准内模型比较。",
        "",
        "## 数量、发现速度与真值覆盖",
        "",
        "真实失效依据实际碰撞，任务内按唯一候选ID计数；不同ADS不合并候选身份。AUC为200步累计发现数的均值，覆盖率分母为对应完整候选库的真实碰撞总数。",
        "",
        "| 方法 | 失效数 | 初始化后新增 | 平均发现AUC | 汇总覆盖率 | 宏覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table:
        lines.append(
            f"| {STYLES[row['method']][0]} | {row['failures']} | {row['post_initial_failures']} | {row['mean_auc']:.4f} | {row['pooled_coverage']:.2%} | {row['macro_coverage']:.2%}（{row['coverage_defined_tasks']}/32） |"
        )
    lines += [
        "",
        "## 贡献方向",
        "",
        "下表为完整方法减对应对照。无历史对照同时移除历史先验和历史表征；目标仅碰撞对照保持同一历史模型；固定来源权重仍保留目标适配。",
        "",
        "| 对照 | 发现数量变化 | 平均AUC变化 | 数量胜／负／平 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in (*PRIMARY_BASELINES, *CONTROLS):
        selected = [p for p in pairs if p["comparator"] == method]
        differences = [p["delta_failures"] for p in selected]
        lines.append(
            f"| {STYLES[method][0]} | {sum(differences):+d} | {np.mean([p['delta_auc'] for p in selected]):+.4f} | {sum(v > 0 for v in differences)}／{sum(v < 0 for v in differences)}／{sum(v == 0 for v in differences)} |"
        )
    rss_delta = by_method["joint"]["failures"] - by_method["collision"]["failures"]
    lines += [
        "",
        f"目标RSS的数量净变化为{rss_delta:+d}。RSS的历史训练作用与目标条件化作用分别解释；原历史RSS流程对照保留离线训练差异，不能替代当前核的纯似然消融。来源分离和元训练没有独立匹配证据时，不单独归因。",
        "",
        "## 分场景结果",
        "",
        "| 场景 | 完整方法 | 无历史GP | ScenarioFuzz | RF-EI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kind in TEST_SCENARIOS:
        values = {r["method"]: r["failures"] for r in output["by_scenario"][kind]}
        lines.append(
            f"| {TITLES[kind]} | {values['joint']} | {values['no_history']} | {values['scenario_fuzz']} | {values['av_bo_rf_ei']} |"
        )
    lines += [
        "",
        "跟车原网格与新增Sobol部分的真值、发现数分别见[分段统计](follow_segments.csv)，两部分共享同一个200次预算，不分别运行或择优合并。初始几何可行不意味着碰撞对所有控制策略都可避免；失效率属于工程候选空间，不解释为自然驾驶事故率。",
        "",
        "## 控制器分组",
        "",
        "| 固定分组 | 完整方法 | 无历史GP | ScenarioFuzz |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in output["by_controller_group"].items():
        found = {r["method"]: r["failures"] for r in values}
        lines.append(
            f"| {name} | {found['joint']} | {found['no_history']} | {found['scenario_fuzz']} |"
        )
    lines += [
        "",
        "## 参数空间多样性",
        "",
        "使用原自由参数按冻结范围归一化、维度等权的Gower距离。同预算统计见metrics.csv；配对比较取双方按发现顺序最先找到的前k个失效，k为双方较小发现数。未挑选最分散子集，PCA图片不参与距离计算。",
        "",
        "| 对照 | 同发现数两两距离变化 | 同发现数最近邻距离变化 | 有效任务数 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in (*PRIMARY_BASELINES, *CONTROLS):
        eligible = [p for p in pairs if p["comparator"] == method and p["matched_count"] >= 2]
        if eligible:
            distances = [
                np.mean([p[f"joint_{m}"] - p[f"control_{m}"] for p in eligible])
                for m in ("pair_distance", "nearest_neighbor_distance")
            ]
            lines.append(
                f"| {STYLES[method][0]} | {distances[0]:+.5f} | {distances[1]:+.5f} | {len(eligible)} |"
            )
    lines += [
        "",
        "数量、速度和距离分别判断；更多发现不自动解释为更大参数距离。零失效库的覆盖率、少于两次发现的距离标为未定义，完整任务表保留零失效及不利结果。单种子不作跨种子显著性结论。",
        "",
        "各方法在其自身付费路径后的未查询预测NLL／Brier见[prediction_metrics.csv](prediction_metrics.csv)。不同方法的已查询集合可能不同，该表不作为相同反馈前缀的因果比较；排序诊断命中数不计入实际发现。",
        "",
        "## 负迁移任务的共同反馈诊断",
        "",
        "对完整方法少于匹配无历史对照的全部任务，重用既有Random付费前缀在20／50／100条时保存的参数与来源权重，分解来源、独立目标和原始参数分量。未增加查询或参数适配步；剩余真值仅用于评分。具体数据见[条件预测分解](negative_transfer_diagnosis.json)。",
        "",
        tradeoff,
        "",
        "共同前缀排名与实际自适应选例路径分别解释；这些分量共享对应付费状态，不单独隔离深表征。跟车沿用旧网格反馈时，在扩展后的2,000点上评分并注明反馈范围。诊断命中不计入实际发现。",
        "",
        "## 知识积累",
        "",
        "新增任务均完成付费归档并引用共享执行，保留原ADS身份、顺序及启动快照。本轮不新增顺序矩阵。旧顺序实验使用原核及replicate=20：固定历史2,083、追加来源2,073、追加并更新模型2,128；属于原配置证据，不替代本次最终核的积累实验。",
        "",
        "## 成本、产物与验收",
        "",
        f"新增物理调用{output['costs']['new_physical_calls']}次，成功{output['costs']['new_physical_successes']}次。新增方法回放{output['new_method_runs']}次、{output['new_method_runs'] * 200:,}次逻辑查询；查询复用真值，不再计作物理仿真。",
        "",
        "运行成本见[costs.csv](costs.csv)，物理采集成本见[collection.json](collection.json)。各运行墙钟求和不等于并行端到端耗时，RSS不与包含它的回放时间重复相加。复用记录与新增运行来自不同批次，不作为严格硬件性能对照。首次跟车缓存导出未单独计时；仅对最终所选核复算一次相同先验，核对数值一致并保存[source_export_timing.json](source_export_timing.json)，该计时不冒充首次导出耗时。旧离线构建成本保留原报告引用。",
        "",
        "[数据核验](data_audit.json)、[结果核验](results_audit.json)、[模型选择](model_selection.json)、[全部任务](task_metrics.csv)、[配对距离](paired_differences.csv)均保留。600 dpi PNG和实际20 Hz轨迹GIF位于figures/。",
    ]
    (OUTPUT / "reports/results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report():
    started = time.perf_counter()
    if read_json(OUTPUT / "reports/results_audit.json")["status"] != "passed":
        raise ValueError("Result acceptance is required before final paper figures")
    protocol = read_json(OUTPUT / "reports/protocol.json")
    summary = read_json(OUTPUT / "reports/summary.json")
    rows = summary["runs"]
    methods = ("joint", *PRIMARY_BASELINES, *CONTROLS)
    table = aggregate(rows, methods)
    pairs = paired_distances(rows, methods)
    directory = OUTPUT / "reports"
    write_csv(directory / "metrics.csv", table)
    write_csv(directory / "paired_differences.csv", pairs)
    per_task = [
        {
            "sut_id": r["sut_id"],
            "scenario": r["scenario"],
            "method": r["method"],
            "kernel": r["kernel"],
            "path": r["path"],
            "operation": r["operation"],
            **{k: v for k, v in r["metrics"].items() if not isinstance(v, (list, dict))},
        }
        for r in rows
    ]
    write_csv(directory / "task_metrics.csv", per_task)
    grouped = {
        kind: aggregate([r for r in rows if r["scenario"] == kind], methods)
        for kind in TEST_SCENARIOS
    }
    families = {
        group: aggregate(
            [
                r
                for r in rows
                if (r["sut_id"] in ("independent_ppo", "independent_dqn")) == learning
            ],
            methods,
        )
        for group, learning in (("rule_optimization_search", False), ("ppo_dqn", True))
    }
    figures = directory / "figures"
    main = ("joint", *PRIMARY_BASELINES)
    plot_curves(rows, main, figures / "discovery.png")
    plot_curves(rows, ("joint", *CONTROLS), figures / "controls.png")
    plot_curves(rows, main, figures / "coverage.png", coverage=True)
    plot_diversity(pairs, figures / "matched_diversity.png")
    plot_space(rows, protocol, figures / "parameter_space.png")
    plot_adaptation(rows, protocol, figures / "adaptation.png")
    selection = read_json(directory / "model_selection.json")
    kernel_rows = [
        {**r, "method": "joint" if r["kernel"] == "joint" else "no_history"}
        for r in selection["runs"]
    ]
    plot_curves(
        kernel_rows,
        ("joint", "no_history"),
        figures / "kernel_comparison.png",
        labels={"joint": "Original joint kernel", "no_history": "Shared/private kernel"},
    )
    primary = next(
        r
        for r in rows
        if r["sut_id"] == protocol["systems"][0]["sut_id"]
        and r["scenario"] == "car_following"
        and r["method"] == "joint"
    )
    bank = task_bank(primary["sut_id"], "car_following")
    saved = read_json(ROOT / primary["path"])
    index = next(
        (
            i
            for i in saved["result"]["selected_indices"]
            if i >= 1000 and bank(bank.candidates[i]).feedback.collision
        ),
        saved["result"]["selected_indices"][0],
    )
    execution = bank(bank.candidates[index])
    case = render_execution(
        bank.candidates[index],
        execution.trajectory,
        figures / "case",
        static=True,
        provenance={
            "execution_ref": relative(bank.execution_path(index)),
            "method_result": primary["path"],
        },
    )
    output = {
        "status": "complete",
        "selected_kernel": summary["selected_kernel"],
        "overall": table,
        "by_scenario": grouped,
        "by_controller_group": families,
        "new_physical_executions": 8000,
        "new_method_runs": summary["new_kernel_runs"] + summary["new_comparison_runs"],
        "kernel_comparison": selection["totals"],
        "case": case,
        "figures": [relative(p) for p in figures.glob("*.png")],
        "single_algorithm_seed": SEED,
        "cross_seed_intervals": False,
        "model_selection_uses_evaluation_targets": True,
    }
    output["costs"] = costs(rows, selection)
    output["follow_segments"] = follow_segments(rows)
    output["prediction"] = prediction_summary(rows)
    output["negative_transfer_diagnosis"] = diagnose_negative_transfers(rows, protocol)
    output["knowledge_archive_check"] = read_json(directory / "knowledge_archive_check.json")
    output["acceptance"] = {
        k: v for k, v in read_json(directory / "results_audit.json").items() if k != "runs"
    }
    write_results(output, pairs)
    output["reporting_wall_s"] = time.perf_counter() - started
    write_json(directory / "report.json", output)
    return output


if __name__ == "__main__":
    archive_reused_tasks()
    audit_results()
    report()
