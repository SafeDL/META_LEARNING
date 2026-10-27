"""Paired independent discovery, source holdout, interaction mechanisms and costs."""

from collections import Counter, defaultdict

import numpy as np

from controllers.highway.catalog import INDEPENDENT_SUTS, SUTS
from controllers.highway.learning import CHECKPOINT_ROOT

from .collision_metrics import equal_count_coverage, lineage_survival, paired_lineage_bootstrap
from .discovery_config import read_json, write_json
from .domain_comparison_report import METRICS
from .independent_models import FAMILIES, model_root, scope_name
from .independent_preparation import OUTPUT
from .independent_protocol import (
    CONDITIONS,
    CONTROLS,
    METHODS,
    jobs,
    methods_for,
    reuse_jobs,
    run_path,
    target_bank_path,
    tasks,
    validate_protocol,
)
from .independent_validation import bank_audit_path
from .qualification import QUALIFICATION_ROOT
from .scenarios import DOMAINS, dedup_key, sample
from .statistics import method_summary, summary


def paired_effects(rows, method, references):
    effects = []
    for other in references:
        for metric in METRICS:
            chosen = [r for r in rows if r["method"] in (method, other) and r[metric] is not None]
            if chosen:
                effects.append(paired_lineage_bootstrap(chosen, method, other, metric))
    return effects


def curves(rows, methods):
    result = {}
    for method in methods:
        chosen = [r for r in rows if r["method"] == method]
        result[method] = {}
        for metric in ("discovery_curve", "recall_curve"):
            groups = defaultdict(list)
            for row in chosen:
                if row[metric][0] is not None:
                    groups[row["lineage"]].append(row[metric])
            result[method][metric] = (
                np.mean([np.mean(group, axis=0) for group in groups.values()], axis=0).tolist()
                if groups
                else [None] * 128
            )
        result[method]["checkpoints"] = {
            str(budget): {
                "unique_collisions": result[method]["discovery_curve"][budget - 1],
                "collision_recall": result[method]["recall_curve"][budget - 1],
            }
            for budget in (16, 32, 64, 128)
        }
    return result


def interaction_analysis(rows):
    scenes = sample("cut_in_interaction", 512, "new")
    pools, allocated = [], []
    for sut in INDEPENDENT_SUTS:
        key = {"target_id": sut.sut_id, "scenario": "cut_in_interaction"}
        private = read_json(bank_audit_path(key))
        labels = [f["collision"] for f in private["feedback"]]
        pairs = Counter((labels[i], labels[i + 1]) for i in range(0, 512, 2))
        pools.append(
            {
                "target_id": sut.sut_id,
                "lineage": sut.lineage,
                "paired_parameter_points": 256,
                "neither": pairs[False, False],
                "restricted_only": pairs[False, True],
                "open_only": pairs[True, False],
                "both": pairs[True, True],
                "open_collisions": sum(labels[::2]),
                "restricted_collisions": sum(labels[1::2]),
                "restricted_collision_partners": dict(
                    Counter(
                        p
                        for i, p in enumerate(private["collision_partners"])
                        if i % 2 and p is not None
                    )
                ),
            }
        )
        for row in [
            r
            for r in rows
            if r["target_id"] == sut.sut_id and r["scenario"] == "cut_in_interaction"
        ]:
            task = {k: row[k] for k in tasks()[0]}
            queries = read_json(run_path(task, row["method"]))["result"]["queries"]
            for space in ("open", "restricted"):
                selected = [
                    q for q in queries if scenes[q["index"]].categories["adjacent_space"] == space
                ]
                found = {
                    dedup_key(scenes[q["index"]]) for q in selected if q["feedback"]["collision"]
                }
                denominator = {
                    dedup_key(s)
                    for s, label in zip(scenes, labels)
                    if label and s.categories["adjacent_space"] == space
                }
                latencies = [
                    private["events"][q["index"]]["brake_latency_from_scheduled_event_s"]
                    for q in selected
                ]
                observed = [v for v in latencies if v is not None]
                maneuvers = [
                    private["events"][q["index"]].get("ego_lane_change_onset_s") for q in selected
                ]
                allocated.append(
                    {
                        **task,
                        "method": row["method"],
                        "adjacent_space": space,
                        "queries_allocated": len(selected),
                        "unique_collisions": len(found),
                        "pool_unique_collisions": len(denominator),
                        "recall_under_joint_budget": len(found) / len(denominator)
                        if denominator
                        else None,
                        "mean_observed_brake_latency_s": float(np.mean(observed))
                        if observed
                        else None,
                        "no_observed_post_event_brake_queries": len(latencies) - len(observed),
                        "queries_with_ego_lane_change": sum(
                            value is not None for value in maneuvers
                        ),
                        "paid_collision_partner_counts": dict(
                            Counter(
                                private["collision_partners"][q["index"]]
                                for q in selected
                                if q["feedback"]["collision"]
                            )
                        ),
                    }
                )
    aggregated = {
        condition: {
            space: {
                method: {
                    metric: summary(
                        [
                            r
                            for r in allocated
                            if r["condition"] == condition
                            and r["adjacent_space"] == space
                            and r["method"] == method
                        ],
                        metric,
                    )
                    for metric in (
                        "queries_allocated",
                        "unique_collisions",
                        "recall_under_joint_budget",
                    )
                }
                for method in (
                    *methods_for(condition),
                    *(CONTROLS if condition == "all_sources" else ()),
                )
            }
            for space in ("open", "restricted")
        }
        for condition in CONDITIONS
    }
    return {
        "budget": "128 shared by open and restricted conditions, including the common eight",
        "complete_paired_pools": pools,
        "paid_allocations": allocated,
        "summary": aggregated,
    }


def policy_training_cost(sut):
    record = read_json(CHECKPOINT_ROOT / sut.training_run / "training.json")
    return {
        "sut_id": record["sut_id"],
        "training_seed": record["training_seed"],
        "decision_steps": record["actual_timesteps"],
        "training_and_qualification_wall_s": record["elapsed_s"],
        "qualification_executions": record["qualification_executions"],
    }


def costs(rows, reuse_references=()):
    data = read_json(OUTPUT / "construction_data_audit.json")["banks"]
    target = read_json(OUTPUT / "target_bank_index.json")
    training = []
    for kind in DOMAINS:
        for held in FAMILIES:
            root = model_root("gcrt", kind, held)
            main = {
                part: read_json(root / filename)
                for part, filename in (
                    ("target", "target_kernel.json"),
                    ("encoder", "historical_encoder.json"),
                    ("priors", "source_priors.json"),
                )
            }
            classifiers = {
                owner: read_json(
                    OUTPUT
                    / "model_construction/required_classifiers"
                    / owner
                    / kind
                    / f"{scope_name(held)}.json"
                )
                for owner in ("scenario_fuzz", "indago")
                if owner != "indago" or held is None
            }
            training.append(
                {
                    "scenario": kind,
                    "scope": scope_name(held),
                    "main_target_training_s": main["target"]["training_s"],
                    "main_target_new_training_s": main["target"]["new_training_s"],
                    "main_encoder_including_prefix_cache_s": main["encoder"]["training_s"],
                    "main_prefix_cache_s": main["encoder"]["prefix_cache_s"],
                    "historical_prior_construction_s": main["priors"]["construction_s"],
                    "classifier_training": {
                        owner: {
                            "original_training_s": record["report"]["training_s"],
                            "new_training_s": record["new_training_s"],
                            "reused": record["reused"],
                        }
                        for owner, record in classifiers.items()
                    },
                }
            )
    source_suts = [s for s in SUTS if s.role in ("history", "development")]
    source_policy = [policy_training_cost(s) for s in source_suts if s.family == "learning"]
    policy = [policy_training_cost(s) for s in INDEPENDENT_SUTS if s.family == "learning"]
    qualification = {
        name: [
            read_json(QUALIFICATION_ROOT / s.sut_id / "qualification.json")
            for s in objects
            if s.family != "learning"
        ]
        for name, objects in (("reused_source", source_suts), ("new_independent", INDEPENDENT_SUTS))
    }
    bank_costs = []
    for bank in target["banks"]:
        manifest = read_json(target_bank_path(bank) / "manifest.json")
        bank_costs.append(
            {
                **bank,
                **{
                    key: manifest[key]
                    for key in (
                        "successful_calls",
                        "total_successful_call_wall_s",
                        "sut_initialization_s",
                        "failed_call_wall_s",
                        "interrupted_call_duration_unknown",
                        "last_collection_wall_s",
                    )
                },
            }
        )
    online = []
    for row in rows:
        task = {k: row[k] for k in tasks()[0]}
        saved = read_json(run_path(task, row["method"]))
        online.append(
            {
                **task,
                "method": row["method"],
                **saved["cost"],
                "selection_s": sum(q.get("selection_s", 0) for q in saved["result"]["queries"]),
                "adaptation_s": sum(q.get("adaptation_s", 0) for q in saved["result"]["queries"]),
                "retained_paid_simulation_s": sum(
                    q["feedback"]["execution_s"] for q in saved["result"]["queries"]
                ),
            }
        )
    return {
        "allowed_history_development_banks": data,
        "reused_old_history_development_calls": sum(
            r["actual_physical_calls"] for r in data if r["scenario"] != "cut_in_interaction"
        ),
        "new_interaction_history_development_calls": sum(
            r["actual_physical_calls"] for r in data if r["scenario"] == "cut_in_interaction"
        ),
        "reused_source_policy_runs": source_policy,
        "new_target_pool_attempts": target["attempted_calls"],
        "target_banks": bank_costs,
        "new_target_pool_collection_wall_s": target["last_collection_wall_s"],
        "nonlearning_qualification": {
            name: {
                "course_calls": sum(len(r["courses"]) for r in reports),
                "simulated_s": sum(c["physics_ticks"] / 20 for r in reports for c in r["courses"]),
                "wall_s": None,
            }
            for name, reports in qualification.items()
        },
        "new_policy_runs": policy,
        "model_construction": training,
        "additional_classifier_execution_schedules": {
            path.name: read_json(path)
            for path in sorted((OUTPUT / "model_construction").glob("classifiers_*_parallel.json"))
        },
        "early_public_prior_schedule": (
            read_json(OUTPUT / "model_construction/partial_priors_schedule.json")
            if (OUTPUT / "model_construction/partial_priors_schedule.json").exists()
            else None
        ),
        "online_runs": online,
        "source_agnostic_reuse": {
            "references": len(reuse_references),
            "new_queries": 0,
            "new_physical_calls": 0,
            "new_selection_wall_s": 0,
        },
        "online_summary": {
            condition: {
                method: {
                    metric: summary(
                        [
                            r
                            for r in online
                            if r["condition"] == condition and r["method"] == method
                        ],
                        metric,
                    )
                    for metric in (
                        "wall_s",
                        "construction_and_loading_s",
                        "selection_s",
                        "adaptation_s",
                        "retained_paid_simulation_s",
                    )
                }
                for method in sorted({r["method"] for r in online if r["condition"] == condition})
            }
            for condition in CONDITIONS
        },
        "new_physical_calls_during_selector_replays": 0,
        "wall_time_scope": "component wall times under recorded concurrency; summing parallel model times is not elapsed workflow latency; retained simulation seconds are separate",
    }


def plot(curve_sets):
    import matplotlib.pyplot as plt

    names = {
        "gcrt": "Ours",
        "random": "Random",
        "av_bo_rf_ei": "RF-EI",
        "samota": "SAMOTA",
        "scenario_fuzz": "ScenarioFuzz",
        "paired_system_active_evaluation": "CAVE",
        "indago": "Indago",
    }
    colors = {
        m: c
        for m, c in zip(
            METHODS, ("#a51c30", "#777777", "#009E73", "#E69F00", "#0072B2", "#CC79A7", "#56B4E9")
        )
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column, condition in enumerate(CONDITIONS):
        for method in methods_for(condition):
            for row, metric in enumerate(("discovery_curve", "recall_curve")):
                values = curve_sets[condition][method][metric]
                if values[0] is not None:
                    axes[row, column].plot(
                        range(1, 129),
                        values,
                        label=names[method],
                        color=colors[method],
                        linewidth=2.5 if method == "gcrt" else 1.3,
                    )
        axes[0, column].set_title(
            "All sources" if condition == "all_sources" else "Strict family holdout"
        )
        axes[1, column].set_xlabel("Paid queries (common 8 included)")
        for axis in axes[:, column]:
            axis.grid(alpha=0.2)
            axis.set_xlim(1, 128)
        axes[1, column].set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("Unique collisions discovered")
    axes[1, 0].set_ylabel("Finite-pool collision recall")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    for extension in ("png", "svg", "pdf"):
        figure.savefig(OUTPUT / f"discovery_curves.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_mechanisms(curve_set):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = {
        "gcrt": "Ours (H1F1)",
        "H1F0": "Static history (H1F0)",
        "H0F1": "Aggregate + feedback (H0F1)",
    }
    colors = {"gcrt": "#a51c30", "H1F0": "#777777", "H0F1": "#0072B2"}
    for axis, metric, title in zip(
        axes,
        ("discovery_curve", "recall_curve"),
        ("Unique collisions discovered", "Finite-pool collision recall"),
    ):
        for method in ("gcrt", *CONTROLS):
            values = curve_set[method][metric]
            if values[0] is not None:
                axis.plot(
                    range(1, 129),
                    values,
                    label=labels[method],
                    color=colors[method],
                    linewidth=2.5 if method == "gcrt" else 1.5,
                )
        axis.set_xlabel("Paid queries (common 8 included)")
        axis.set_ylabel(title)
        axis.set_xlim(1, 128)
        axis.grid(alpha=0.2)
    axes[1].set_ylim(0, 1.02)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Paired open/restricted cut-in: one shared 128-query budget")
    figure.tight_layout(rect=(0, 0.1, 1, 0.93))
    for extension in ("png", "svg", "pdf"):
        figure.savefig(OUTPUT / f"mechanism_curves.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def comparison_rows(index, reuse):
    rows = list(index["runs"])
    by_path = {row["path"]: row for row in rows}
    for reference in reuse["references"]:
        source = by_path[reference["source_path"]]
        rows.append(
            {
                **source,
                **reference["task"],
                "reused": True,
                "reuse_source_sha256": reference["source_sha256"],
            }
        )
    return rows


def transfer_effects(rows):
    results = {}
    for method in ("gcrt", "scenario_fuzz"):
        paired = [
            {**row, "method": row["condition"], "condition": "matched_source_conditions"}
            for row in rows
            if row["method"] == method
        ]
        results[method] = paired_effects(paired, "strict_leave_family", ("all_sources",))
    return results


def report():
    validate_protocol()
    audit = read_json(OUTPUT / "audit.json")
    index = read_json(OUTPUT / "run_index.json")
    reuse = read_json(OUTPUT / "reuse_index.json")
    if (
        audit["status"] != "passed"
        or audit["runs"] != len(jobs())
        or audit["queries"] != 128 * len(jobs())
        or audit["reused_references"] != len(reuse_jobs())
        or audit["independent_main_choices"] != 1440
        or reuse["status"] != "complete"
        or len(reuse["references"]) != len(reuse_jobs())
        or index["status"] != "complete"
        or len(index["runs"]) != len(jobs())
    ):
        raise ValueError("Complete and audit the whole independent matrix before reporting")
    rows = comparison_rows(index, reuse)
    conditions, curve_sets = {}, {}
    for condition in CONDITIONS:
        methods = methods_for(condition)
        references = tuple(method for method in methods if method != "gcrt")
        chosen = [r for r in rows if r["condition"] == condition and r["method"] in methods]
        stats = method_summary(chosen)
        strongest = max(
            references,
            key=lambda m: (
                stats[m]["collision_recall"]["mean"]
                if stats[m]["collision_recall"]["mean"] is not None
                else -1,
                stats[m]["unique_collisions"]["mean"],
                stats[m]["discovery_auc"]["mean"],
                m,
            ),
        )
        conditions[condition] = {
            "summary": stats,
            "strongest_reference": strongest,
            "paired_comparisons": paired_effects(chosen, "gcrt", references),
            "paid_auxiliary_measurements": {
                method: {
                    metric: summary([r for r in chosen if r["method"] == method], metric)
                    for metric in (
                        "behavior_coverage",
                        "collision_executions",
                        "minimum_clearance_m",
                        "minimum_ttc_s",
                        "brier",
                    )
                }
                for method in methods
            },
            "first_collision_survival": {
                m: lineage_survival([r for r in chosen if r["method"] == m], restriction=128)
                for m in methods
            },
        }
        curve_sets[condition] = curves(chosen, methods)
    mechanism_rows = [
        r
        for r in rows
        if r["condition"] == "all_sources"
        and r["scenario"] == "cut_in_interaction"
        and r["method"] in ("gcrt", *CONTROLS)
    ]
    mechanisms = {
        "summary": method_summary(mechanism_rows),
        "paired_comparisons": paired_effects(mechanism_rows, "gcrt", CONTROLS),
        "curves": curves(mechanism_rows, ("gcrt", *CONTROLS)),
        "scope": "40 full-source interaction tasks; H1F1 reuses E1; two matched controls; no new full factorial interaction estimate",
    }
    coverage = []
    for task in tasks():
        chosen = [
            r for r in rows if all(r[k] == v for k, v in task.items()) and r["method"] in METHODS
        ]
        coverage.append(
            {"task": task, "methods": [r["method"] for r in chosen], **equal_count_coverage(chosen)}
        )
    result = {
        "status": "complete_audited_independent_validation",
        "runs": len(jobs()),
        "equivalent_queries": 128 * len(jobs()),
        "reused_references": len(reuse["references"]),
        "logical_comparison_rows": len(rows),
        "independent_lineages": 8,
        "source_conditions": conditions,
        "mechanisms": mechanisms,
        "within_method_source_holdout_effects": transfer_effects(rows),
        "primary_transfer_reference": "scenario_fuzz",
        "by_scenario": {
            condition: {
                kind: method_summary(
                    [
                        r
                        for r in rows
                        if r["condition"] == condition
                        and r["scenario"] == kind
                        and r["method"] in METHODS
                    ]
                )
                for kind in DOMAINS
            }
            for condition in CONDITIONS
        },
        "interaction_task": interaction_analysis(rows),
        "equal_collision_count_behavior_coverage": coverage,
        "cost": costs(index["runs"], reuse["references"]),
        "development_gate_rewritten": False,
        "independent_target_results_used_for_model_selection": False,
    }
    write_json(OUTPUT / "report.json", result)
    write_json(OUTPUT / "curves.json", curve_sets)
    plot(curve_sets)
    plot_mechanisms(mechanisms["curves"])
    lines = [
        "# 独立驾驶系统的有限预算碰撞发现",
        "",
        "8个独立对象、5类场景、5个配对种子；全来源主比较1,400项、严格留家族新增400项、新交互任务两个控制80项。全部1,880项、240,640次等效查询及600项复用引用通过审计。",
        "",
        "严格留家族中的Random、RF-EI和SAMOTA引用相同任务的全来源记录，不重复查询或计入执行成本；Indago及配对系统方法保留在完整主比较中。",
        "",
    ]
    for condition in CONDITIONS:
        lines += [
            f"## {condition}",
            "",
            "| 方法 | 去重碰撞数 | 有限库召回 | 发现AUC |",
            "|---|---:|---:|---:|",
        ]
        for method in methods_for(condition):
            stats = conditions[condition]["summary"][method]
            values = [stats[metric]["mean"] for metric in METRICS]
            lines.append(
                "| "
                + method
                + " | "
                + " | ".join("不适用" if v is None else f"{v:.6f}" for v in values)
                + " |"
            )
        lines += [
            "",
            f"本条件最强参考为{conditions[condition]['strongest_reference']}；完整配对区间和谱系统计见JSON。",
            "",
        ]
    lines += [
        "## 新交互任务的历史与反馈对照",
        "",
        "主方法与静态历史排序H1F0、同信息聚合教师反馈H0F1比较，主方法记录复用主比较。保留各项差异及区间，不将三单元比较解释为完整H×F交互量。",
        "",
        "| 单元 | 去重碰撞数 | 有限库召回 | 发现AUC |",
        "|---|---:|---:|---:|",
    ]
    for method in ("gcrt", *CONTROLS):
        values = [mechanisms["summary"][method][metric]["mean"] for metric in METRICS]
        lines.append(
            "| "
            + method
            + " | "
            + " | ".join("不适用" if v is None else f"{v:.6f}" for v in values)
            + " |"
        )
    lines += [
        "## 场景机制与成本",
        "",
        "开放/受限切入条件各256个、共用128次预算；配对条件变化、预算分配和碰撞对象单独报告。",
        "",
        "来源建设、模型训练、在线选例与保留仿真成本分项列在report.json，不把选例种子作为独立系统。",
        "",
        "![发现曲线](discovery_curves.png)",
        "",
        "![交互任务机制对照](mechanism_curves.png)",
        "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
