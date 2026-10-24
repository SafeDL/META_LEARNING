"""Exhaustive historical/development audit kept outside selector processes."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from controllers.highway.catalog import BY_ID

from .collision_metrics import behavior_signature, collision_recall
from .construction import load_construction
from .discovery_config import OUTPUT, SCENARIOS, development_tasks, write_json
from .execution_events import analyze_events
from .executions import ExecutionBank, execution_contract
from .protocol import DATA_ROOT, DEFAULT
from .scenarios import DOMAINS


def audit_bank(archive):
    kind = archive.scenarios[0].kind
    root = DATA_ROOT / "physical" / archive.role / kind / archive.source_id
    bank = ExecutionBank(root)
    if (bank.candidates != archive.scenarios
            or bank.manifest["contract"] != execution_contract(BY_ID[archive.source_id])
            or bank.manifest["environment_seed"] != 81001):
        raise ValueError(f"Execution bank differs from construction: {root}")
    labels, signatures = [], Counter()
    latency, event_seen = [], 0
    for index, scenario in enumerate(bank.candidates):
        execution = bank(scenario)
        label = execution.feedback.collision
        safety = execution.feedback.safety
        if (label != bool(archive.collision[index])
                or not np.array_equal([safety.min_clearance_m, safety.min_ttc_s], archive.safety[index])
                or label != any(row["collision"] for row in execution.trajectory)):
            raise ValueError(f"Raw feedback/trajectory differs from archive: {root}, {index}")
        labels.append(label)
        events = analyze_events(scenario, execution.trajectory)
        event_seen += events["measured_background_onset_s"] is not None
        value = events["brake_latency_from_scheduled_event_s"]
        if value is not None:
            latency.append(value)
        if label:
            partner, contact, action, phase = behavior_signature(scenario, execution.trajectory)
            signatures[(partner, contact, "/".join(action), phase)] += 1
    pool = collision_recall(bank.candidates, labels, [])
    return {"source_id": archive.source_id, "role": archive.role, "scenario": kind,
            "lineage": archive.lineage, "path": str(root.relative_to(DATA_ROOT.parent.parent)),
            "status": "passed", "candidates": len(labels), "pool_collisions": labels,
            "pool_unique_collisions": pool["pool_unique_collisions"],
            "collision_executions": sum(labels), "events_observed": event_seen,
            "brake_latency_count": len(latency),
            "mean_brake_latency_s": float(np.mean(latency)) if latency else None,
            "collision_signatures": [{"signature": list(k), "count": v}
                                      for k, v in sorted(signatures.items())],
            "physical_bank_calls": bank.manifest["attempted_calls"],
            "physical_bank_execution_s": bank.manifest["physical_execution_s"]}


def pair_diagnosis(archives):
    labels = np.stack([a.collision for a in archives])
    x = archives[0].x
    groups = np.minimum((x[:, 1] * 4).astype(int), 3)
    pairs = []
    for i, left in enumerate(archives):
        for j in range(i + 1, len(archives)):
            right = archives[j]
            codes = 2 * labels[i].astype(int) + labels[j].astype(int)
            global_correct = local_correct = 0
            folds = []
            for block in np.unique(groups):
                train, test = groups != block, groups == block
                majority = np.bincount(codes[train], minlength=4).argmax()
                tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=16, random_state=1501)
                tree.fit(x[train], codes[train])
                local = int((tree.predict(x[test]) == codes[test]).sum())
                constant = int((codes[test] == majority).sum())
                local_correct += local
                global_correct += constant
                folds.append({"held_gap_block": int(block), "test_points": int(test.sum()),
                              "local_accuracy": local / int(test.sum()),
                              "majority_accuracy": constant / int(test.sum())})
            pairs.append({"left": left.source_id, "right": right.source_id,
                          "both_failure": int((labels[i] & labels[j]).sum()),
                          "left_only": int((labels[i] & ~labels[j]).sum()),
                          "right_only": int((~labels[i] & labels[j]).sum()),
                          "both_safe": int((~labels[i] & ~labels[j]).sum()),
                          "blocked_local_accuracy": local_correct / len(x),
                          "blocked_majority_accuracy": global_correct / len(x), "folds": folds})
    return {"candidates": len(x), "historical_counts": {a.source_id: int(a.collision.sum()) for a in archives},
            "all_safe": int((~labels.any(0)).sum()), "all_failure": int(labels.all(0).sum()),
            "disagreement_points": int((labels.any(0) & ~labels.all(0)).sum()),
            "response_patterns": len(np.unique(labels.T, axis=0)),
            "spatial_validation": "Four held-out normalized initial-gap blocks; depth-3 decision tree vs majority",
            "pairs": pairs}


def diagnose():
    started = time.perf_counter()
    groups, archives = {}, []
    for kind in DOMAINS:
        history, development = load_construction(kind)
        groups[kind] = pair_diagnosis(history)
        archives.extend((*history, *development))
    banks, missing = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for archive, result in zip(archives, pool.map(audit_bank, archives)):
            banks.append(result)
            print("diagnose", len(banks), "/", len(archives), archive.source_id,
                  result["scenario"], flush=True)
    # An exhaustive audit is required before any selector receives a task.
    development = [b for b in banks if b["role"] == "development" and b["scenario"] in SCENARIOS]
    if len(development) != 8 or any(b["candidates"] != DEFAULT.candidate_count for b in development):
        raise ValueError("The pilot requires all eight complete 512-point development banks")
    report = {"status": "passed", "historical_scenarios": groups, "banks": banks,
              "tasks": development_tasks(), "initialization": "Seeded permutation, first eight; labels unused",
              "settings": asdict(DEFAULT), "supplement_limit": 2048,
              "new_physical_calls": 0, "missing": missing,
              "selection_leakage": "All pool labels and diagnostic partitions remain evaluator-only",
              "wall_s": time.perf_counter() - started}
    destination = OUTPUT / "diagnosis"
    write_json(destination / "report.json", report)
    lines = ["# 阶段 A：历史与开发数据诊断", "", "全部原始执行、档案标签和轨迹碰撞一致。新增物理调用：0。", "",
             "局部结构以初始间距的四个留出区块检查；该诊断不进入选例输入，也不证明迁移收益。", "",
             "| 场景 | 全安全 | 全碰撞 | 存在分歧 | 历史响应模式数 |", "|---|---:|---:|---:|---:|"]
    for kind, data in groups.items():
        lines.append(f"| {kind} | {data['all_safe']} | {data['all_failure']} | {data['disagreement_points']} | {data['response_patterns']} |")
    lines.extend(["", "首轮固定为四个开发对象、切入与切出、五个选例种子；各任务512候选、128查询、8次共同初始化。",
                  "完整系统对计数、空间分块结果、事件响应和碰撞签名见同目录 report.json。", ""])
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
