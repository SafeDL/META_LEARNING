"""Audit saved E05 records and report the registered paired comparisons."""

import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean

import numpy as np

from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.training import load_model

from experiments.budget.audit import verify_configuration
from experiments.budget.selectors import greedy_index
from experiments.collect_transfer_records import ApprovedCase
from experiments.history import completed_archives, model_version
from experiments.initialization import initial_indices
from experiments.record_audit import BankTruth, verify_metrics
from experiments.resume import paid_feedback_hash, trajectory_hash
from experiments.accumulation.run import BUDGET, INITIAL, LIMITS, OUTPUT, ROOT, RUN_SCENARIOS, SEED, SETTINGS, SUTS, TRAINING_CONFIG, declaration, load_context, location, sequence, sequences, sha
from experiments.storage import read_json, write_json
from experiments.transfer_execution import _paths, approval_path, canonical
from experiments.transfer_prediction_audit import verify_weight_updates
from experiments.accumulation.sequence import REGIMES, execute_sequence

SCENE_NAMES = dict(zip(RUN_SCENARIOS, ("跟车制动", "受限切入", "切出露障碍", "侧向侵入")))
ADS_NAMES = {
    f"target_{name}": label for name, label in (
        ("idm_mobil", "IDM/MOBIL"), ("linear_mpc", "Linear MPC"), ("uct", "UCT"),
        ("ppo", "PPO"), ("nonlinear_trajectory", "Nonlinear trajectory"),
        ("fvdm_lane", "FVDM"), ("dqn", "修订 DQN"), ("beam", "Beam"),
    )
}
REPORTS = OUTPUT / "reports"


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def write_table(name, rows):
    path = REPORTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def check_result(path, bank, truth, checked):
    if path in checked:
        return checked[path]
    saved = read_json(path)
    result, ledger, spec = saved.get("result"), saved["query_ledger"], saved["specification"]
    metrics = None
    if result is not None:
        indices = [r["index"] for r in ledger]
        verify_configuration("full", result)
        require(
            len(ledger) == BUDGET and len(set(indices)) == BUDGET
            and indices == result["selected_indices"]
            and indices[:INITIAL] == initial_indices(len(bank.candidates), INITIAL, SEED)
            and saved["evaluation"]["candidate_collisions"] == truth.labels.tolist(),
            f"Candidate labels, ordering, or budget changed: {path}",
        )
        paid = read_json(path.parent / "paid_feedback.json.gz")
        require(
            paid["status"] == "complete" and paid["query_ledger"] == ledger
            and paid["paid_feedback_hash"] == paid_feedback_hash(ledger)
            and paid["specification"]["task"] == spec,
            f"Paid receipt changed: {path}",
        )
        for call, (row, query) in enumerate(zip(ledger, result["queries"], strict=True), 1):
            physical_feedback = truth.paid_feedback(row["index"], native=True)
            require(
                row["status"] == "complete" and row["call"] == call
                and query["query"] == call and query["index"] == row["index"]
                and query["conditioning_queries"] == call - 1
                and row["feedback"] == query["feedback"] == physical_feedback
                and truth.hashes[row["index"]] == paid["trajectory_hashes"][call - 1],
                f"Paid response, feedback prefix or trajectory changed: {path} query {call}",
            )
        source = spec["source_snapshot"]
        source_key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        source_root = next(p for p in path.parents if (p / "protocol.json").exists())
        with np.load(source_root / "sources" / f"{source_key}.npz", allow_pickle=False) as prior:
            require(json.loads(str(prior["specification"])) == source
                    and np.array_equal(prior["weights"], np.full(len(source["archives"]), 1 / len(source["archives"]))),
                    "Saved source posterior belongs to a different task")
            verify_weight_updates(result, ledger, prior)
        snapshots = result["prediction_snapshots"]
        require([r["paid_queries"] for r in snapshots] == [0, 20, 50, 100, 200],
                "Unexpected budget checkpoints")
        for snapshot in snapshots:
            k = snapshot["paid_queries"]
            probability = np.asarray(snapshot["probabilities"])
            require(snapshot["observed_indices"] == indices[:k]
                    and probability.shape == (len(bank.candidates),)
                    and np.isfinite(probability).all()
                    and ((0 <= probability) & (probability <= 1)).all(),
                    "Prediction checkpoint does not match its paid prefix")
            if INITIAL <= k < BUDGET:
                remaining = set(range(len(bank.candidates))) - set(indices[:k])
                require(greedy_index(bank.candidates, probability, remaining) == indices[k],
                        "Saved checkpoint did not choose the highest available collision probability")
        verify_metrics(bank.candidates, truth.labels.tolist(), indices, result["queries"],
                       ledger, saved["metrics"], INITIAL, domain=bank.domain)
        metrics = saved["metrics"]
    else:
        require(len(ledger) <= BUDGET, f"Failed prefix exceeded the task budget: {path}")
        for row in ledger:
            i = row["index"]
            require(row["scenario_id"] == bank.candidates[i].scenario_id,
                    f"Failed prefix references a foreign coordinate: {path}")
            if row["status"] == "complete":
                require(row["feedback"] == truth.paid_feedback(i, native=True),
                        f"Failed prefix response differs from its native record: {path}")
    checked[path] = ledger, metrics
    return checked[path]


class BlockTruth(BankTruth):
    """Cache native trajectory identities across each target's repeated paths."""

    def __init__(self, bank):
        super().__init__(bank)
        self.hashes = [trajectory_hash(bank(scene).trajectory) for scene in bank.candidates]


def check_archive(root, branch, task, row, inputs, ledger, bank, prior_archives):
    archives = [] if branch["regime"] == "fixed_history" else prior_archives
    require(inputs["available_completed_archives"] == [
        {"path": p.relative_to(root).as_posix(), "sha256": sha(p)} for p in archives
    ], "Task accessed an archive outside its own completed prefix")
    spec = inputs["specification"]
    snapshot = spec["task_snapshot"]
    case = ApprovedCase("budget", task["sut_id"], branch["scenario"], SEED, "full", spec)
    require(read_json(approval_path(root, case)) == {
        "key": list(case.key), "scope": "formal", "specification": spec
    }, "Task inputs differ from their formal approval")
    require(inputs["task"] == task and inputs["model_versions"] == row["model_versions"]
            and [spec["target_model_version"], spec["encoder_version"]] == row["model_versions"],
            "Task identity or shared-model identity changed")
    require(
        spec["candidates"] == canonical([asdict(s) for s in bank.candidates])
        and snapshot["bank_snapshot"] == bank.snapshot_fingerprint
        and snapshot["sut_id"] == task["sut_id"] and spec["bank_contract"] == bank.manifest["contract"],
        "Task specification changed its registered 500-point block",
    )
    path = Path(row["result_path"])
    local = _paths(root, case)[0]
    if row["reused_prefix"]:
        receipt = read_json(local.parent / "prefix_reuse.json")
        require(receipt == {"origin": str(path), "sha256": sha(path), "paid_queries": len(ledger)},
                "Reused result differs from the sealed common prefix")
    else:
        require(path.parent == local.parent, "Independent result belongs to a different branch")
    completed = [r for r in ledger if r["status"] == "complete"]
    require(len(completed) == row["successful_paid_feedback"], "Archive paid count changed")
    if not completed:
        require(row["archive"] is None, "An empty paid prefix entered historical sources")
        return len(archives), 0
    archive_path = root / row["archive"]
    archive = read_json(archive_path)
    require(
        archive["attempts"] == ledger and archive["branch_id"] == branch["branch_id"]
        and archive["completed_order"] == task["ordinal"] and archive["sut_id"] == task["sut_id"]
        and len(archive["executions"]) == len(completed),
        "Branch archive does not match its own result position",
    )
    for record, entry in zip(completed, archive["executions"], strict=True):
        i = record["index"]
        require(
            entry["query"] == record["call"]
            and entry["scenario"] == canonical(asdict(bank.candidates[i]))
            and entry["feedback"] == record["feedback"]
            and entry["execution_ref"] == bank.execution_path(i).relative_to(ROOT).as_posix(),
            "Shared archive response or physical index does not match the paid query",
        )
    past_paid = sum(len(read_json(p)["executions"]) for p in archives)
    return len(archives), past_paid


def audit_scene(kind, *, output=OUTPUT, scene_context=load_context, sequence_jobs=sequences,
                sequence_builder=sequence, sequence_location=location):
    context = scene_context(kind)
    initial_models = [model_version(m) for m in context["models"]]
    initial_history = context["history"]
    blocks = context["banks"]
    truths = {target: BlockTruth(bank) for target, bank in blocks.items()}
    checked, ledgers, rows, updates = {}, {}, [], []
    for scene, group, order in sequence_jobs():
        if scene != kind:
            continue
        for regime in REGIMES:
            root = sequence_location(kind, group, order, regime)
            actual = sequence_builder(kind, group, order, regime)
            saved = read_json(root / "progress.json")
            require(saved["status"] == "complete" and saved["sequence"] == actual
                    and len(saved["completed"]) == len(actual["tasks"])
                    and saved["new_physical_executions"] == 0, f"Incomplete branch: {root}")
            by_target = {r["sut_id"]: r for r in saved["completed"]}
            past = []
            for task, row in zip(actual["tasks"], saved["completed"], strict=True):
                target, ordinal = task["sut_id"], task["ordinal"]
                bank, truth = blocks[target], truths[target]
                require(row["ordinal"] == ordinal and row["sut_id"] == target,
                        f"Wrong task position: {root}")
                own_path = root / "task_inputs" / f"{ordinal:02d}_{target}.json"
                inputs = read_json(own_path)
                allowed = initial_history if regime == "fixed_history" else [
                    *initial_history,
                    *completed_archives(past, before_order=ordinal, branch_id=actual["branch_id"]),
                ]
                source = inputs["specification"]["source_snapshot"]
                view_ids = {
                    **context["view_record_ids"],
                    **{a.source_id: a.provenance["record_ids"] for a in allowed[8:]},
                }
                require(row["actual_source_count"] == len(allowed)
                        and row["source_ids"] == inputs["actual_source_ids"] == [a.source_id for a in allowed]
                        and source["archives"] == archive_fingerprints(allowed)
                        and source["view_record_ids"] == view_ids,
                        "Source construction includes a future, foreign or missing archive")
                path = Path(row["result_path"])
                ledger, metrics = check_result(path, bank, truth, checked)
                expected_reuse = (
                    (regime == "fixed_history" and order == "reverse")
                    or (regime != "fixed_history" and ordinal == 1)
                    or (regime == "update_shared" and ordinal == 2)
                )
                require(row["reused_prefix"] == expected_reuse,
                        "A result position did not use its registered path/reuse rule")
                require((metrics is not None) == (row["status"] == "complete"),
                        "Terminal task status differs from its result")
                archive_count, past_paid = check_archive(root, actual, task, row, inputs, ledger, bank, past)
                require(len(allowed) == 8 + archive_count, "Unexpected admitted archive count")
                row["past_paid"] = past_paid
                row["metrics"] = metrics
                row["failure"] = read_json(path).get("error") if metrics is None else None
                row["pool_collisions"] = int(truth.labels.sum())
                if row["archive"] is not None:
                    past.append(root / row["archive"])
                ledgers[group, order, regime, target] = ledger
                if regime == "update_shared" and ordinal <= 2:
                    require(ledger == ledgers[group, order, "append_sources", target],
                            "Update and append regimes saw different paid feedback before task 3")
                if regime != "fixed_history" and ordinal == 1:
                    require(ledger == ledgers[group, order, "fixed_history", target],
                            "The regimes did not share their first task")
                if ordinal == 2 and regime == "update_shared":
                    past_before_update = list(past)
            published_models = initial_models
            published_at = accepted = None
            if regime == "update_shared":
                event, = saved["updates"]
                model_dir = root / "models/after_02_tasks"
                event_file = model_dir / "sequence_event.json"
                require(event == read_json(event_file) and event["after_task"] == 2,
                        "Update did not occur directly after the two common paid tasks")
                publication = event["publication"]
                accepted = publication["accepted"]
                published_models = event["model_versions"]
                if "specification" in publication:
                    spec = publication["specification"]
                    new_history = completed_archives(past_before_update, before_order=3,
                                                     branch_id=actual["branch_id"])
                    require(
                        spec["history_fingerprints"] == archive_fingerprints([*initial_history, *new_history])
                        and spec["new_source_ids"] == [a.source_id for a in new_history if len(a.scenarios) >= 2]
                        and spec["config"] == asdict(TRAINING_CONFIG)
                        and spec["parent_version"] == ":".join(initial_models),
                        "Update training accessed feedback beyond the first two tasks",
                    )
                if accepted:
                    require(
                        all(publication["training"][name]["final_step"] == TRAINING_CONFIG.steps
                            for name in ("target", "encoder"))
                        and published_models == [
                            model_version(load_model(model_dir / name, kernel_type=type(context["models"][0]))[0])
                            for name in ("target_kernel.pt", "historical_encoder.pt")
                        ], "Published update weights or step counts changed",
                    )
                    published_at = datetime.fromtimestamp(
                        event_file.stat().st_mtime, timezone.utc).isoformat()
                else:
                    require(published_models == initial_models, "A failed update changed the model")
                updates.append({
                    "scene": kind, "group": group, "order": order,
                    "accepted": accepted, "reason": publication["reason"], "published_utc": published_at,
                    "target_training_s": publication.get("training", {}).get("target", {}).get("training_s"),
                    "encoder_training_s": publication.get("training", {}).get("encoder", {}).get("training_s"),
                    "event": event_file.relative_to(output).as_posix(),
                })
            else:
                require(not saved["updates"], "An unregistered model update occurred")
            for task in actual["tasks"]:
                ordinal, target = task["ordinal"], task["sut_id"]
                row = by_target[target]
                require(row["model_versions"] == (initial_models if ordinal <= 2 else published_models),
                        "The updated model was applied at the wrong position")
                path = Path(row["result_path"])
                metrics = row["metrics"]
                rows.append({
                    "scene": kind, "group": group, "order": order, "ordinal": ordinal,
                    "target": target, "regime": regime, "status": row["status"],
                    "discovery_auc": metrics["discovery_auc"] if metrics else None,
                    "c200": metrics["unique_collisions"] if metrics else None,
                    "bank_collisions": row["pool_collisions"],
                    "successful_paid_feedback": row["successful_paid_feedback"],
                    "source_count": row["actual_source_count"], "past_paid_records": row["past_paid"],
                    "reuse": row["reused_prefix"],
                    "model_update_accepted": accepted if ordinal > 2 else None,
                    "model_published_utc": published_at if ordinal > 2 else None,
                    "model_target": row["model_versions"][0], "model_encoder": row["model_versions"][1],
                    "result": path.relative_to(output).as_posix(), "result_sha256": sha(path),
                    "failure": row["failure"],
                })
    return rows, updates, sum(len(ledger) for ledger, _ in checked.values())


def differences(rows):
    by_position, comparisons = {}, []
    for row in rows:
        key = tuple(row[k] for k in ("scene", "group", "order", "ordinal", "target"))
        by_position.setdefault(key, {})[row["regime"]] = row
    for (scene, group, order, ordinal, target), slot in by_position.items():
        if ordinal < 2:
            continue
        fixed, append, update = [slot[r] for r in REGIMES]

        def delta(a, b, field):
            if a[field] is None or b[field] is None:
                return None
            if field == "discovery_auc":
                return (round(a[field] * BUDGET) - round(b[field] * BUDGET)) / BUDGET
            return a[field] - b[field]

        comparisons.append({
            "scene": scene, "group": group, "order": order, "ordinal": ordinal, "target": target,
            "append_fixed_auc": delta(append, fixed, "discovery_auc"),
            "update_append_auc": delta(update, append, "discovery_auc"),
            "update_fixed_auc": delta(update, fixed, "discovery_auc"),
            "append_fixed_c200": delta(append, fixed, "c200"),
            "update_append_c200": delta(update, append, "c200"),
            "update_fixed_c200": delta(update, fixed, "c200"),
            "model_update_accepted": update["model_update_accepted"],
        })
    return comparisons


def describe(values):
    usable = [v for v in values if v is not None]
    return {
        "total": len(values), "paired": len(usable),
        "mean": mean(usable) if usable else None,
        "positive": sum(v > 1e-12 for v in usable), "tie": sum(abs(v) <= 1e-12 for v in usable),
        "negative": sum(v < -1e-12 for v in usable),
    }


def aggregate(comparisons):
    append_key = "append_fixed_auc"
    sequence_rows = []
    for kind, group, order in sequences():
        slots = [r for r in comparisons if (r["scene"], r["group"], r["order"]) == (kind, group, order)]
        values = [r[append_key] for r in slots]
        sequence_rows.append({
            "scene": kind, "group": group, "order": order, "task2": values[0],
            "task3": values[1], "task4": values[2],
            "mean_append_fixed_auc": mean(values) if all(v is not None for v in values) else None,
            "task3_update_append_auc": slots[1]["update_append_auc"],
            "task4_update_append_auc": slots[2]["update_append_auc"],
            "mean_update_fixed_auc": mean(r["update_fixed_auc"] for r in slots)
            if all(r["update_fixed_auc"] is not None for r in slots) else None,
        })
    per_target = []
    for kind in RUN_SCENARIOS:
        for target in SUTS:
            subset = [r for r in comparisons if r["scene"] == kind and r["target"] == target]
            entry = {"scene": kind, "target": target, "paired_positions": len(subset)}
            for field in ("append_fixed_auc", "append_fixed_c200", "update_fixed_auc", "update_fixed_c200"):
                values = [r[field] for r in subset]
                entry[field] = mean(values) if all(v is not None for v in values) else None
            per_target.append(entry)

    def balanced(rows, field):
        values = [r[field] for r in rows]
        return mean(values) if all(v is not None for v in values) else None

    strata = {}
    for kind in RUN_SCENARIOS:
        slots = [r for r in comparisons if r["scene"] == kind]
        cells = [r for r in per_target if r["scene"] == kind]
        strata[kind] = {
            "append": describe([r[append_key] for r in slots]),
            "append_balanced_auc": balanced(cells, append_key),
            "append_balanced_c200": balanced(cells, "append_fixed_c200"),
            "update_balanced_auc": balanced(cells, "update_fixed_auc"),
            "task3": describe([r["update_append_auc"] for r in slots if r["ordinal"] == 3]),
            "task4": describe([r["update_append_auc"] for r in slots if r["ordinal"] == 4]),
        }
    summary = {
        "append_overall_balanced_auc": balanced(per_target, append_key),
        "append_overall_balanced_c200": balanced(per_target, "append_fixed_c200"),
        "update_overall_balanced_auc": balanced(per_target, "update_fixed_auc"),
        "update_overall_balanced_c200": balanced(per_target, "update_fixed_c200"),
        "append_all_positions": describe([r[append_key] for r in comparisons]),
        "append_sequences": describe([r["mean_append_fixed_auc"] for r in sequence_rows]),
        "task3_update_append_auc": describe([r["update_append_auc"] for r in comparisons if r["ordinal"] == 3]),
        "task3_update_append_c200": describe([r["update_append_c200"] for r in comparisons if r["ordinal"] == 3]),
        "task4_update_append_auc": describe([r["update_append_auc"] for r in comparisons if r["ordinal"] == 4]),
        "task4_update_append_c200": describe([r["update_append_c200"] for r in comparisons if r["ordinal"] == 4]),
        "scenes": strata,
    }
    return sequence_rows, per_target, summary


def materialize_reverse_references(kind):
    """Record reverse-order fixed positions and their own archives with zero queries."""
    for scene, group, order in sequences():
        if scene != kind or order != "reverse":
            continue
        reg = "fixed_history"
        reverse = sequence(kind, group, order, reg)
        root = location(kind, group, order, reg)
        if (root / "progress.json").exists() and read_json(root / "progress.json")["status"] == "complete":
            continue
        origin = location(kind, group, "forward", reg)
        execute_sequence(
            OUTPUT, reverse, load_context(kind), settings=SETTINGS, limits=LIMITS,
            scope="formal", fixed_reference="execute",
            reuse_from={task["ordinal"]: origin for task in reverse["tasks"]},
        )


def write_interpretation(result, pairs, figure):
    summary = result["summary"]
    archive = summary["append_all_positions"]
    archive_end = describe([row["append_fixed_c200"] for row in pairs])
    direct = summary["task3_update_append_auc"]
    downstream = summary["task4_update_append_auc"]

    def signed(value, digits=2):
        return "缺失" if value is None else f"{value:+.{digits}f}"

    def counts(values):
        return f'{values["positive"]} / {values["tie"]} / {values["negative"]}'

    group_label = {"group_a": "A 组", "group_b": "B 组"}
    order_label = {"forward": "正序", "reverse": "反序"}
    lines = [
        "# E05 任务间知识积累",
        "",
        "每次测试新 ADS 时，三种做法使用同一个 500 点候选库和相同的前 20 条反馈，总预算均为 200 次。"
        "固定历史一直使用最初八个来源。仅追加来源把刚查过的碰撞与安全距离余量（RSS 裕度）归档，"
        "供下一次测试使用。更新方案另外用这些反馈在第 2 个任务结束后更新一次共享模型。",
        "",
        f'本次共核对 {result["positions"]} 个结果位置、{result["distinct_paths"]} 条不同选例轨迹、'
        f'{result["logical_queries"]:,} 次逻辑查询。新增物理仿真 {result["new_physical_executions"]} 次，'
        f'{result["updates"]} 次登记更新中有 {result["published"]} 次发布了模型。'
        f'{result["failed_positions"]} 个结果位置记录为失败。',
        "",
        "发现 AUC 是第 1 至 200 次查询后累计碰撞数的平均值。AUC 增加 1，表示在这些查询时点，"
        "手里平均多掌握 1 个已发现的碰撞用例。C(200) 是用完预算时发现的碰撞数。"
        "两个量分别反映发现过程和最后收获。",
        "",
        "## 归档以前任务的反馈",
        "",
        "在四场景、八个 ADS 等权的比较中，仅追加来源相对固定历史，"
        f'发现 AUC 差为 **{signed(summary["append_overall_balanced_auc"], 3)}**，'
        f'C(200) 差为 **{signed(summary["append_overall_balanced_c200"], 3)}**。'
        f'第 2–4 任务的 {archive["paired"]} 个有效配对中，提高 / 持平 / 下降分别为 {counts(archive)}。'
        f'按序列先平均第 2–4 项后，完整序列的方向计数为 {counts(summary["append_sequences"])}。',
        "",
        f'用完预算时，C(200) 的提高 / 持平 / 下降数为 {counts(archive_end)}。'
        "归档对早期发现顺序的影响，体现在 AUC 中。",
        "",
        "| 场景 | 仅追加 − 固定 AUC | 仅追加 − 固定 C(200) | 位置提高 / 持平 / 下降 | 第 3 项更新增量 AUC | 第 4 项制度差 AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scene, label in SCENE_NAMES.items():
        row = summary["scenes"][scene]
        lines.append(
            f'| {label} | {signed(row["append_balanced_auc"])} | {signed(row["append_balanced_c200"])} | '
            f'{counts(row["append"])} | {signed(row["task3"]["mean"])} | {signed(row["task4"]["mean"])} |'
        )
    for direction, select in (("增益最大", max), ("下降最大", min)):
        sign = 1 if direction == "增益最大" else -1
        ranked = [p for p in pairs if p["append_fixed_auc"] is not None and sign * p["append_fixed_auc"] > 0]
        if ranked:
            row = select(ranked, key=lambda p: p["append_fixed_auc"])
            description = (
                f"{SCENE_NAMES[row['scene']]}，{group_label[row['group']]}{order_label[row['order']]}，"
                f"第 {row['ordinal']} 项 {ADS_NAMES[row['target']]}"
            )
            lines.extend(["", f'{direction}的配对发生在{description}。AUC 差为 '
                          f'{signed(row["append_fixed_auc"], 3)}，C(200) 差为 '
                          f'{signed(row["append_fixed_c200"], 0)}。'])
    lines.extend([
        "",
        "## 更新共享模型",
        "",
        "第 3 项开始时，更新版与仅追加版可见的前两项付费记录相同。"
        f'模型更新的直接 AUC 增量均值为 **{signed(direct["mean"])}**，'
        f'C(200) 增量均值为 **{signed(summary["task3_update_append_c200"]["mean"])}**。'
        f'{direct["paired"]} 个配对的提高 / 持平 / 下降分别为 {counts(direct)}。',
        "",
    ])
    direct_pairs = [row for row in pairs if row["ordinal"] == 3 and row["update_append_auc"] is not None]
    if direct_pairs:
        worst = min(direct_pairs, key=lambda row: row["update_append_auc"])
        if worst["update_append_auc"] < 0:
            description = (f"{SCENE_NAMES[worst['scene']]}，{group_label[worst['group']]}"
                           f"{order_label[worst['order']]}的 {ADS_NAMES[worst['target']]}")
            remaining = [row["update_append_auc"] for row in direct_pairs if row is not worst]
            lines.append(f'最大的直接 AUC 下降出现在{description}，差为 '
                         f'{signed(worst["update_append_auc"], 3)}，'
                         f'C(200) 差为 {signed(worst["update_append_c200"], 0)}。')
            if remaining:
                lines.append(f'完整均值计入该对照。其余 {len(remaining)} 个配对的 AUC 均值差为 '
                             f'{signed(mean(remaining), 3)}。')
            lines.append("")
    lines.extend([
        f'第 4 项的 AUC 差均值为 **{signed(downstream["mean"])}**，'
        f'提高 / 持平 / 下降分别为 {counts(downstream)}。第 3 项的选例已经可能产生不同反馈，'
        "因此第 4 项差值涵盖共享模型更新及随后归档数据变化的共同作用。"
        f'完整方法相对固定历史在第 2–4 项的等权 AUC 差为 '
        f'**{signed(summary["update_overall_balanced_auc"])}**，'
        f'C(200) 差为 **{signed(summary["update_overall_balanced_c200"])}**。',
        "",
        "## 论文中的证据位置",
        "",
    ])
    if archive["positive"] and archive["negative"]:
        lines.append("追加付费档案在登记配对中既有收益也有损失，其价值随场景、ADS 和任务顺序而变。")
    elif archive["positive"]:
        lines.append("追加付费档案在登记条件下带来一部分更早的碰撞发现。")
    elif archive["negative"]:
        lines.append("追加付费档案在登记条件下出现了发现效率退化。")
    elif archive["paired"]:
        lines.append("追加付费档案后的发现 AUC 在可比较的登记位置保持相同。")
    else:
        lines.append("追加付费档案的主比较缺少可用配对。")
    if direct["positive"] and direct["negative"]:
        lines.append("同一批档案上的模型更新也出现了方向相反的直接效果。")
    if direct["mean"] is not None and direct["mean"] < 0:
        lines.append("论文可以报告新反馈怎样进入下一次测试，并把这套固定更新配置下的效率下降列为方法的适用边界。")
    lines.extend([
        "E05 补上了测试所得反馈能否回到方法中、影响下一批 ADS 测试的实证环节。"
        "E01–E03 分别提供风险预测、单任务预算发现及权重更新的证据。"
        "本实验的适用范围是四种交通场景、八个 ADS、两组固定的四任务正反序列和一次共享模型更新。"
        "同组顺序复用 ADS 与坐标，按完整配对作描述性比较。",
        "",
        "## 图注与复算材料",
        "",
        f"[四场景知识积累配对图](../../../{figure.relative_to(ROOT).as_posix()})",
        "",
        "图 a 显示仅追加来源相对固定历史的 AUC 差值。点与横线显示每条序列第 2–4 项配对值及其最小至最大范围，"
        "空心菱形是序列均值，实心菱形是场景内八 ADS 等权均值。图 b 显示更新相对仅追加的 AUC 差值。"
        "圆形对应共享模型更新后的第 3 项，方形对应第 4 项的制度整体效果。实心点是场景内四条序列的算术均值。"
        "A、B 表示目标组，右箭头为正序，左箭头为反序。两个面板使用相同的 AUC 单位与各自的横轴范围，"
        "零线表示两版发现 AUC 相同。",
        "",
        "[192 个结果位置](reports/positions.csv) · [48 个配对结果](reports/paired_outcomes.csv) · "
        "[16 条序列](reports/sequences.csv) · [32 个场景–ADS 汇总](reports/scene_targets.csv) · "
        "[16 次模型更新](reports/model_updates.csv) · [复算凭据](reports/completion.json) · "
        "[图形质量核验](reports/figure_quality/paired_accumulation_effects.json)",
    ])
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report():
    require(read_json(OUTPUT / "protocol.json") == canonical(declaration()),
            "Official E05 experiment registration changed before the audit")
    rows, updates, paid = [], [], 0
    for kind in RUN_SCENARIOS:
        materialize_reverse_references(kind)
        scene_rows, scene_updates, scene_paid = audit_scene(kind)
        rows.extend(scene_rows)
        updates.extend(scene_updates)
        paid += scene_paid
        print({"audited_scene": kind, "positions": len(scene_rows), "paid_queries": scene_paid}, flush=True)
    paths = {r["result"] for r in rows}
    require((len(rows), len(paths), len(updates)) == (192, 112, 16),
            "E05 registered matrix does not close")
    pairs = differences(rows)
    sequence_rows, targets, summary = aggregate(pairs)
    require((len(pairs), len(sequence_rows), len(targets)) == (48, 16, 32),
            "Incorrect E05 summary denominators")
    failures = sum(r["status"] == "failed" for r in rows)
    require(paid <= 22400 and (failures > 0 or paid == 22400), "Incorrect paid query total")
    files = [
        write_table("positions.csv", rows), write_table("paired_outcomes.csv", pairs),
        write_table("sequences.csv", sequence_rows), write_table("scene_targets.csv", targets),
        write_table("model_updates.csv", updates),
    ]
    result = {
        "status": "complete" if failures == 0 else "complete_with_task_failures",
        "positions": len(rows), "failed_positions": failures, "distinct_paths": len(paths),
        "logical_queries": paid, "new_physical_executions": 0,
        "updates": len(updates), "published": sum(r["accepted"] for r in updates),
        "files": {p.name: sha(p) for p in files}, "summary": summary,
    }
    write_json(REPORTS / "completion.json", result)
    from experiments.plotting.transfer_sequence import render

    figure = render()
    write_interpretation(result, pairs, figure)
    return result


if __name__ == "__main__":
    report()
