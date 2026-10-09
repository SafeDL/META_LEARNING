"""Paired support-policy diagnostics on sealed models and validation profiles."""
from collections import defaultdict

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.prediction import behavior_grid
from research.risk_feedback_meta_testing.config import (
    BUDGET, COHORT, MODELS, QUERY_COUNT, RESULTS, SEEDS, SUPPORT_COUNTS,
)
from research.risk_feedback_meta_testing.evaluate import verify_or_lock
from research.risk_feedback_meta_testing.model import (
    FailureDecoder, FrozenResponseBackbone, collision_probabilities,
)
from research.risk_feedback_meta_testing.train import load_banks, meta_inputs


@torch.no_grad()
def measure(decoder, inputs, labels, support_count):
    collision, safe = decoder.log_probabilities(*inputs)
    probability = collision_probabilities(collision, safe)
    count = min(BUDGET - support_count, len(labels))
    selected = torch.topk(collision - safe, count).indices
    found = float(labels[selected].sum())
    total = float(labels.sum())
    return {
        "log_loss": float(-(labels * collision + (1 - labels) * safe).mean()),
        "brier_score": float((probability - labels).square().mean()),
        "top_budget_count": count,
        "top_budget_failures": found,
        "top_budget_precision": found / count,
        "query_subset_collision_count": total,
        "query_subset_recall": found / total if total else 1.0,
    }


def aggregate(records):
    cells = defaultdict(list)
    for row in records:
        key = (row["support_count"], row["policy"], row["mode"])
        cells[key].append(row)
    keys = ("log_loss", "brier_score", "top_budget_failures",
            "top_budget_precision", "query_subset_recall")
    result = []
    for (count, policy, mode), values in sorted(cells.items()):
        row = {"support_count": count, "policy": policy, "mode": mode,
               "contexts": len(values)}
        row.update({key: float(np.mean([value[key] for value in values]))
                    for key in keys})
        row["by_controller"] = {
            controller: {
                key: float(np.mean([value[key] for value in values
                                    if value["controller"] == controller]))
                for key in keys
            }
            for controller in ("IDM", "FVDM")
        }
        result.append(row)
    return result


def report(summary):
    rows = ["# 锁定模型的风险上下文验证诊断", "",
            "仅使用已有验证配置；固定未查询集合，均匀与历史策略支持分别嵌套。",
            "排名指标针对 256 个查询场景，不是完整池策略性能。", "",
            "| 支持数 | 查询方式 | 读出 | Log loss | Brier | 剩余预算候选命中数 |",
            "|---:|---|---|---:|---:|---:|"]
    for row in summary["aggregate"]:
        rows.append(f"| {row['support_count']} | {row['policy']} | "
                    f"{row['mode']} | {row['log_loss']:.6f} | "
                    f"{row['brier_score']:.6f} | {row['top_budget_failures']:.3f} |")
    rows += ["", "配置／种子／场景族的记录及风险披露索引见同名 JSON。",
             "不同支持规模下的候选数是 200−支持数，命中数不作跨规模直接比较。", ""]
    (RESULTS / "validation_context_diagnostic.md").write_text(
        "\n".join(rows), encoding="utf-8")


def main():
    torch.set_num_threads(1)
    verify_or_lock()
    protocol = read_json(RESULTS / "protocol.json")
    names = protocol["split"]["validation"]
    profiles = {row["name"]: row for row in protocol["profiles"]}
    banks = load_banks(names)
    grid = torch.as_tensor(behavior_grid(), device="cpu")
    discrepancy = read_json(COHORT / "protocol.json")["candidate"][
        "risk_discrepancy"]
    output = RESULTS / "validation_context_diagnostic.json"
    if output.exists():
        raise FileExistsError("Existing diagnostics must be reviewed before rerunning")
    records, contexts = [], []
    for seed in SEEDS:
        states = torch.load(MODELS / f"predictor_{seed}.pt",
                            map_location="cpu", weights_only=True)
        backbone = FrozenResponseBackbone(states)
        decoders = {}
        for mode in ("supervised", "meta"):
            saved = torch.load(RESULTS / "models" / f"decoder_{mode}_{seed}.pt",
                               map_location="cpu", weights_only=True)
            decoder = FailureDecoder(saved["weight"], saved["bias"],
                                     saved["center"], saved["scale"])
            decoder.load_state_dict(saved)
            decoders[mode] = decoder.eval()
        for position, name in enumerate(names):
            replicate = position % 2
            bank = banks[name, replicate]
            rng = np.random.default_rng(20261029 + seed * 100 + position)
            supports = {
                "uniform": rng.permutation(len(bank["x"]))[:max(SUPPORT_COUNTS)],
                "historical": bank["frozen_order"][:max(SUPPORT_COUNTS)],
            }
            excluded = np.unique(np.concatenate(list(supports.values())))
            remaining = np.setdiff1d(np.arange(len(bank["x"])), excluded)
            query = rng.choice(remaining, QUERY_COUNT, replace=False)
            labels = torch.as_tensor(bank["collision"][query], dtype=torch.float64)
            zero_inputs = None
            for count in SUPPORT_COUNTS:
                for policy, maximum_support in supports.items():
                    support = maximum_support[:count]
                    assert not np.intersect1d(support, query).size
                    task = name, replicate, support, query
                    if count == 0 and zero_inputs is not None:
                        inputs = zero_inputs
                    else:
                        inputs = meta_inputs(task, banks, backbone, grid,
                                             discrepancy[str(seed)])
                        if count == 0:
                            zero_inputs = inputs
                    context_id = len(contexts)
                    contexts.append({
                        "profile": name, "replicate": replicate, "seed": seed,
                        "support_count": count, "policy": policy,
                        "support_indices": support.tolist(),
                        "query_indices": query.tolist(),
                    })
                    for mode, decoder in decoders.items():
                        full = measure(decoder, inputs, labels, count)
                        family = inputs[1]
                        family_metrics = {}
                        for value, template in ((0, "cutin"), (1, "lead_brake")):
                            mask = family == value
                            sliced = (inputs[0][:, mask], family[mask], inputs[2],
                                      inputs[3][:, mask], inputs[4][:, mask])
                            family_metrics[template] = measure(
                                decoder, sliced, labels[mask], count)
                        records.append({
                            "context_id": context_id, "profile": name,
                            "controller": profiles[name]["controller"],
                            "replicate": replicate, "seed": seed,
                            "support_count": count, "policy": policy,
                            "mode": mode, **full, "by_family": family_metrics,
                        })
            print("VALIDATION CONTEXT", seed, name, len(contexts), flush=True)
    verify_or_lock()
    summary = {
        "role": "paired validation diagnosis; no model changes or blind confirmation",
        "device": "CPU, one thread; no GPU timing competition",
        "profiles": names, "seeds": list(SEEDS),
        "context_count": len(contexts), "expected_context_count": 720,
        "prediction_records": len(records), "new_physical_measurements": 0,
        "query_set_size": QUERY_COUNT, "same_query_set_across_supports": True,
        "models_and_source_seal_verified": True,
        "contexts": contexts, "records": records, "aggregate": aggregate(records),
    }
    assert len(contexts) == 720 and len(records) == 1440
    write_json(output, summary)
    report(summary)
    print("VALIDATION CONTEXTS COMPLETE", len(contexts), len(records), flush=True)


if __name__ == "__main__":
    main()
