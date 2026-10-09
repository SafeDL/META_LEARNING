"""Compare selected pilot checkpoints with the frozen same-data greedy head."""
from collections import defaultdict

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.prediction import behavior_grid
from research.risk_feedback_meta_testing.evaluate import verify_or_lock
from research.risk_feedback_meta_testing.model import (
    FailureDecoder, FrozenResponseBackbone, collision_probabilities,
)
from research.risk_feedback_meta_testing.posterior import condition_risk
from research.risk_feedback_meta_testing.train import load_banks

from .config import BACKBONES, BUDGET, COHORT, PREVIOUS, RESULTS, SEEDS


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    verify_or_lock()
    protocol = read_json(PREVIOUS / "results" / "protocol.json")
    names = protocol["split"]["validation"]
    banks = load_banks(names)
    grid = torch.as_tensor(behavior_grid(), device="cuda")
    discrepancy = read_json(COHORT / "protocol.json")["candidate"]["risk_discrepancy"]
    records, selected_steps = [], {}
    for seed in SEEDS:
        manifest = read_json(RESULTS / "models" / f"training_{seed}.json")
        selected_steps[str(seed)] = {}
        for mode in ("supervised", "meta"):
            step = min(manifest["validation_curve"], key=lambda row:
                       row[mode]["historical_early_log_loss"])["step"]
            selected_steps[str(seed)][mode] = step
            measurement = read_json(RESULTS / "validation" / f"seed_{seed}_step_{step}.json")
            records += [{**row, "seed": seed, "mode": f"inference_{mode}"}
                        for row in measurement["records"] if row["mode"] == mode]
        backbone = FrozenResponseBackbone(torch.load(
            BACKBONES / f"predictor_{seed}.pt", map_location="cuda", weights_only=True)).cuda()
        saved = torch.load(PREVIOUS / "results" / "models" /
                           f"decoder_supervised_{seed}.pt", map_location="cuda", weights_only=True)
        decoder = FailureDecoder(saved["weight"], saved["bias"], saved["center"], saved["scale"])
        decoder.load_state_dict(saved)
        for position, name in enumerate(names):
            replicate = position % 2
            bank = banks[name, replicate]
            x = torch.as_tensor(bank["x"], dtype=torch.float64, device="cuda")
            frozen, features = backbone.features(x.float(), grid)
            family = x[:, 4].long()
            rng = np.random.default_rng(20261031 + position)
            supports = {"uniform": rng.permutation(len(x))[:150],
                        "historical": bank["frozen_order"][:150]}
            for policy, maximum_support in supports.items():
                for count in (0, 10, 50, 150):
                    support = maximum_support[:count]
                    query = np.setdiff1d(np.arange(len(x)), support)
                    weights, mean, variance = condition_risk(
                        x, frozen.double(), support, bank["risk"][support], discrepancy[str(seed)])
                    collision, safe = decoder.log_probabilities(
                        features[:, query], family[query], weights, mean[:, query], variance[:, query])
                    labels = torch.as_tensor(bank["collision"][query], dtype=torch.float64, device="cuda")
                    probability = collision_probabilities(collision, safe)
                    chosen = torch.topk(collision - safe, BUDGET - count).indices
                    found = float(labels[chosen].sum())
                    support_found = int(bank["collision"][support].sum())
                    total = int(bank["collision"].sum())
                    records.append({
                        "seed": seed, "profile": name, "replicate": replicate,
                        "mode": "frozen_supervised_greedy", "policy": policy,
                        "support_count": count,
                        "log_loss": float(-(labels * collision + (1 - labels) * safe).mean()),
                        "brier_score": float((probability - labels).square().mean()),
                        "static_projected_F200": support_found + found,
                        "static_budget_deficit": min(BUDGET, total) - support_found - found,
                    })
            print("PILOT FROZEN REFERENCE", seed, name, flush=True)
    groups = defaultdict(list)
    for row in records:
        groups[row["mode"], row["policy"], row["support_count"]].append(row)
    metrics = ("log_loss", "brier_score", "static_projected_F200", "static_budget_deficit")
    aggregate = [{"mode": mode, "policy": policy, "support_count": count,
                  **{metric: float(np.mean([row[metric] for row in rows])) for metric in metrics}}
                 for (mode, policy, count), rows in sorted(groups.items())]
    early = {mode: [row for row in records if row["mode"] == mode
                   and row["policy"] == "historical" and row["support_count"] in (10, 50)]
             for mode in ("inference_supervised", "inference_meta", "frozen_supervised_greedy")}
    summary_early = {mode: {metric: float(np.mean([row[metric] for row in rows]))
                           for metric in metrics} for mode, rows in early.items()}
    candidate = summary_early["inference_meta"]
    supported = all(
        candidate["log_loss"] < reference["log_loss"]
        and candidate["static_budget_deficit"] < reference["static_budget_deficit"]
        for mode, reference in summary_early.items() if mode != "inference_meta")
    summary = {"role": "fixed two-seed validation; no independent significance claim",
               "record_count": len(records), "selected_steps": selected_steps,
               "early_historical": summary_early, "pilot_supported": supported,
               "static_continuation_is_not_policy_replay": True,
               "new_physical_measurements": 0, "aggregate": aggregate, "records": records}
    write_json(RESULTS / "pilot_summary.json", summary)
    rows = ["# 风险推断元学习：两种子验证", "",
            f"固定验证晋级：{'支持下一阶段' if supported else '不支持扩大本设计'}。",
            "模型依据既定验证损失选择；完整池静态延续不是实际后续逐次查询。", "",
            "| 模型 | 历史10/50支持后的 Log loss | Brier | 静态预计 F200 | 静态预算缺口 |",
            "|---|---:|---:|---:|---:|"]
    for mode, values in summary_early.items():
        rows.append(f"| {mode} | {values['log_loss']:.6f} | {values['brier_score']:.6f} | "
                    f"{values['static_projected_F200']:.3f} | {values['static_budget_deficit']:.3f} |")
    rows += ["", "两种子、12 个验证配置；普通监督与元模型结构和训练预算相同。",
             "完整支持规模／策略记录见 `pilot_summary.json`。", ""]
    (RESULTS / "pilot_report.md").write_text("\n".join(rows), encoding="utf-8")
    verify_or_lock()
    print("INFERENCE PILOT SUMMARIZED", supported, summary_early, flush=True)


if __name__ == "__main__":
    main()
