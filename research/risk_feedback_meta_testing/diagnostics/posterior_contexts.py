"""Inspect risk posteriors on a fixed subset of existing validation contexts."""
import math
from collections import defaultdict

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.model import behavior_coordinates
from research.behavior_response_testing.prediction import behavior_grid
from research.risk_feedback_meta_testing.config import COHORT, MODELS, RESULTS, SEEDS
from research.risk_feedback_meta_testing.evaluate import verify_or_lock
from research.risk_feedback_meta_testing.model import FrozenResponseBackbone
from research.risk_feedback_meta_testing.posterior import condition_risk


SUPPORT_COUNTS = (10, 50, 150)


def expected_clipped_normal(mean, variance):
    scale = variance.sqrt()
    lower, upper = -mean / scale, (1 - mean) / scale
    density_lower = torch.exp(-0.5 * lower.square()) / math.sqrt(2 * math.pi)
    density_upper = torch.exp(-0.5 * upper.square()) / math.sqrt(2 * math.pi)
    return (mean * (torch.special.ndtr(upper) - torch.special.ndtr(lower))
            + scale * (density_lower - density_upper)
            + torch.special.ndtr(-upper)).clamp(0, 1)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    verify_or_lock()
    contexts = read_json(RESULTS / "validation_context_diagnostic.json")["contexts"]
    chosen = [row for row in contexts if row["support_count"] in SUPPORT_COUNTS]
    assert len(chosen) == 360
    protocol = read_json(RESULTS / "protocol.json")
    profiles = {row["name"]: row for row in protocol["profiles"]}
    discrepancy = read_json(COHORT / "protocol.json")["candidate"]["risk_discrepancy"]
    grid = torch.as_tensor(behavior_grid())
    records, banks = [], {}
    for seed in SEEDS:
        states = torch.load(MODELS / f"predictor_{seed}.pt",
                            map_location="cpu", weights_only=True)
        backbone = FrozenResponseBackbone(states)
        for name in protocol["split"]["validation"]:
            for context in (row for row in chosen
                            if row["seed"] == seed and row["profile"] == name):
                replicate = context["replicate"]
                key = name, replicate
                if key not in banks:
                    with np.load(COHORT / name / f"pool_{replicate}" /
                                 "responses.npz") as bank:
                        banks[key] = {key: bank[key].copy() for key in ("x", "risk")}
                bank = banks[key]
                support = np.asarray(context["support_indices"], dtype=int)
                query = np.asarray(context["query_indices"], dtype=int)
                assert not np.intersect1d(support, query).size
                indices = np.concatenate((support, query))
                x = torch.as_tensor(bank["x"][indices], dtype=torch.float64)
                prediction, _ = backbone.features(x.float(), grid)
                prediction = prediction.double()
                log_weights, mean, variance = condition_risk(
                    x, prediction, np.arange(len(support)), bank["risk"][support],
                    discrepancy[str(seed)])
                weights = log_weights.exp()
                selected = slice(len(support), None)
                family = x[selected, 4].long()
                settings = discrepancy[str(seed)]
                noise = x.new_tensor(settings["noise_variance"])[family]
                prior_variance = x.new_tensor(settings["gp_variance"])[family]
                prior_risk = expected_clipped_normal(
                    prediction[:, selected], prior_variance + noise).mean(0)
                posterior_risk = (weights[:, None] * expected_clipped_normal(
                    prediction[:, selected] + mean[:, selected],
                    variance[:, selected] + noise)).sum(0)
                actual_risk = torch.as_tensor(bank["risk"][query], dtype=torch.float64)
                entropy = float(-(weights * log_weights).sum())
                truth = torch.as_tensor(behavior_coordinates(profiles[name])).double()
                posterior_coordinates = weights @ grid.double()
                true_controller = grid[:, 3] == truth[3]
                records.append({
                    "profile": name, "controller": profiles[name]["controller"],
                    "replicate": replicate, "seed": seed,
                    "policy": context["policy"],
                    "support_count": context["support_count"],
                    "hypothesis_entropy": entropy,
                    "effective_hypotheses": math.exp(entropy),
                    "prior_effective_hypotheses": len(grid),
                    "kl_from_uniform_prior": math.log(len(grid)) - entropy,
                    "true_controller_posterior_mass_for_audit_only":
                        float(weights[true_controller].sum()),
                    "normalized_behavior_mean_error_for_audit_only":
                        float(torch.linalg.vector_norm(posterior_coordinates[:3] - truth[:3])),
                    "query_risk_rmse_prior": float((prior_risk - actual_risk).square().mean().sqrt()),
                    "query_risk_rmse_posterior":
                        float((posterior_risk - actual_risk).square().mean().sqrt()),
                })
            print("POSTERIOR CONTEXT", seed, name, len(records), flush=True)
    cells = defaultdict(list)
    for row in records:
        cells[row["support_count"], row["policy"]].append(row)
    numeric = ("hypothesis_entropy", "effective_hypotheses", "kl_from_uniform_prior",
               "true_controller_posterior_mass_for_audit_only",
               "normalized_behavior_mean_error_for_audit_only",
               "query_risk_rmse_prior", "query_risk_rmse_posterior")
    aggregate = [
        {"support_count": count, "policy": policy,
         **{key: float(np.mean([row[key] for row in values])) for key in numeric},
         "by_controller": {
             controller: {key: float(np.mean([row[key] for row in values
                                              if row["controller"] == controller]))
                          for key in numeric}
             for controller in ("IDM", "FVDM")}}
        for (count, policy), values in sorted(cells.items())
    ]
    verify_or_lock()
    summary = {
        "role": "posterior diagnosis on 360 previously fixed validation contexts",
        "risk_inputs": "support risk only; query risk and true parameters used for audit metrics only",
        "risk_prediction": "expected clipped Gaussian including working observation noise",
        "record_count": len(records), "new_physical_measurements": 0,
        "new_contexts": 0, "models_and_source_seal_verified": True,
        "records": records, "aggregate": aggregate,
    }
    write_json(RESULTS / "posterior_context_diagnostic.json", summary)
    rows = ["# 固定验证上下文的风险后验诊断", "",
            "复用已有上下文的 10/50/150 风险支持；目标参数只用于诊断，不进入推断。",
            "有效假设数为 exp(后验熵)，均匀初始值为 512；风险 RMSE 针对固定 256 场景。", "",
            "| 支持数 | 方式 | 有效假设数 | 真控制器后验质量 | 风险 RMSE 初始 | 风险 RMSE 更新后 |",
            "|---:|---|---:|---:|---:|---:|"]
    for row in aggregate:
        rows.append(f"| {row['support_count']} | {row['policy']} | "
                    f"{row['effective_hypotheses']:.2f} | "
                    f"{row['true_controller_posterior_mass_for_audit_only']:.3f} | "
                    f"{row['query_risk_rmse_prior']:.4f} | "
                    f"{row['query_risk_rmse_posterior']:.4f} |")
    rows += ["", "后验集中不等于识别正确，真实碰撞搜索收益由完整池回放检验。", ""]
    (RESULTS / "posterior_context_diagnostic.md").write_text(
        "\n".join(rows), encoding="utf-8")
    print("POSTERIOR DIAGNOSTIC COMPLETE", len(records), flush=True)


if __name__ == "__main__":
    main()
