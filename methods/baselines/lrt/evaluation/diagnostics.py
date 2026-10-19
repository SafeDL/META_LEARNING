"""Evaluator-only continuation audits and fixed-support prediction checks."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from scipy.special import stdtr
from scipy.stats import spearmanr
import torch
from threadpoolctl import threadpool_limits

from methods.baselines.cg_brt.posterior import RiskPosterior
from methods.baselines.dart.evaluation.config import SUT_IDS
from methods.baselines.hgt.config import PROFILES
from methods.baselines.hgt.history import load_history, split_indices, subset
from methods.baselines.hgt.io import read_json, write_json

from ..config import JITTER, NOISE, ROOT, SEEDS
from ..data import assemble
from ..posterior import model_set
from ..teacher import branch_return, replay_reference
from .evaluate import bank_path, trace_path


def audit_case(case):
    pool, seed = case
    destination = ROOT / "diagnostics" / f"pool_{pool}_seed_{seed}.json"
    if destination.exists():
        return read_json(destination)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        history = load_history()
        train, _ = split_indices(next(iter(history.values()))["x"])
        sources = subset(history, train)
        with np.load(bank_path(pool, SUT_IDS[0])) as bank:
            x = bank["x"].copy()
        common, prior = assemble(x, sources, seed)
        audits, few_shot = [], []
        for sut in SUT_IDS:
            with np.load(bank_path(pool, sut)) as bank:
                risk = bank["risk"].copy()
            traces = {
                variant: read_json(trace_path(pool, sut, seed, variant))
                for variant in ("cg_brt", "previous", "full")
            }
            for variant, trace in traces.items():
                for step in (10, 50, 100, 150, 190):
                    anchor, remaining = replay_reference(
                        common, trace["queries"][:step], 200, 10
                    )
                    advised, _ = anchor.recommend(remaining, step)
                    selected = trace["selected_indices"][step]
                    record = trace["decisions"][step]
                    if variant != "cg_brt" and advised != record["reference_index"]:
                        raise ValueError(
                            "Recorded same-history reference differs from replay"
                        )
                    if variant == "cg_brt" and advised != selected:
                        raise ValueError(
                            "Frozen baseline differs from reconstructed advice"
                        )
                    ref_return, _ = branch_return(anchor, remaining, advised, risk)
                    chosen_return = (
                        ref_return
                        if selected == advised
                        else branch_return(anchor, remaining, selected, risk)[0]
                    )
                    if variant == "full":
                        predicted = record["conservative_gain"]
                    elif variant == "previous":
                        predicted = record.get("selected_value", 0) - record.get(
                            "reference_value", 0
                        )
                    else:
                        predicted = 0.0
                    audits.append(
                        {
                            "pool": pool,
                            "seed": seed,
                            "sut": sut,
                            "variant": variant,
                            "observations": step,
                            "selected": selected,
                            "reference": advised,
                            "predicted_gain": float(predicted),
                            "realized_gain": chosen_return - ref_return,
                            "selected_return": chosen_return,
                            "reference_return": ref_return,
                            "label_role": "evaluation_only",
                        }
                    )
            anchor, remaining = replay_reference(common, [], 200, 10)
            dart_posterior = RiskPosterior(
                prior.enhanced_covariance,
                common["priors"],
                common["probability"],
                common["fixed_prior"],
            )
            models = model_set(prior)
            for step, query in enumerate(traces["cg_brt"]["queries"][:20], 1):
                anchor.recommend(remaining, step - 1)
                anchor.observe(query["index"], query["risk"], query["high_risk"])
                dart_posterior.observe(query["index"], query["risk"])
                for model in models:
                    model.observe(query["index"], query["risk"])
                remaining[query["index"]] = False
                if step not in (1, 2, 4, 10, 20):
                    continue
                anchor.recommend(remaining, step)
                probability = {
                    "cg_brt": np.exp(anchor.gated[0]),
                    "previous": np.exp(dart_posterior.log_probabilities()[0][0]),
                    "full": models[1].probabilities(),
                }
                available = np.flatnonzero(remaining)
                high = risk[available] > 0.5
                for variant, p in probability.items():
                    top = np.argsort(-p[available], kind="stable")[: 200 - step]
                    few_shot.append(
                        {
                            "pool": pool,
                            "seed": seed,
                            "sut": sut,
                            "K": step,
                            "variant": variant,
                            "brier": float(np.mean((p[available] - high) ** 2)),
                            "remaining_top_budget_hits": int(
                                np.count_nonzero(high[top])
                            ),
                            "support_indices": traces["cg_brt"]["selected_indices"][
                                :step
                            ],
                        }
                    )
                source_names = [p.name for p in PROFILES if p.name in sources]
                for variant, posterior in (
                    ("cg_brt", anchor.model),
                    ("previous", dart_posterior),
                ):
                    for family in (0, 1):
                        unqueried = available[x[available, 4] == family]
                        component = stdtr(
                            2 * posterior.alpha,
                            (posterior.source_mean[unqueried] - 0.5)
                            / np.sqrt(
                                (posterior.variance[unqueried, None] + NOISE + JITTER)
                                * posterior.beta
                                / posterior.alpha
                            ),
                        )
                        few_shot.append(
                            {
                                "pool": pool,
                                "seed": seed,
                                "sut": sut,
                                "K": step,
                                "variant": "source_diagnostic",
                                "policy": variant,
                                "family": family,
                                "source_names": source_names,
                                "weights": posterior.weights.tolist(),
                                "component_brier": np.mean(
                                    (component - (risk[unqueried] > 0.5)[:, None]) ** 2,
                                    axis=0,
                                ).tolist(),
                                "reference_trust": (
                                    anchor.gate.trust if variant == "cg_brt" else None
                                ),
                                "label_role": "evaluation_only",
                            }
                        )
                for family, (rows, block) in models[1].blocks.items():
                    unqueried = rows[remaining[rows]]
                    mean, variance, beta = block.parameters(
                        models[1].positions[unqueried]
                    )
                    component = stdtr(
                        2 * block.alpha,
                        (mean - 0.5) / np.sqrt(variance * beta / block.alpha),
                    )
                    errors = np.mean(
                        (component - (risk[unqueried] > 0.5)[:, None]) ** 2, axis=0
                    )
                    few_shot.append(
                        {
                            "pool": pool,
                            "seed": seed,
                            "sut": sut,
                            "K": step,
                            "variant": "source_diagnostic",
                            "policy": "full",
                            "family": family,
                            "source_names": [*prior.names, "target_only"],
                            "weights": np.exp(block.log_weight).tolist(),
                            "component_brier": errors.tolist(),
                            "label_role": "evaluation_only",
                        }
                    )
    result = {"audits": audits, "few_shot": few_shot}
    write_json(destination, result)
    print(f"DIAGNOSTICS {pool} {seed}", flush=True)
    return result


def gain_statistics(rows):
    predicted = np.array([r["predicted_gain"] for r in rows])
    actual = np.array([r["realized_gain"] for r in rows])
    horizon = np.array([200 - r["observations"] for r in rows])
    informative = len(rows) > 1 and np.std(predicted) > 0 and np.std(actual) > 0
    normalized = (
        len(rows) > 1
        and np.std(predicted / horizon) > 0
        and np.std(actual / horizon) > 0
    )
    return {
        "audited_deviations": len(rows),
        "gain_mse": float(np.mean((predicted - actual) ** 2)) if rows else None,
        "mean_realized_gain": float(actual.mean()) if rows else None,
        "positive_gain_fraction": float(np.mean(actual > 0)) if rows else None,
        "nonnegative_gain_fraction": float(np.mean(actual >= 0)) if rows else None,
        "gain_pearson_correlation": (
            float(np.corrcoef(predicted, actual)[0, 1]) if informative else None
        ),
        "gain_spearman_correlation": (
            float(spearmanr(predicted, actual).statistic) if informative else None
        ),
        "normalized_gain_mse": (
            float(np.mean(((predicted - actual) / horizon) ** 2)) if rows else None
        ),
        "normalized_gain_correlation": (
            float(np.corrcoef(predicted / horizon, actual / horizon)[0, 1])
            if normalized
            else None
        ),
    }


def main():
    cases = [(pool, seed) for pool in (0, 1, 2) for seed in SEEDS]
    paths = [
        ROOT / "diagnostics" / f"pool_{pool}_seed_{seed}.json" for pool, seed in cases
    ]
    pending = [case for case, path in zip(cases, paths) if not path.exists()]
    if pending:
        with ProcessPoolExecutor(
            max_workers=12, mp_context=mp.get_context("spawn")
        ) as executor:
            list(executor.map(audit_case, pending))
    reports = [read_json(path) for path in paths]
    audits = [row for report in reports for row in report["audits"]]
    few_shot = [row for report in reports for row in report["few_shot"]]
    summary = {}
    for variant in ("previous", "full"):
        rows = [
            row
            for row in audits
            if row["variant"] == variant and row["selected"] != row["reference"]
        ]
        summary[variant] = gain_statistics(rows)
        summary[variant]["by_observations"] = {
            str(step): gain_statistics(
                [row for row in rows if row["observations"] == step]
            )
            for step in (10, 50, 100, 150, 190)
        }
    write_json(
        ROOT / "diagnostics.json",
        {
            "summary": summary,
            "fixed_support_summary": {
                variant: {
                    str(k): {
                        metric: float(
                            np.mean(
                                [
                                    row[metric]
                                    for row in few_shot
                                    if row["variant"] == variant and row["K"] == k
                                ]
                            )
                        )
                        for metric in ("brier", "remaining_top_budget_hits")
                    }
                    for k in (1, 2, 4, 10, 20)
                }
                for variant in ("cg_brt", "previous", "full")
            },
            "source_choice_brier_regret": {
                variant: {
                    str(k): float(
                        np.mean(
                            [
                                row["component_brier"][int(np.argmax(row["weights"]))]
                                - min(row["component_brier"])
                                for row in few_shot
                                if row["variant"] == "source_diagnostic"
                                and row["policy"] == variant
                                and row["K"] == k
                            ]
                        )
                    )
                    for k in (1, 2, 4, 10, 20)
                }
                for variant in ("cg_brt", "previous", "full")
            },
            "case_files": [path.relative_to(ROOT).as_posix() for path in paths],
            "audit_count": len(audits),
            "few_shot_count": len(few_shot),
            "labels_returned_to_selector": False,
            "physical_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
