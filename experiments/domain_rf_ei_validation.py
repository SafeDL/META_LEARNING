"""Stage A: one real development-bank replay and independent RF-EI checks."""

import time
from dataclasses import asdict
from importlib.metadata import version

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from controllers.highway.catalog import BY_ID
from testing_methods.baselines.av_bo_rf_ei.method import DEFAULT as RF_CONFIG
from testing_methods.baselines.av_bo_rf_ei.method import risk, run

from .collision_metrics import collision_recall, summarize
from .construction import load_archives
from .discovery_config import write_json
from .domain_risk import disclose_impact
from .executions import ExecutionBank, execution_contract
from .initialization import initial_indices
from .protocol import DEFAULT, ROOT, TEST_RECORDS_ROOT
from .query import QueryLedger, run_isolated
from .scenarios import features
from .source_registration import specification_matches

OUTPUT = ROOT / "experiment_outputs/av_bo_rf_ei/collision/discovery/domain_comparison/stage_a"
TARGET = "development_idm_mobil"
SCENARIO = "cut_in"
SEED = 11


def independent_acquisition_check(result, candidates, steps=(9, 51, 101, 128)):
    """Refit the paper equations independently at initialization and tuning boundaries."""
    x = features(candidates)
    rows = result["queries"]
    checks = []
    for step in steps:
        observed = rows[: step - 1]
        indices = [r["index"] for r in observed]
        y = np.asarray([r["risk"] for r in observed])
        parameters = {
            name: getattr(RF_CONFIG, name)
            for name in ("n_estimators", "max_depth", "min_samples_split")
        }
        for tuning in result["tuning"]:
            if tuning["paid_observations"] <= len(observed):
                parameters = tuning["parameters"]
        full = RandomForestRegressor(**parameters, random_state=SEED, n_jobs=1).fit(x[indices], y)
        available = sorted(
            set(range(len(candidates))) - set(indices), key=lambda i: candidates[i].scenario_id
        )
        mean = full.predict(x[available])
        k = min(10, len(observed))
        folds = KFold(n_splits=k, shuffle=True, random_state=SEED)
        values = np.array(
            [
                RandomForestRegressor(**parameters, random_state=SEED, n_jobs=1)
                .fit(x[np.asarray(indices)[train]], y[train])
                .predict(x[available])
                for train, _ in folds.split(y)
            ]
        )
        # Algebraically equivalent to the pseudo-value formula, without calling it.
        variance = (k - 1) / k * ((values - values.mean(axis=0)) ** 2).sum(axis=0)
        sigma = np.sqrt(variance)
        delta = mean - y.max()
        z = np.divide(delta, sigma, out=np.zeros_like(delta), where=sigma > 0)
        ei = np.where(sigma > 0, delta * norm.cdf(z) + sigma * norm.pdf(z), np.maximum(delta, 0))
        position = int(np.argmax(ei))
        row = rows[step - 1]
        if available[position] != row["index"]:
            raise ValueError(f"Independent RF-EI selection differs at query {step}")
        errors = {
            "mean": abs(float(mean[position]) - row["prequery_risk_mean"]),
            "variance": abs(float(variance[position]) - row["prequery_risk_variance"]),
            "ei": abs(float(ei[position]) - row["expected_improvement"]),
        }
        if max(errors.values()) > 1e-10:
            raise ValueError(f"Independent RF-EI formula check failed at query {step}: {errors}")
        checks.append(
            {
                "query": step,
                "selected_index": row["index"],
                "absolute_errors": errors,
                "remaining_candidates": len(available),
            }
        )
    return checks


def main():
    _, development = load_archives(SCENARIO)
    archive = next(a for a in development if a.source_id == TARGET)
    bank = ExecutionBank(TEST_RECORDS_ROOT / "physical/development" / SCENARIO / TARGET)
    if (
        bank.candidates != archive.scenarios
        or len(bank.candidates) != DEFAULT.candidate_count
        or not specification_matches(bank.manifest["contract"], execution_contract(BY_ID[TARGET]))
    ):
        raise ValueError("Stage A requires the original complete qualified development bank")
    public = {
        "candidates": bank.candidates,
        "initial_indices": initial_indices(len(bank.candidates), DEFAULT.initial_count, SEED),
        "budget": DEFAULT.budget,
        "seed": SEED,
    }
    ledger = QueryLedger(
        bank.candidates, DEFAULT.budget, lambda scenario: disclose_impact(bank(scenario))
    )
    started = time.perf_counter()
    result = run_isolated(run, public, ledger)
    wall_s = time.perf_counter() - started
    write_json(
        OUTPUT / "first_run.json",
        {
            "stage": "A mechanism validation; not the frozen development comparison",
            "target_id": TARGET,
            "scenario": SCENARIO,
            "algorithm_seed": SEED,
            "initial_indices": public["initial_indices"],
            "settings": asdict(RF_CONFIG),
            "packages": {
                name: version(name) for name in ("numpy", "scipy", "scikit-learn", "optuna")
            },
            "result": result,
            "query_ledger": ledger.records,
            "metrics": {
                **summarize(bank.candidates, result["queries"], ledger.trajectories),
                **collision_recall(bank.candidates, archive.collision.tolist(), result["queries"]),
            },
            "cost": {
                "selector_and_replay_wall_s": wall_s,
                "new_physical_calls": 0,
                "equivalent_queries": DEFAULT.budget,
                "source_execution_bank": str(bank.root.relative_to(ROOT)),
            },
        },
    )
    checks = independent_acquisition_check(result, bank.candidates)
    for row in result["queries"]:
        original = bank(bank.candidates[row["index"]])
        expected = disclose_impact(original).feedback
        if row["feedback"] != expected.to_dict() or row["risk"] != risk(expected):
            raise ValueError("A disclosed scalar differs from its selected paid trajectory")
    report = {
        "status": "passed",
        "queries": len(result["queries"]),
        "initialization_included": True,
        "new_physical_calls": 0,
        "independent_acquisition_checks": checks,
        "tuning_sample_counts": [r["paid_observations"] for r in result["tuning"]],
        "feedback_trace_checks": len(result["queries"]),
        "run": str((OUTPUT / "first_run.json").relative_to(ROOT)),
        "full_comparison_frozen": False,
    }
    write_json(OUTPUT / "audit.json", report)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    main()
