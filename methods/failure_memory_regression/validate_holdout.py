"""Run the separately approved age080 holdout without adaptive filtering."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from methods.failure_memory_regression.interaction import (
    measure, read_jsonl,
)
from methods.failure_memory_regression.prepare_holdout import (
    BUILDS, CATALOGUE, ROOT, SEEDS, prepare_holdout,
)
from methods.failure_memory_regression.replay_utils import is_parent_pass
from methods.failure_memory_regression.schema import stable_hash
from methods.failure_memory_regression.selector import (
    METHODS, TargetOracle, run_selector,
)


TARGET = "mobil_rear_state_age080"
PARENT = "mobil_ref_v2"
SOURCE = "mobil_rear_guard_off_v2"
BUDGET = 20
REPEATS = 10
CHECKPOINTS = (1, 5, 10, 20)


def _seed_for(protocol: dict, simulator_seed: int, repeat: int) -> int:
    token = {"schema": protocol["schema"], "simulator_seed": simulator_seed,
             "repeat": repeat, "candidate_sha256": protocol[
                 "candidate_sha256_by_seed"][str(simulator_seed)]}
    return int(stable_hash(token)[:8], 16)


def _bank(root: Path, seed: int, build: str, cases: list[dict],
          protocol: dict) -> dict[str, dict]:
    rows = read_jsonl(root / f"seed{seed}" / "measured" / f"{build}.jsonl")
    by_id = {row["scenario_id"]: row for row in rows}
    case_by_id = {case["scenario_id"]: case for case in cases}
    if len(rows) != len(cases) or len(by_id) != len(cases) or set(by_id) != set(case_by_id):
        raise ValueError(f"incomplete or duplicate measured bank: {seed}/{build}")
    for scenario_id, row in by_id.items():
        if (row.get("scenario") != case_by_id[scenario_id]
                or row.get("build_id") != build
                or row.get("simulator_seed") != seed
                or row.get("build_fingerprint") != protocol[
                    "build_fingerprints"][build]):
            raise ValueError(f"measured contract mismatch: {seed}/{build}/{scenario_id}")
    return by_id


def evaluate_seed(cases: list[dict], banks: dict[str, dict[str, dict]],
                  protocol: dict, seed: int, budget: int = BUDGET,
                  repeats: int = REPEATS) -> tuple[list[dict], list[dict], int]:
    """Evaluate fixed cases; only the isolated oracle reveals target rows."""
    history = [{**row, "visibility": "historical",
                "context_id": row["scenario"]["context_id"]}
               for row in banks[SOURCE].values()]
    candidates = [{**case, "scenario": case, "parent_pass": True}
                  for case in cases if is_parent_pass(banks[PARENT][case["scenario_id"]])]
    failure_pool = sum(banks[TARGET][case["scenario_id"]].get("ego_collision") is True
                       and not banks[TARGET][case["scenario_id"]].get("inconclusive", False)
                       for case in candidates)
    summaries, runs = [], []
    for repeat in range(repeats):
        selector_seed = _seed_for(protocol, seed, repeat)
        for method in METHODS:
            oracle = TargetOracle(banks[TARGET])
            queries, _, _, updates = run_selector(
                method, candidates, history, oracle, budget, selector_seed,
                target_build_id=TARGET, parent_build_id=PARENT, mode="regression",
                session_id=f"age080:{seed}:{method}:{repeat}")
            summary = {"simulator_seed": seed, "selector_seed": selector_seed,
                       "repeat": repeat, "method": method,
                       "eligible_candidates": len(candidates),
                       "target_failure_pool": failure_pool}
            for checkpoint in CHECKPOINTS:
                summary[f"hits_at_{checkpoint}"] = sum(
                    row["selection_reward"] for row in queries[:checkpoint])
            summaries.append(summary)
            runs.append({**summary, "queries": queries, "updates": updates})
    return summaries, runs, failure_pool


def _sign_flip_p(values: list[float]) -> float:
    nonzero = [value for value in values if value]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    extreme = sum(abs(sum(sign * value for sign, value in zip(signs, nonzero)))
                  >= observed - 1e-12
                  for signs in itertools.product((-1, 1), repeat=len(nonzero)))
    return extreme / 2 ** len(nonzero)


def analyze(summaries: list[dict]) -> dict:
    seeds = sorted({row["simulator_seed"] for row in summaries})
    if seeds != list(SEEDS):
        raise ValueError("the complete eight-seed holdout is required")
    lookup = {(row["simulator_seed"], row["repeat"], row["method"]): row
              for row in summaries}
    if len(lookup) != len(SEEDS) * REPEATS * len(METHODS):
        raise ValueError("incomplete or duplicate selector runs")
    differences = {}
    failure_pools = {}
    totals = defaultdict(int)
    for seed in SEEDS:
        seed_rows = [row for row in summaries if row["simulator_seed"] == seed]
        failure_pools[str(seed)] = seed_rows[0]["target_failure_pool"]
        if any(row["target_failure_pool"] != failure_pools[str(seed)]
               for row in seed_rows):
            raise ValueError("failure pool changed across methods")
        differences[str(seed)] = sum(
            lookup[seed, repeat, "FBRT-Memory"]["hits_at_20"] -
            lookup[seed, repeat, "FBRT-NoMemory"]["hits_at_20"]
            for repeat in range(REPEATS)) / REPEATS
        for repeat in range(REPEATS):
            selector_seeds = {lookup[seed, repeat, method]["selector_seed"]
                              for method in METHODS}
            if len(selector_seeds) != 1:
                raise ValueError("methods did not share a selector seed")
            for method in METHODS:
                totals[method] += lookup[seed, repeat, method]["hits_at_20"]
    memory, no_memory = totals["FBRT-Memory"], totals["FBRT-NoMemory"]
    p_value = _sign_flip_p(list(differences.values()))
    nonnegative = sum(value >= 0 for value in differences.values())
    eligible_failures = sum(failure_pools.values())
    return {
        "seed_differences_memory_minus_no_memory_at_20": differences,
        "target_failure_pool_by_seed": failure_pools,
        "total_hits_at_20_by_method": dict(totals),
        "sign_flip_p_two_sided": p_value,
        "nonnegative_seed_count": nonnegative,
        "memory_improvement_gate": bool(
            eligible_failures > 0 and memory > no_memory
            and memory >= 1.2 * no_memory and p_value < 0.05
            and nonnegative >= 6),
        "memory_overall_best_descriptive": all(
            memory >= count for count in totals.values()),
        "physical_seed_count": len(SEEDS),
        "selector_repeats_per_seed": REPEATS,
    }


def validate_measurements(root: Path, protocol: dict) -> tuple[dict[int, list[dict]],
                                                                dict[int, dict]]:
    cases_by_seed, banks_by_seed = {}, {}
    for seed in SEEDS:
        cases = read_jsonl(root / f"seed{seed}" / "scenario_cases.jsonl")
        if len(cases) != 128:
            raise ValueError(f"missing frozen cases for seed {seed}")
        cases_by_seed[seed] = cases
        banks_by_seed[seed] = {build: _bank(root, seed, build, cases, protocol)
                               for build in BUILDS}
    return cases_by_seed, banks_by_seed


def run_evaluation(root: Path = ROOT) -> dict:
    root = Path(root)
    protocol = prepare_holdout(root)
    cases_by_seed, banks_by_seed = validate_measurements(root, protocol)
    summaries, runs = [], []
    for seed in SEEDS:
        local, local_runs, _ = evaluate_seed(cases_by_seed[seed],
                                             banks_by_seed[seed], protocol, seed)
        summaries.extend(local)
        runs.extend(local_runs)
    result = analyze(summaries)
    result["physical_protocol_sha256"] = hashlib.sha256(
        (root / "protocol.json").read_bytes()).hexdigest()
    result["evaluation_code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    evaluation = root / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / "summary.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in summaries), encoding="utf-8")
    (evaluation / "queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in runs), encoding="utf-8")
    (evaluation / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return result


def _measure_seed(root: Path, protocol: dict, seed: int) -> None:
    seed_root = root / f"seed{seed}"
    cases = read_jsonl(seed_root / "scenario_cases.jsonl")
    for build in BUILDS:
        existing = read_jsonl(seed_root / "measured" / f"{build}.jsonl")
        if existing:
            by_id = {row["scenario_id"]: row for row in existing}
            case_by_id = {case["scenario_id"]: case for case in cases}
            if len(existing) != len(by_id) or not set(by_id).issubset(case_by_id):
                raise ValueError(f"invalid partial measured bank: {seed}/{build}")
            for scenario_id, row in by_id.items():
                if (row.get("scenario") != case_by_id[scenario_id]
                        or row.get("simulator_seed") != seed
                        or row.get("build_fingerprint") != protocol[
                            "build_fingerprints"][build]):
                    raise ValueError(f"stale measured cache: {seed}/{build}/{scenario_id}")
        measure(seed_root, (build,), seed=seed, catalogue_path=CATALOGUE)


def run_measurement(root: Path = ROOT, workers: int = 1) -> None:
    root = Path(root)
    protocol = prepare_holdout(root)
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        for seed in SEEDS:
            _measure_seed(root, protocol, seed)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_measure_seed, root, protocol, seed)
                       for seed in SEEDS]
            for future in futures:
                future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--measure", action="store_true",
                        help="run all 3072 frozen physical case/build episodes")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallelize measurement across independent frozen seeds")
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluate the complete measured holdout bank")
    args = parser.parse_args()
    if args.measure:
        run_measurement(args.root, workers=args.workers)
    if args.evaluate:
        print(json.dumps(run_evaluation(args.root), ensure_ascii=False, indent=2))
    if not args.measure and not args.evaluate:
        print(json.dumps(prepare_holdout(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
