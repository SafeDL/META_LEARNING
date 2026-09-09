"""Run one explicit DIVA target-SUT protocol after source configuration is frozen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..diva.episode_executor import DivaEpisodeExecutor
from ..diva.baselines import TargetOnlyGP
from ..diva.miner import DivaMiner
from ..diva.prior import LowRankVulnerabilityPrior
from ..diva.source_bank import SourceBank, observation_from_dict, retained_task, sobol_designs
from ..evaluation.diva_protocol import DivaBudgetLedger
from ..scenario.catalog import mvr_parameter_spaces
from ..scenario.executor import ScenarioExecutor
from ..scenario.registry import load_adapters
from ..scenario.taskbook import load_taskbook
from ..training.runner import HierarchicalRunner
from ..diva.validity import RBFEvaluability


def _load(path: str):
    return [observation_from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _source_logical_domain(observations) -> str:
    domains = {observation.logical_domain_id for observation in observations}
    if len(domains) != 1:
        raise ValueError("DIVA source bank must contain exactly one frozen logical domain")
    return domains.pop()


METHODS = (
    "random",
    "frozen_source_mean",
    "target_only_gp",
    "online_diva_k0",
    "diva_random_support",
    "diva_diagnostic",
)


def run(config_path: str, source_path: str, prior_path: str, output_path: str, *, sut_split: str, protocol: str, support_shots: int, algorithm_seed: int, method: str = "diva_diagnostic") -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if protocol not in {"adaptation_quality", "frozen_selection", "budget_efficiency"}:
        raise ValueError("unsupported DIVA evaluation protocol")
    if method not in METHODS:
        raise ValueError("unsupported DIVA evaluation method")
    if sut_split not in {"validation", "test"}:
        raise ValueError("DIVA evaluation only accepts a held-out validation or test SUT")
    target_ref = config["study"][f"{sut_split}_sut_ref"]
    tasks = load_taskbook(config["taskbook"])
    source_observations = _load(source_path)
    logical_domain_id = _source_logical_domain(source_observations)
    task = retained_task(tasks, target_ref, logical_domain_id)
    bank = SourceBank.from_observations(source_observations, tuple(config["study"]["source_sut_refs"]))
    features = np.tile(bank.features, (len(bank.source_refs), 1))
    evaluability = RBFEvaluability(features, bank.eligible.reshape(-1), float(config["validity"]["bandwidth"]))
    pool = sobol_designs(task, int(config["candidate_pool"]["per_candidate"]), int(algorithm_seed))
    prior = LowRankVulnerabilityPrior.load(prior_path)
    miner = DivaMiner(prior, pool, evaluability.predict(np.asarray([design.feature_vector() for design in pool])))
    executor = DivaEpisodeExecutor(
        ScenarioExecutor(load_adapters(), mvr_parameter_spaces()),
        HierarchicalRunner(int(config["execution"]["runner_step_budget"])),
        int(config["execution"]["environment_horizon"]),
    )
    rng = np.random.default_rng(algorithm_seed)
    target_only = TargetOnlyGP(pool, rng, device=str(config["device"])) if method == "target_only_gp" else None
    support_count = support_shots if method in {"diva_random_support", "diva_diagnostic", "target_only_gp"} else 0
    if method in {"random", "frozen_source_mean", "online_diva_k0"} and support_shots not in {0, 4}:
        raise ValueError("this baseline does not use diagnostic support")
    budget = DivaBudgetLedger(20 if protocol == "budget_efficiency" else support_count + 8)
    observations = []
    for index in range(support_count):
        if method == "diva_diagnostic":
            design = miner.select_diagnostic()
        elif method == "diva_random_support":
            design = miner.select_random(rng)
        else:
            assert target_only is not None
            design = target_only.select()
        slot = budget.reserve(phase="support", design_id=design.design_id, episode_seed=algorithm_seed + index)
        observation = executor.run(task, design, algorithm_seed + index)
        budget.complete(slot, status=observation.status, score=observation.score)
        if method != "frozen_source_mean":
            if target_only is not None:
                target_only.observe(observation)
            else:
                miner.observe(observation)
        observations.append(observation.to_dict())
    remaining = 8 if protocol in {"adaptation_quality", "frozen_selection"} else budget.total_budget - budget.consumed
    query_designs = (
        sobol_designs(task, 4, int(algorithm_seed) + 1_000_000)
        if protocol == "adaptation_quality" else ()
    )
    for offset in range(remaining):
        if protocol == "adaptation_quality":
            design = query_designs[offset]
            selected_ids = target_only.selected_ids if target_only is not None else miner.selected_ids
            if design.design_id in selected_ids:
                raise RuntimeError("fixed query overlaps the support set")
            selected_ids.add(design.design_id)
        elif method == "random":
            design = miner.select_random(rng)
        elif target_only is not None:
            design = target_only.select()
        elif method == "frozen_source_mean":
            design = miner.select_source_mean()
        else:
            design = miner.select_mining()
        phase = "query" if protocol in {"adaptation_quality", "frozen_selection"} else "mining"
        slot = budget.reserve(phase=phase, design_id=design.design_id, episode_seed=algorithm_seed + support_count + offset)
        prediction = None
        if protocol == "adaptation_quality" and method != "random" and target_only is None:
            mean, variance, _, _ = prior.predict((design,), miner.posterior)
            prediction = {"mean": float(mean[0]), "variance": float(variance[0])}
        observation = executor.run(task, design, algorithm_seed + support_count + offset)
        budget.complete(slot, status=observation.status, score=observation.score)
        if protocol == "budget_efficiency":
            if target_only is not None:
                target_only.observe(observation)
            elif method not in {"random", "frozen_source_mean"}:
                miner.observe(observation)
        record = observation.to_dict()
        if prediction is not None:
            record["prediction_before_query"] = prediction
        observations.append(record)
    report = {
        "schema": "diva_target_evaluation",
        "protocol": protocol,
        "sut_split": sut_split,
        "sut_ref": target_ref,
        "logical_domain_id": logical_domain_id,
        "support_shots": support_shots,
        "method": method,
        "algorithm_seed": algorithm_seed,
        "budget": budget.report(),
        "observations": observations,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mvr/configs/diva_cutin.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sut-split", choices=("validation", "test"), required=True)
    parser.add_argument("--protocol", choices=("adaptation_quality", "frozen_selection", "budget_efficiency"), required=True)
    parser.add_argument("--support-shots", type=int, default=4)
    parser.add_argument("--algorithm-seed", type=int, default=1701)
    parser.add_argument("--method", choices=METHODS, default="diva_diagnostic")
    args = parser.parse_args()
    run(args.config, args.source, args.prior, args.output, sut_split=args.sut_split, protocol=args.protocol, support_shots=args.support_shots, algorithm_seed=args.algorithm_seed, method=args.method)


if __name__ == "__main__":
    main()
