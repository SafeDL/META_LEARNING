"""Run fixed-suite policy regression with independently resumable records."""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from experiments.regression.fixed_inputs import CHECKPOINTS, H0_COUNTS, PPO_METHODS, VERSIONS, fixed_bank, reference_history, reference_protocol, reference_root, scene_root, verify_chain, version_directory
from experiments.parallel import COLLECTION_WORKERS, initialize_worker, prepare_processes
from experiments.regression.execution import checkpoint_identity, execute_record
from experiments.regression.preflight import CHECKPOINTS, DQN_CHECKPOINTS, DQN_CONTINUAL, POLICIES, preflight
from experiments.regression.protocol import repair_subset
from experiments.regression.queries import query_reference_round
from experiments.regression.repair import repair
from experiments.schema import Scenario
from experiments.storage import read_json, write_json
from simulator.highway.repair_env import physical_coordinate


def collect(job):
    checkpoint, scene, path, purpose, policy = job
    result = execute_record(checkpoint, scene, path, purpose=purpose,
                            public_interface=(purpose == "fixed_suite"), policy=policy)
    return bool(result["feedback"]["collision"])


def collect_all(jobs, workers, label):
    prepare_processes()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        hits = 0
        for n, collision in enumerate(pool.map(collect, jobs, chunksize=8), 1):
            hits += collision
            if n % 200 == 0 or n == len(jobs):
                print(label, n, "/", len(jobs), "collisions", hits, flush=True)


def build_reference_chain(kind, protocol, suite, workers, policy):
    root = reference_root(kind, policy)
    path = root / "random/chain.json"
    if path.exists():
        return
    if preflight(policy)["status"] != "passed":
        raise RuntimeError(f"The historical {policy.upper()} checkpoints did not pass capability checks")
    checkpoints = CHECKPOINTS if policy == "ppo" else DQN_CHECKPOINTS
    jobs = [(checkpoint, Scenario(**scene), root / "history" / name / f"{i:04d}.json.gz", "history", policy)
            for name, checkpoint in checkpoints.items() for i, scene in enumerate(protocol["history"])]
    collect_all(jobs, workers, f"{kind} ancestors")
    archives = reference_history(kind, protocol, suite, policy)
    old = [s for s, hit in zip(archives[-1].scenarios, archives[-1].collision) if hit]
    excluded = {physical_coordinate(s) for s in suite}
    parent = checkpoints["parent"]
    current, historical, rng = repair_subset(old), [], None
    chain = {"method": "random", "kind": kind, "rounds": [], "status": "running"}
    for iteration, version in enumerate(VERSIONS[1:]):
        entry = {"iteration": iteration, "parent_sha256": checkpoint_identity(parent)}
        print(kind, version, "repair reset points", len(current), flush=True)
        if not current:
            chain.update(status="terminated_without_new_failure", last_version=VERSIONS[iteration])
            break
        destination = root / ("random/repairs" if iteration else "initialization")
        try:
            receipt = repair(parent, current, historical, excluded, destination,
                             preparation=(iteration == 0), parent_random_state=rng,
                             iteration=iteration or None, policy=policy)
        except Exception as error:
            chain.update(status="repair_interrupted", last_version=VERSIONS[iteration], error=repr(error))
            write_json(scene_root(kind, policy) / "status.json", chain)
            raise
        parent = destination / receipt["checkpoint_name"]
        rng = destination / receipt["random_state_name"]
        entry.update(checkpoint_sha256=checkpoint_identity(parent),
                     actual_training_steps=receipt["actual_environment_steps"],
                     wall_s=receipt["wall_s"], status="complete" if receipt["capability"]["qualified"] else "unqualified")
        if iteration:
            chain["rounds"].append(entry)
        else:
            chain["common_checkpoint_sha256"] = entry["checkpoint_sha256"]
            chain["preparation"] = entry
        if not receipt["capability"]["qualified"]:
            chain.update(status="terminated_unqualified", last_version=version,
                         failed_courses=[c["course"] for c in receipt["capability"]["courses"] if not c["passed"]])
            break
        if version == "R3":
            chain["status"] = "complete"
            break
        candidates = [Scenario(**s) for s in protocol["candidates"][iteration]]
        result = query_reference_round(candidates, protocol["initial_indices"], parent,
                                       root / "random/queries", iteration=iteration + 1, execution_root=root,
                                       policy=policy)
        selected = result["result"]["queries"]
        historical = repair_subset(old)
        current = [candidates[q["index"]] for q in selected if q["feedback"]["collision"]]
        old.extend(current)
    write_json(path, chain)


def prepare(kind, workers, *, build_reference=False, policy="ppo"):
    if policy == DQN_CONTINUAL:
        from experiments.regression.dqn_evaluation import prepare_regression

        return prepare_regression(kind)
    bank = fixed_bank(kind, policy)
    protocol = reference_protocol(kind, bank.candidates, policy)
    if not (reference_root(kind, policy) / "random/chain.json").exists():
        if not build_reference:
            raise RuntimeError("The reference chain must finish before replay or independent reporting")
        build_reference_chain(kind, protocol, bank.candidates, workers, policy)
    archives = reference_history(kind, protocol, bank.candidates, policy)
    models, paid = verify_chain(kind, bank, protocol, archives, policy)
    return bank, archives, models, paid


def physical_suite(kind, workers, policy="ppo"):
    bank, _, models, _ = prepare(kind, workers, build_reference=True, policy=policy)
    jobs = [(models[version], s, version_directory(kind, "truth", version, policy) / f"{i:06d}.json.gz",
             "fixed_suite", policy)
            for version in models if version != "h0" for i, s in enumerate(bank.candidates)]
    if jobs:
        collect_all(jobs, workers, kind)


def main():
    from experiments.paper_results import regression_policy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reference", "physical", "replay", "report"))
    parser.add_argument("--policy", choices=tuple(POLICIES), default="ppo")
    parser.add_argument("--scene", choices=H0_COUNTS, action="append")
    parser.add_argument("--version", choices=VERSIONS[1:4], action="append",
                        help="Restrict replay to the selected versions")
    parser.add_argument("--method", choices=PPO_METHODS, action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="PyTorch device for Ours method replay")
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="PyTorch threads per Ours method replay worker")
    parser.add_argument("--workers", type=int, default=COLLECTION_WORKERS)
    options = parser.parse_args()
    options.policy = regression_policy(options.policy)
    initialize_worker()
    kinds = options.scene or H0_COUNTS
    if options.stage == "report":
        from experiments.regression.fixed_report import report
        report(kinds, policy=options.policy)
        return
    if options.stage == "replay":
        from experiments.regression.fixed_replay import replay
        replay(kinds, options.workers, options.version or VERSIONS[1:4],
               options.method, options.device, options.cpu_threads, policy=options.policy)
        return
    failures = []
    for kind in kinds:
        try:
            if options.stage == "physical":
                physical_suite(kind, options.workers, options.policy)
            else:
                prepare(kind, options.workers, build_reference=True, policy=options.policy)
        except Exception as error:
            failures.append((kind, str(error)))
            print(kind, repr(error), flush=True)
    if failures:
        raise RuntimeError(f"Incomplete {options.policy.upper()} reference or suite: {failures}")


if __name__ == "__main__":
    main()
