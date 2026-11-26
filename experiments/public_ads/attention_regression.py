"""Build and audit the public Social Attention DDQN fixed-suite repair chain."""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
from math import prod

import numpy as np
from filelock import FileLock

from controllers.highway.public_agents import CHECKPOINTS, runtime_contract
from experiments.budget.protocol import sha
from experiments.formal_scenarios import FREE_BOUNDS, coordinate_stream, development_coordinates
from experiments.history_transfer_experiment import freeze
from experiments.parallel import initialize_worker, prepare_processes
from experiments.protocol import ROOT, episode_duration
from experiments.public_ads import extend
from experiments.public_ads.attention_repair import train_round
from experiments.public_ads.evaluate_e01 import OUTPUT as PUBLIC
from experiments.public_ads.checks import adapted_interface
from experiments.public_ads.qualify import capabilities, development_scenes, drive, verification_source
from experiments.qualification import verify_course_evidence
from experiments.regression.audit import verify_execution
from experiments.regression.dqn_learning import ANCESTOR_HISTORY, STUDY, used_coordinates
from experiments.regression.execution import ATTENTION, checkpoint_identity, response_archive
from experiments.regression.fixed_inputs import (
    AUC_WINDOW, NEW_WINDOW, SUITE_SIZE, VERSIONS, initial_indices, methods_for, output_root,
    paid_reference_records, query_directory, scene_root, version_directory,
)
from experiments.regression.fixed_replay import replay_job, source_snapshot, truth_rows
from experiments.regression.fixed_report import audit_ledger, compare, score, version_counts, write_csv
from experiments.regression.preflight import RUN_ROOT
from experiments.regression.protocol import QUERY_LIMIT, numeric_coordinate, repair_subset, sample_block
from experiments.regression.queries import query_reference_round
from experiments.regression.run_fixed import collect_all
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.schema import Scenario
from experiments.storage import read_json, write_json
from simulator.highway.repair_env import physical_coordinate
from simulator.highway.timing import PHYSICS_HZ
from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import load_model, train

OUTPUT = output_root(ATTENTION)
H0 = CHECKPOINTS / ATTENTION / "policy.pt"
METHODS = methods_for(ATTENTION)
SEED = 81001
HISTORY_SEED, REFERENCE_SEED, SELECTION_SEED = 75301, 75303, 75307
SELECTION_SIZE = 64


def preflight():
    source = PUBLIC / "qualification" / ATTENTION / "acceptance.json"
    frozen = read_json(source)
    directory = OUTPUT / "qualification"
    path = directory / "preflight.json"
    contract = runtime_contract(ATTENTION)
    if contract != frozen["runtime_contract"]:
        raise ValueError("Regression requires the original admitted attention deployment")
    spec = dict(runtime_contract=contract, verification_source=verification_source(),
                original_acceptance_sha256=sha(source))
    if path.exists():
        proof = read_json(path)
        evidence = verify_course_evidence(directory / "h0")
        if proof["specification"] != spec or not evidence["qualified"] or proof["courses"] != evidence["courses"]:
            raise ValueError("Current attention qualification differs from its regression receipt")
        return proof
    interface = adapted_interface(ATTENTION)
    qualified = capabilities(ATTENTION, directory / "h0")
    if not qualified["qualified"] or qualified["courses"] != frozen["courses"]:
        raise ValueError("Current attention h0 failed to reproduce its five admitted driving courses")
    development = []
    for scene, duration, _ in development_scenes():
        recorded = next(row for row in frozen["development_replays"] if row["scenario_id"] == scene.scenario_id)
        native_path = ROOT / recorded["trajectory"]
        if sha(native_path) != recorded["trajectory_sha256"]:
            raise ValueError("Sealed verification trajectory changed")
        native = read_json(native_path)
        current = drive(ATTENTION, scene, SEED, duration)
        if any(value != native[key] for key, value in current.items()):
            raise ValueError("Current attention policy did not reproduce its native development trajectory")
        development.append(recorded["trajectory_sha256"])
    return freeze(path, {**qualified, "specification": spec, "interface": interface,
                         "development_trajectory_hashes": development, "physical_executions": 14})


@lru_cache(maxsize=4)
def admitted_bank(kind):
    frozen = read_json(PUBLIC / "k_shot" / ATTENTION / kind / "11/full.json.gz")
    extend.initialize({"sources": {"specification": frozen["specification"]["source_snapshot"]}})
    return extend.bank_for(kind, ATTENTION)


def excluded_coordinates(kind, *, include_unpaid=True):
    paths = {
        RUN_ROOT / kind / "protocol.json",
        RUN_ROOT / "fixed_suite" / kind / "reference/protocol.json",
        ANCESTOR_HISTORY / kind / "protocol.json",
    }
    paths.update(STUDY.glob("**/coordinates.json"))
    paths.update(STUDY.glob(f"**/{kind}/reference/**/protocol.json"))
    used = used_coordinates(kind)
    if not include_unpaid:
        ancestor = read_json(ANCESTOR_HISTORY / kind / "protocol.json")
        # used_coordinates also reserves two earlier diagnostic cohorts. Keep
        # them while releasing the unexecuted reference-candidate coordinates.
        used.difference_update(numeric_coordinate(Scenario(**r)) for b in ancestor["candidates"] for r in b)
        used.update(development_coordinates(kind))
        for split in ("history", "target"):
            used.update(tuple(r["independent"][n] for n in FREE_BOUNDS[kind]) for r in coordinate_stream(kind, split))
        paid_paths = [*RUN_ROOT.glob(f"**/{kind}/**/ledger_*.json"), *STUDY.glob(f"**/{kind}/**/ledger_*.json")]
        for path in paid_paths:
            ledger = read_json(path)
            pool = ledger["specification"]["candidates"]
            for row in ledger["query_ledger"]:
                used.add(numeric_coordinate(Scenario(**pool[row["index"]])))

    def walk(value):
        if isinstance(value, list):
            for entry in value:
                walk(entry)
        elif isinstance(value, dict):
            if value.get("kind") == kind and set(FREE_BOUNDS[kind]) <= set(value.get("parameters", {})):
                used.add(numeric_coordinate(Scenario(
                    value["scenario_id"], kind, value["parameters"], value.get("categories", {})
                )))
            else:
                for key, entry in value.items():
                    if include_unpaid or key != "candidates":
                        walk(entry)

    for path in sorted(paths):
        if path.exists():
            walk(read_json(path))
    return used


def protocol(kind, bank, device):
    used = excluded_coordinates(kind)
    grid_size = prod(hi - lo + 1 for lo, hi in FREE_BOUNDS[kind].values())
    full_reservation = grid_size - len(used) >= 500 + 3 * SUITE_SIZE + 4 * SELECTION_SIZE
    if not full_reservation:
        used = excluded_coordinates(kind, include_unpaid=False)
    used.update(numeric_coordinate(s) for s in bank.candidates)
    pool_size = min(SUITE_SIZE, ((grid_size - len(used) - 500 - 4 * SELECTION_SIZE) // 6) * 2)
    if pool_size < QUERY_LIMIT:
        raise ValueError("Clean coordinate grid cannot supply the registered reference-query budget")
    history = sample_block(kind, 500, HISTORY_SEED, used, "attention_history")
    candidates = [sample_block(kind, pool_size, REFERENCE_SEED + i * 100, used, f"attention_reference_{i}")
                  for i in range(3)]
    selection = [sample_block(kind, SELECTION_SIZE, SELECTION_SEED + i * 100, used, f"attention_development_{i}")
                 for i in range(4)]
    # Existing selected records use an ordinal in their filename. Allocate a
    # globally unique ordinal even when a round carries forward its parent.
    candidates = [[replace(s, scenario_id=f"regression:{kind}:attention_reference:{j + i * pool_size:04d}")
                   for j, s in enumerate(block)] for i, block in enumerate(candidates)]
    return freeze(scene_root(kind, ATTENTION) / "reference/protocol.json", dict(
        history=[asdict(s) for s in history], candidates=[[asdict(s) for s in b] for b in candidates],
        selection=[[asdict(s) for s in b] for b in selection],
        initial_indices=initial_indices(pool_size), environment_seed=SEED, duration_s=episode_duration(kind),
        bank_sha256=bank.snapshot_fingerprint, h0_sha256=checkpoint_identity(H0), training_device=device,
        history_seed=HISTORY_SEED, reference_seeds=[REFERENCE_SEED + i * 100 for i in range(3)],
        selection_seeds=[SELECTION_SEED + i * 100 for i in range(4)],
        response_training=dict(fold_seed=SEED, folds=[250, 250], inference_sources=1),
    ))


def reference_history(kind, plan, workers):
    root = scene_root(kind, ATTENTION) / "reference/history"
    jobs = [(H0, Scenario(**scene), root / f"{i:04d}.json.gz", "history", ATTENTION)
            for i, scene in enumerate(plan["history"])]
    if any(not job[2].exists() for job in jobs):
        collect_all(jobs, workers, f"{kind} attention h0 history")
    rows = []
    for checkpoint, scene, path, _, _ in jobs:
        record = read_json(path)
        verify_execution(record, asdict(scene), plan["h0_sha256"])
        if record["purpose"] != "history":
            raise ValueError("Initial attention source needs measured history")
        rows.append(record)
    return rows


def collect_h0(kind, workers, device):
    preflight()
    bank = admitted_bank(kind)
    return reference_history(kind, protocol(kind, bank, device), workers)


def repair_chain(kind, bank, plan, history):
    root = scene_root(kind, ATTENTION) / "reference"
    outside = [Scenario(**s) for block in (plan["history"], *plan["candidates"], *plan["selection"])
               for s in block]
    coordinates = {numeric_coordinate(s) for s in outside}
    if (len(coordinates) != len(outside) or
            coordinates.intersection(excluded_coordinates(kind, include_unpaid=False))):
        raise ValueError("Public repair registration overlaps existing or reserved coordinates")
    excluded = {(s.kind, physical_coordinate(s)) for s in bank.candidates}
    excluded.update((s["kind"], physical_coordinate(Scenario(**s))) for b in plan["selection"] for s in b)
    parent = H0
    prior = [Scenario(**r["scenario"]) for r in history if r["feedback"]["collision"]]
    current, historical = repair_subset(prior), []
    rounds, records = [], []
    for i, version in enumerate(VERSIONS[1:]):
        if not current:
            break
        directory = root / version
        parent_sha = checkpoint_identity(parent)
        selection = [Scenario(**r) for r in plan["selection"][i]]
        report = train_round(parent, current, historical, selection, excluded, directory,
                             device=plan["training_device"])
        if report["step"]:
            parent = directory / "policy.pt"
        evidence = verify_course_evidence(directory / "qualification")
        training = read_json(directory / "training_protocol.json")
        if (evidence["courses"] != report["qualification"]["courses"] or not evidence["qualified"]
                or training["parent_sha256"] != parent_sha
                or report["checkpoint_sha256"] != checkpoint_identity(parent)
                or report["actual_online_steps"] != sum(sum(e["transitions"].values()) for e in report["curriculum"])):
            raise ValueError("Attention release failed its lineage, step or capability audit")
        rounds.append(dict(version=version, selected_step=report["step"], parent_sha256=parent_sha,
                           path=parent.relative_to(ROOT).as_posix(), sha256=report["checkpoint_sha256"],
                           reason="eligible_development_improvement" if report["step"] else "parent_retained",
                           training_wall_s=report["elapsed_s"], online_steps=report["actual_online_steps"],
                           capabilities=evidence["courses"], demonstration_episodes=len(report["demonstrations"]),
                           selection_executions=len(selection) * (1 + len(report["candidates"])),
                           driving_course_executions=40 * (1 + len(report["candidates"]))
                           + 5 * (1 + len(report["candidates"]) + sum("formal" in r for r in report["candidates"]))))
        print(kind, version, rounds[-1]["reason"], flush=True)
        if i < 3:
            pool = [Scenario(**r) for r in plan["candidates"][i]]
            query_reference_round(pool, plan["initial_indices"], parent, root / "random/queries",
                                  iteration=i + 1, execution_root=root, policy=ATTENTION)
            paid = paid_reference_records(root, plan, i + 1, rounds[-1]["sha256"])
            records.append(paid)
            historical = repair_subset(prior)
            current = [Scenario(**r["scenario"]) for r in paid if r["feedback"]["collision"]]
            prior.extend(current)
    freeze(root / "chain.json", dict(status="complete" if len(rounds) == 4 else "no_paid_failure",
                                     rounds=rounds, last_version=rounds[-1]["version"] if rounds else "h0"))
    return rounds, records


def model_pair(kind, source):
    # The two-stage encoder needs a context outside each training episode.
    # Only h0 is eligible at R0, so episodes alternate fixed disjoint h0
    # training folds. Inference retains all 500 records as one policy source.
    order = np.random.default_rng(SEED).permutation(len(source.scenarios))
    folds = [replace(source, source_id=f"h0_training_fold_{i}",
                     scenarios=[source.scenarios[j] for j in indices],
                     collision=source.collision[indices], safety=source.safety[indices],
                     rss_margin=source.rss_margin[indices])
             for i, indices in enumerate(np.split(order, 2))]
    directory = scene_root(kind, ATTENTION) / "response_model"
    directory.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "cache" / f"attention_model_{kind}.lock"
    with FileLock(lock):
        train(folds, directory, kernel_type=SharedPrivateKernel)
    models = [load_model(directory / f"{name}.pt", kernel_type=SharedPrivateKernel)
              for name in ("target_kernel", "historical_encoder")]
    if any(metadata["specification"]["history_fingerprints"] != archive_fingerprints(folds)
           for _, metadata in models):
        raise ValueError("Attention response model did not use its registered h0 folds")
    return [m for m, _ in models]


def collect_released_suite(kind, version, workers):
    """Collect this frozen version while the reference chain trains later ones."""
    bank = admitted_bank(kind)
    parent = H0
    for row in VERSIONS[1:VERSIONS.index(version) + 1]:
        directory = scene_root(kind, ATTENTION) / "reference" / row
        training = read_json(directory / "training_protocol.json")
        report = read_json(directory / "training.json")
        capability = verify_course_evidence(directory / "qualification")
        if (training["parent_sha256"] != checkpoint_identity(parent)
                or not capability["qualified"] or capability["courses"] != report["qualification"]["courses"]):
            raise ValueError("Only a qualified version in the registered lineage can enter the suite")
        if report["step"]:
            parent = directory / "policy.pt"
        if checkpoint_identity(parent) != report["checkpoint_sha256"]:
            raise ValueError("Released attention checkpoint differs from its qualification")
    destination = version_directory(kind, "truth", version, ATTENTION)
    jobs = [(parent, scene, destination / f"{i:06d}.json.gz", "fixed_suite", ATTENTION)
            for i, scene in enumerate(bank.candidates)]
    pending = [job for job in jobs if not job[2].exists()]
    if pending:
        collect_all(pending, workers, f"{kind} attention {version} fixed suite")


def run_scene(kind, workers, device, *, history_only=False):
    initialize_worker()
    if not (OUTPUT / "qualification/preflight.json").exists():
        raise ValueError("Run the attention regression preflight before starting scene workers")
    preflight()
    bank = admitted_bank(kind)
    plan = protocol(kind, bank, device)
    original = reference_history(kind, plan, workers)
    if history_only:
        return
    rounds, reference = repair_chain(kind, bank, plan, original)
    root = scene_root(kind, ATTENTION)
    e02 = read_json(PUBLIC / "budget_discovery/budget" / ATTENTION / kind / "11/full/result.json.gz")
    registration = freeze(root / "registration.json", dict(
        candidates=[asdict(s) for s in bank.candidates], bank_sha256=bank.snapshot_fingerprint,
        initial_indices=initial_indices(SUITE_SIZE), h0_collisions=sum(e02["evaluation"]["candidate_collisions"]),
        environment_seed=SEED, duration_s=episode_duration(kind), methods=METHODS,
        checkpoints={row["version"]: {"path": row["path"], "sha256": row["sha256"]} for row in (
            {"version": "h0", "path": H0.relative_to(ROOT).as_posix(), "sha256": plan["h0_sha256"]}, *rounds
        )},
    ))
    for version in registration["checkpoints"]:
        if version != "h0":
            collect_released_suite(kind, version, workers)
        truth_rows(kind, version, bank, registration, ATTENTION)
    if not rounds:
        audit_scene(kind, bank, registration, {}, rounds, reference, original)
        return
    history = [response_archive("h0", original, ATTENTION)]
    base, encoder = model_pair(kind, history[0])
    jobs, admitted = [], {}
    for i, row in enumerate(rounds[:3]):
        version = row["version"]
        admitted[version] = list(history)
        sources = source_snapshot(encoder, history, bank.candidates, root / "sources" / f"{version}.npz")
        for method in METHODS:
            jobs.append((kind, version, method, bank.candidates, registration["initial_indices"],
                         list(history) if method != "birchler" else [],
                         base if method == "full" else None, sources if method == "full" else None,
                         "cpu", 1, ATTENTION))
        if i < len(reference):
            history.append(response_archive(f"paid_policy_{i + 1}", reference[i], ATTENTION))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        for finished in pool.map(replay_job, jobs):
            print("Replay complete", *finished, flush=True)
    audit_scene(kind, bank, registration, admitted, rounds, reference, original)


def audit_scene(kind, bank, registration, admitted, rounds, reference, original):
    root = scene_root(kind, ATTENTION)
    versions, metrics, curves, costs, by_task = [], [], [], [], {}
    previous = None
    for version in registration["checkpoints"]:
        truth = truth_rows(kind, version, bank, registration, ATTENTION)
        labels = [r["collision"] for r in truth["feedback"]]
        versions.append(version_counts(kind, version, labels, previous))
        costs.append({"kind": kind, "version": version, "method": "physical_suite",
                      "logical_queries": 0, "executions": truth["new_physical_executions"],
                      "physical_s": truth["physical_execution_s"], "ranking_s": None, "adaptation_s": None,
                      "selector_wall_s": None})
        if version in admitted:
            by_task[kind, version] = {}
            for method in METHODS:
                directory = query_directory(kind, version, method, ATTENTION)
                result = read_json(directory / "result.json.gz")
                order = audit_ledger(directory, result, truth, registration, method, admitted[version])
                values = score(previous, labels, order)
                cumulative = values.pop("cumulative_collisions")
                row = dict(kind=kind, version=version, method=method, **values)
                metrics.append(row)
                by_task[kind, version][method] = row
                curves.extend(dict(kind=kind, version=version, method=method, query=i,
                                   collisions=count, recall=count / sum(labels) if sum(labels) else None)
                              for i, count in enumerate(cumulative, 1))
                stats = result["result"]
                costs.append({"kind": kind, "version": version, "method": method,
                              "logical_queries": len(order), "executions": 0, "physical_s": 0,
                              "ranking_s": stats.get("ranking_s"),
                              "adaptation_s": sum(q.get("adaptation_s", 0) for q in stats["queries"]),
                              "selector_wall_s": stats.get("resources", {}).get("wall_s")})
        previous = labels
    contrasts, _ = compare(by_task, methods=METHODS)
    write_json(root / "audit.json", dict(kind=kind, status="complete" if len(rounds) == 4 else "no_paid_failure",
        versions=versions, metrics=metrics, contrasts=contrasts, costs=costs, repairs=rounds,
        history_records=len(original), history_physical_s=sum(r["feedback"]["execution_s"] for r in original),
        reference_queries=[len(r) for r in reference],
        reference_physical_s=[sum(r["feedback"]["execution_s"] for r in records) for records in reference],
        spectre_sortings_rebuilt=sum(r["method"] == "spectre" for r in metrics),
        curves=curves))


def report():
    from statistics import median

    rows = [read_json(scene_root(kind, ATTENTION) / "audit.json") for kind in FORMAL_SCENARIOS]
    tables = {key: [r for s in rows for r in s[key]] for key in ("versions", "metrics", "contrasts", "costs", "curves")}
    for row in tables["costs"]:
        resource = {}
        if row["method"] in METHODS:
            directory = query_directory(row["kind"], row["version"], row["method"], ATTENTION)
            selector = read_json(directory / "result.json.gz")["result"]
            resource = selector.get("resources", {})
            if row["ranking_s"] is None:
                row["ranking_s"] = sum(q.get("selection_s", 0) for q in selector["queries"])
        row.update(device=resource.get("device"), sampled_peak_cpu_bytes=resource.get("sampled_peak_cpu_bytes"),
                   peak_cuda_allocated_bytes=resource.get("peak_cuda_allocated_bytes"))
    repairs = []
    for scene in rows:
        kind = scene["kind"]
        for repair in scene["repairs"]:
            path = scene_root(kind, ATTENTION) / "reference" / repair["version"] / "training.json"
            training = read_json(path)
            courses = [*training["parent_skills"], *training["qualification"]["courses"]]
            for candidate in training["candidates"]:
                courses.extend(candidate["development_courses"]["courses"])
                courses.extend(candidate["skills"])
                courses.extend(candidate.get("formal", {}).get("courses", []))
            if len(courses) != repair["driving_course_executions"]:
                raise ValueError("Driving course costs disagree with the recorded repair checks")
            repairs.append({
                "kind": kind, **{k: v for k, v in repair.items() if k != "capabilities"},
                "driving_course_simulated_s": sum(c["physics_ticks"] for c in courses) / PHYSICS_HZ,
                "demonstration_simulated_s": sum(c["physics_ticks"] for c in training["demonstrations"]) / PHYSICS_HZ,
            })
    tables["repair_costs"] = repairs
    tables["reference_costs"] = [
        {"kind": s["kind"], "history_records": s["history_records"], "history_physical_s": s["history_physical_s"],
         "reference_queries": sum(s["reference_queries"]), "reference_physical_s": sum(s["reference_physical_s"])}
        for s in rows
    ]
    tables["contrasts"] = [
        {**r, "q90_queries_saved": -r["q90_ours_minus_control"] if r["q90_ours_minus_control"] is not None else None}
        for r in tables["contrasts"]
    ]
    for key, table in tables.items():
        if table:
            write_csv(OUTPUT / "reports" / f"{key}.csv", table)
    summary = {}
    for method in METHODS[1:]:
        pair = [r for r in tables["contrasts"] if r["comparator"] == method]
        savings = [-r["q90_ours_minus_control"] for r in pair if r["q90_ours_minus_control"] is not None]
        summary[method] = dict(paired=len(savings), median_queries_saved=median(savings) if savings else None,
                               win_tie_loss=[sum(r["q90_outcome"] == category for r in pair)
                                             for category in ("win", "tie", "loss")])
    if all(r["status"] == "complete" for r in rows):
        expected = {(kind, version, method) for kind in FORMAL_SCENARIOS
                    for version in VERSIONS[1:4] for method in METHODS}
        actual = {(r["kind"], r["version"], r["method"]) for r in tables["metrics"]}
        if (actual != expected or len(tables["metrics"]) != 60 or len(tables["versions"]) != 20
                or len(repairs) != 16 or sum(r["executions"] for r in tables["costs"]) != 32000
                or sum(r["online_steps"] for r in repairs) != 2097152
                or sum(s["spectre_sortings_rebuilt"] for s in rows) != 12):
            raise ValueError("Complete attention regression does not match its registered scale")
    write_json(OUTPUT / "reports/summary.json", dict(
        status="complete" if len(tables["metrics"]) == 60 else "partial_stop_at_paid_failure_gate",
        replays=len(tables["metrics"]), versions=len(tables["versions"]), auc_window=AUC_WINDOW,
        new_collision_window=NEW_WINDOW, h0_reused_records=8000,
        spectre_sortings_rebuilt=sum(s["spectre_sortings_rebuilt"] for s in rows),
        qualified_releases=len(tables["repair_costs"]),
        releases_with_new_weights=sum(r["selected_step"] > 0 for r in tables["repair_costs"]),
        training_environment_steps=sum(r["online_steps"] for r in tables["repair_costs"]),
        physical_h0_requalification_executions=preflight()["physical_executions"],
        new_suite_records=sum(r["executions"] for r in tables["costs"]),
        replay_queries=sum(r["logical_queries"] for r in tables["metrics"]), pairwise=summary,
        tables_sha256={f"{k}.csv": sha(OUTPUT / "reports" / f"{k}.csv") for k, table in tables.items() if table},
        scene_audits={r["kind"]: sha(scene_root(r["kind"], ATTENTION) / "audit.json") for r in rows},
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "history", "model", "scene", "suite", "report"))
    parser.add_argument("--kind", choices=FORMAL_SCENARIOS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--version", choices=VERSIONS[1:], action="append")
    args = parser.parse_args()
    if args.stage == "report":
        report()
    elif args.stage == "preflight":
        initialize_worker()
        preflight()
    elif not args.kind:
        parser.error("The history, model, scene and suite stages require --kind")
    elif args.stage == "model":
        initialize_worker()
        rows = collect_h0(args.kind, args.workers, args.device)
        model_pair(args.kind, response_archive("h0", rows, ATTENTION))
    elif args.stage == "suite":
        if not args.version:
            parser.error("The suite stage requires a released --version")
        initialize_worker()
        preflight()
        for version in args.version:
            collect_released_suite(args.kind, version, args.workers)
    else:
        run_scene(args.kind, args.workers, args.device, history_only=args.stage == "history")


if __name__ == "__main__":
    main()
