"""Seal physical responses and disclose only individually paid replay feedback."""

import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

from testing_methods.baselines.birchler_greedy.method import prioritize as greedy
from testing_methods.baselines.detour.method import prioritize as detour
from testing_methods.baselines.sdc_scissor import method as sdc_scissor
from testing_methods.baselines.spectre import method as spectre
from testing_methods.common.response import archive_fingerprints
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel
from testing_methods.history_transfer.training import load_model, train

from experiments.costs import PeakMemory, measure_resources
from experiments.regression.fixed_inputs import AUC_WINDOW, BRAKING_KIND, DQN_H0_COUNTS, H0_COUNTS, VERSIONS, methods_for, model_root, scene_root, query_directory, version_directory
from experiments.history import build_source
from experiments.history_transfer_experiment import freeze
from experiments.parallel import initialize_worker, prepare_processes
from experiments.protocol import episode_duration
from experiments.query import run_isolated
from experiments.regression.audit import verify_execution
from experiments.regression.execution import ATTENTION, response_archive
from experiments.regression.preflight import DQN_CONTINUAL
from experiments.resume import paid_feedback_hash
from experiments.schema import QueryFeedback
from experiments.source_registration import source_contract_matches
from experiments.storage import compact_result, read_json, write_json
from experiments.transfer_settings import native_settings


def target_count(collisions):
    return (9 * collisions + 9) // 10


def truth_rows(kind, version, bank, registration, policy="ppo"):
    path = version_directory(kind, "truth", version, policy)
    checkpoint = registration["checkpoints"][version]["sha256"]
    rows, hashes = [], hashlib.sha256()
    physical_s = 0.0
    if version == "h0":
        for index, scene in enumerate(bank.candidates):
            row = read_json(bank.execution_path(index))
            if (row["scenario"] != asdict(scene) or row["call"] != index + 1
                    or row["feedback"]["scenario_id"] != scene.scenario_id
                    or row["feedback"]["collision"] != any(f["collision"] for f in row["trajectory"])):
                raise ValueError("The original native record disagrees with its sealed suite")
            rows.append(row["feedback"])
        physical_s = (sum(r["execution_s"] for r in rows) if policy == ATTENTION
                      else bank.manifest["physical_execution_s"])
        digest = bank.snapshot_fingerprint
    else:
        for index, scene in enumerate(bank.candidates):
            execution = path / f"{index:06d}.json.gz"
            row = read_json(execution)
            verify_execution(row, asdict(scene), checkpoint)
            time_s = row["trajectory"][-1]["time_s"]
            if (row["purpose"] != "fixed_suite" or time_s > episode_duration(kind) + 1e-9
                    or (time_s < episode_duration(kind) - 1e-9 and not (
                        row["feedback"]["collision"] or row["feedback"]["safety"]["violations"]))):
                raise ValueError(f"Invalid native horizon or execution purpose in {execution}")
            rows.append(row["feedback"])
            hashes.update(hashlib.sha256(execution.read_bytes()).digest())
            physical_s += row["feedback"]["execution_s"]
        digest = hashes.hexdigest()
    collisions = sum(row["collision"] for row in rows)
    h0_counts = H0_COUNTS if policy == "ppo" else DQN_H0_COUNTS
    if version == "h0":
        if policy == ATTENTION:
            expected = registration["h0_collisions"]
        elif policy == DQN_CONTINUAL:
            expected = read_json(bank.root / "quality.json")["profile"]["collisions"]
        else:
            expected = h0_counts[kind]
        if collisions != expected:
            raise ValueError(f"Unexpected h0 starting collision count for {kind}")
    return freeze(path / "sealed.json", {
        "kind": kind, "version": version, "checkpoint_sha256": checkpoint,
        "execution_sha256": digest, "collisions": collisions, "physical_execution_s": physical_s,
        "reused_physical_executions": len(rows) if version == "h0" else 0,
        "new_physical_executions": 0 if version == "h0" else len(rows), "feedback": rows,
    })


class FixedReplayLedger:
    """Stop at the registered evaluator threshold and persist one receipt per reveal."""

    def __init__(self, truth, initial, directory, specification, *, window=AUC_WINDOW):
        self._truth = truth
        self._initial = initial
        self._directory = directory
        self._window = window
        self._target = target_count(sum(row["collision"] for row in truth))
        self._seen = set()
        self._collisions = 0
        self.budget = len(truth)
        self.records = []
        freeze(directory / "specification.json", specification)
        self._pending = {int(p.stem) for p in directory.glob("paid/*.json")}

    def register(self, _):
        raise ValueError("Regression replays use only the frozen suite")

    def query(self, index, *, purpose):
        step = len(self.records)
        if (type(index) is not int or index in self._seen or not 0 <= index < len(self._truth)
                or step >= self.budget):
            raise ValueError("Invalid, duplicated or unpaid regression query")
        initial = step < len(self._initial)
        if purpose != ("initial" if initial else "target") or (initial and index != self._initial[step]):
            raise ValueError("Selector diverged from the common initialization")
        feedback = QueryFeedback.from_dict(self._truth[index])
        value = feedback.to_dict()
        row = {"call": step + 1, "index": index, "scenario_id": value["scenario_id"],
               "feedback": value, "status": "complete", "purpose": purpose, "account": "target"}
        path = self._directory / "paid" / f"{step + 1:04d}.json"
        if step + 1 in self._pending:
            self._pending.remove(step + 1)
            if read_json(path) != json.loads(json.dumps(row)):
                raise ValueError("Selector diverged from its saved paid prefix")
        else:
            if self._pending:
                raise ValueError("Saved paid prefix has a gap")
            write_json(path, row)
        self.records.append(row)
        self._seen.add(index)
        self._collisions += value["collision"]
        if step + 1 >= self._window and self._collisions >= self._target:
            self.budget = step + 1
        if step > 0 and (step + 1) % 100 == 0:
            print(self._directory.parents[2].name, self._directory.parent.name,
                  self._directory.name, step + 1, "charges", flush=True)
        return feedback

    def finish(self):
        if self._pending or len(self.records) != self.budget:
            raise ValueError("The replay did not reach the registered evaluator stop")


@measure_resources
def fixed_selector(candidates, oracle, *, method, initial_indices, pretrained=None,
                   sources=None, history=None, device="cpu", cpu_threads=1):
    initialize_worker()
    if method == "full":
        return select(candidates, oracle, pretrained=pretrained, sources=sources,
                      initial_indices=initial_indices, budget=len(candidates), snapshot_counts=(),
                      online_settings=native_settings().online, device=device, implicit_evidence=True,
                      cpu_threads=cpu_threads)
    if method == "sdc_scissor":
        return sdc_scissor.select(candidates, oracle, history=history, initial_indices=initial_indices)
    indices, feedback = [], []

    def query(index, purpose):
        value = oracle.query(index, purpose=purpose)
        indices.append(index)
        feedback.append({"query": len(indices), "index": index, "feedback": value.to_dict()})
        return value

    observed = [(i, query(i, "initial")) for i in initial_indices]
    started = time.perf_counter()
    if method == "detour":
        order = detour(history, candidates, observed, len(candidates) - len(observed))
    elif method == "birchler":
        order = greedy(candidates, initial_indices)
    elif method == "spectre":
        order = spectre.prioritize(candidates, history, observed)
    else:
        raise ValueError("Unknown fixed-suite method")
    ranking_s = time.perf_counter() - started
    for index in order:
        if oracle.stopped:
            break
        query(index, "target")
    result = {"method": method, "queries": feedback, "selected_indices": indices, "ranking_s": ranking_s}
    if method == "spectre":
        result["ranking_sha256"] = spectre.order_digest(order)
    return result


def model_paths(kind, history, policy="ppo"):
    directory = model_root(kind, policy)
    if policy != "ppo" or kind == BRAKING_KIND:
        train(history, directory, kernel_type=SharedPrivateKernel)
    models = [load_model(directory / f"{name}.pt", kernel_type=SharedPrivateKernel)
              for name in ("target_kernel", "historical_encoder")]
    for (model, metadata), part in zip(models, ("target", "historical_encoder")):
        if (metadata["part"] != part or metadata["step"] != 1500
                or metadata["specification"]["history_fingerprints"] != archive_fingerprints(history)
                or model.construction_specification["scenario"] != kind):
            raise ValueError("Response model was not built from admitted policy ancestry")
    return [model for model, _ in models]


def replay_job(job):
    kind, version, method, candidates, initial, history, base, sources, device, cpu_threads, policy = job
    truth = read_json(version_directory(kind, "truth", version, policy) / "sealed.json")
    directory = query_directory(kind, version, method, policy)
    path = directory / "result.json.gz"
    spec = {"method": method, "version": version, "checkpoint_sha256": truth["checkpoint_sha256"],
            "initial_indices": initial, "truth_sha256": truth["execution_sha256"],
            "history": archive_fingerprints(history) if method != "birchler" else {},
            "source_snapshot": sources["specification"] if sources is not None else None}
    if method == "sdc_scissor":
        spec["selector_settings"] = sdc_scissor.SETTINGS
    elif method == "spectre":
        spec["selector_settings"] = spectre.SETTINGS
    if path.exists():
        if read_json(path)["specification"] != spec:
            raise ValueError("The completed replay belongs to a different fixed task")
        return kind, version, method
    ledger = FixedReplayLedger(truth["feedback"], initial, directory, spec)
    public = {"candidates": candidates, "method": method, "initial_indices": initial}
    if method == "full":
        public.update(pretrained=base, sources=sources, device=device, cpu_threads=cpu_threads)
    elif method in ("detour", "sdc_scissor"):
        public["history"] = [(s, bool(y)) for a in history for s, y in zip(a.scenarios, a.collision)]
    elif method == "spectre":
        public["history"] = history
    result = run_isolated(fixed_selector, public, ledger, signal_evaluator_stop=True)
    ledger.finish()
    write_json(path, {"specification": spec, "query_ledger": ledger.records,
                      "paid_feedback_hash": paid_feedback_hash(ledger.records),
                      "result": compact_result(result), "new_physical_calls": 0,
                      "evidence_gradient": "implicit_at_converged_mode" if method == "full" else None})
    return kind, version, method


def source_snapshot(encoder, history, candidates, path):
    if not path.exists():
        return build_source(encoder, history, candidates, cache_path=path)
    fresh = build_source(encoder, history, candidates)
    with np.load(path, allow_pickle=False) as archive:
        spec = json.loads(str(archive["specification"]))
        prior = spec["source_contract"]
        current = fresh["specification"]["source_contract"]
        if (prior["packages"] != current["packages"]
                or not source_contract_matches(prior["source_files"], current["source_files"])
                or {**fresh["specification"], "source_contract": prior} != spec
                or any(not np.array_equal(fresh[k], archive[k]) for k in ("means", "blocks", "weights"))):
            raise ValueError("Cached source predictions changed during fixed-suite replay recovery")
    return {**fresh, "specification": spec}


def replay(kinds, workers, versions=VERSIONS[1:4], methods=None, device="cpu", cpu_threads=1,
           *, policy="ppo"):
    from experiments.regression.run_fixed import prepare

    methods = methods_for(policy) if methods is None else methods
    if set(methods) - set(methods_for(policy)):
        raise ValueError(f"Unregistered method requested for {policy}")
    prepare_processes()
    jobs = []
    for kind in kinds:
        if policy != "ppo" and not (scene_root(kind, policy) / "registration.json").exists():
            print(kind, "reference incomplete", flush=True)
            continue
        bank, history, models, records = prepare(kind, workers, policy=policy)
        if len(models) == 1:
            continue
        root = scene_root(kind, policy)
        registry = read_json(root / "registration.json")
        base, encoder = model_paths(kind, history, policy) if "full" in methods else (None, None)
        for i, version in enumerate(VERSIONS[1:4]):
            if version not in models:
                break
            if version in versions:
                if not (version_directory(kind, "truth", version, policy) / "sealed.json").exists():
                    truth_rows(kind, version, bank, registry, policy)
                sources = None
                if "full" in methods:
                    with PeakMemory() as resources:
                        sources = source_snapshot(encoder, history, bank.candidates,
                                                  root / "sources" / f"{version}.npz")
                    construction = root / "sources" / f"{version}.json"
                    if not construction.exists():
                        write_json(construction, resources.result)
                for method in methods:
                    jobs.append((kind, version, method, bank.candidates, registry["initial_indices"],
                                 list(history) if method != "birchler" else [],
                                 base if method == "full" else None, sources if method == "full" else None,
                                 device if method == "full" else "cpu", cpu_threads, policy))
            if i < len(records):
                history.append(response_archive(f"paid_policy_{i+1}", records[i], policy))
    if not jobs:
        print(f"No qualified {policy.upper()} tasks for the requested replay", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        futures = {pool.submit(replay_job, job): job[:3] for job in jobs}
        failures = []
        for future in as_completed(futures):
            try:
                print("replay complete", *future.result(), flush=True)
            except Exception as error:
                kind, version, method = futures[future]
                write_json(query_directory(kind, version, method, policy) / "failure.json",
                           {"kind": kind, "version": version, "method": method, "error": repr(error)})
                failures.append((kind, version, method))
                print("replay failed", kind, version, method, repr(error), flush=True)
        if failures:
            raise RuntimeError(f"Incomplete fixed-suite replays: {failures}")
