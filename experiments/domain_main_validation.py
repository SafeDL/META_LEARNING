"""Paid four-cell first runs and an independent reconstruction of global acquisition."""

import copy
import time

import numpy as np
import torch

from controllers.highway.catalog import BY_ID
from testing_methods.common.fsbo.method import load as load_target
from testing_methods.common.gp_classifier import ClassificationKernel, fit_target
from testing_methods.history_transfer.domain_build import OUTPUT as MODELS
from testing_methods.history_transfer.domain_build import load_sources
from testing_methods.history_transfer.domain_discovery import MODES, aggregate_history, select

from .collision_metrics import collision_recall, summarize
from .construction import load_construction
from .discovery_config import write_json
from .executions import ExecutionBank, execution_contract
from .inference_reference import probabilities_numpy
from .initialization import initial_indices
from .protocol import DEFAULT, ROOT, TEST_RECORDS_ROOT
from .query import QueryLedger, run_isolated
from .scenarios import features
from .source_registration import specification_matches

OUTPUT = ROOT / "experiment_outputs/gcrt/collision/discovery/domain_comparison/stage_a"
CHECKS = (9, 32, 64, 128)


def reconstruct(candidates, base, sources, rows, mode, initial, *, device="cpu"):
    """Rebuild each audited state from the initial weights and previously paid Booleans."""
    x = features(candidates)
    physical = ClassificationKernel(x.shape[1]).to(device)
    target = copy.deepcopy(base).to(device).double()
    feedback_enabled = mode.endswith("F1")
    indices = [row["index"] for row in rows] if feedback_enabled else []
    labels = [row["feedback"]["collision"] for row in rows] if feedback_enabled else []
    if indices:
        fit_target(target, x[indices], labels, steps=10, learning_rate=0.003)
        fit_target(physical, x[indices], labels, steps=10, learning_rate=0.01)
    with torch.no_grad():
        mean = (
            torch.tensor(sources["means"], dtype=torch.float64, device=device)
            + target.mean
            - base.mean.to(device)
        )
        covariance = (
            torch.tensor(sources["covariances"], dtype=torch.float64, device=device)
            + target.covariance(x)[None]
        )
        mean = torch.cat((mean, target.mean.expand(1, len(x)), physical.mean.expand(1, len(x))))
        covariance = torch.cat(
            (covariance, target.covariance(x)[None], physical.covariance(x)[None])
        )
        probabilities = torch.tensor(
            probabilities_numpy(mean.cpu().numpy(), covariance.cpu().numpy(), indices, labels),
            dtype=torch.float64,
        )
        log_weights = torch.tensor(np.r_[0.75 * sources["weights"], 0.25]).log()
        if feedback_enabled:
            for row in rows:
                p = torch.tensor(row["prequery_component_probabilities"][:-1], dtype=torch.float64)
                log_weights += (p if row["feedback"]["collision"] else 1 - p).clamp_min(1e-15).log()
        weights = torch.cat(
            (0.9 * log_weights.softmax(0), torch.tensor([0.1], dtype=torch.float64))
        )
        combined = (weights[:, None] * probabilities).sum(0).numpy()
    remaining = set(range(len(x))) - {row["index"] for row in rows}
    maximum = max(combined[i] for i in remaining)
    index = (
        initial[len(rows)]
        if len(rows) < len(initial)
        else min(
            (i for i in remaining if combined[i] >= maximum - 1e-12),
            key=lambda i: candidates[i].scenario_id,
        )
    )
    return index, combined, weights.numpy(), probabilities.numpy()


def main():
    torch.set_num_threads(1)
    kind, target, seed = "cut_in", "development_idm_mobil", 11
    _, development = load_construction(kind)
    archive = next(a for a in development if a.source_id == target)
    bank = ExecutionBank(TEST_RECORDS_ROOT / "physical/development" / kind / target)
    if bank.candidates != archive.scenarios or not specification_matches(
        bank.manifest["contract"], execution_contract(BY_ID[target])
    ):
        raise ValueError("Main-method validation requires its original qualified development bank")
    base, _ = load_target(MODELS / kind / "target_kernel.pt")
    raw = load_sources(kind)
    initial = initial_indices(len(bank.candidates), DEFAULT.initial_count, seed)
    checks = []
    for mode in MODES:
        sources = aggregate_history(raw) if mode.startswith("H0") else raw
        ledger = QueryLedger(bank.candidates, DEFAULT.budget, bank)
        started = time.perf_counter()
        result = run_isolated(
            select,
            {
                "candidates": bank.candidates,
                "initial_indices": initial,
                "pretrained": base,
                "sources": sources,
                "mode": mode,
                "budget": DEFAULT.budget,
            },
            ledger,
        )
        wall_s = time.perf_counter() - started
        write_json(
            OUTPUT / f"{mode}_pending_audit.json",
            {"result": result, "query_ledger": ledger.records},
        )
        errors = []
        for query in CHECKS:
            row = result["queries"][query - 1]
            index, probability, weights, components = reconstruct(
                bank.candidates, base, sources, result["queries"][: query - 1], mode, initial
            )
            error = max(
                abs(probability[index] - row["prequery_probability"]),
                np.max(abs(weights - row["prequery_component_weights"])),
                np.max(abs(components[:, index] - row["prequery_component_probabilities"])),
            )
            if index != row["index"] or error > 1e-7:
                raise ValueError(
                    f"Independent {mode} acquisition differs at query {query}: {error}; "
                    f"index {index} vs {row['index']}; "
                    f"weight_error {np.max(abs(weights - row['prequery_component_weights']))}; "
                    f"component_error {np.max(abs(components[:, row['index']] - row['prequery_component_probabilities']))}"
                )
            errors.append({"query": query, "index": index, "maximum_error": float(error)})
            if torch.cuda.is_available():
                gpu_index, gpu_probability, _, _ = reconstruct(
                    bank.candidates,
                    base,
                    sources,
                    result["queries"][: query - 1],
                    mode,
                    initial,
                    device="cuda",
                )
                device_error = float(np.max(abs(gpu_probability - probability)))
                if gpu_index != index or device_error > 1e-7:
                    raise ValueError(
                        f"CPU/CUDA {mode} acquisition differs at query {query}: {device_error}"
                    )
                errors[-1]["cpu_cuda_full_pool_probability_error"] = device_error
        write_json(
            OUTPUT / f"{mode}.json",
            {
                "stage": "A mechanism validation; not the frozen development comparison",
                "target_id": target,
                "scenario": kind,
                "algorithm_seed": seed,
                "initial_indices": initial,
                "result": result,
                "query_ledger": ledger.records,
                "metrics": {
                    **summarize(bank.candidates, result["queries"], ledger.trajectories),
                    **collision_recall(
                        bank.candidates, archive.collision.tolist(), result["queries"]
                    ),
                },
                "independent_acquisition_checks": errors,
                "cost": {"selector_and_replay_wall_s": wall_s, "new_physical_calls": 0},
            },
        )
        checks.append({"mode": mode, "queries": len(ledger.records), "acquisition_checks": errors})
        print(mode, "passed", "wall_s", round(wall_s, 2), flush=True)
    report = {
        "status": "passed",
        "cells": checks,
        "equivalent_queries": 4 * DEFAULT.budget,
        "new_physical_calls": 0,
        "formal_comparison_frozen": False,
        "H0_teacher": "same historical mixture latent mean and total covariance; single Gaussian",
    }
    write_json(OUTPUT / "audit.json", report)
    return report


if __name__ == "__main__":
    main()
