"""Audit completed diagonal teachers with canonical serialized feedback."""

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from testing_methods.history_transfer.selection import aggregate_history

from .analyze_controls import OUTPUT, report, run_path
from .analyze_covariance import diagonal_sources
from .collision_metrics import collision_recall, summarize
from .construction import load_archives
from .discovery_config import development_tasks, read_json, write_json
from .domain_risk import disclose_driving_quality, disclose_impact
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import ROOT, TEST_RECORDS_ROOT
from .recorded_models import DEVELOPMENT_MODELS as MODELS
from .recorded_models import load_development_sources as load_sources
from .recorded_models import load_model as load_target
from .schema import QueryFeedback
from .verify_selection import reconstruct


def audit_group(target_id, kind):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        bank = ExecutionBank(TEST_RECORDS_ROOT / "physical/development" / kind / target_id)
        executions = [disclose_driving_quality(disclose_impact(bank(s))) for s in bank.candidates]
        _, development = load_archives(kind)
        archive = next(a for a in development if a.source_id == target_id)
        base, _ = load_target(MODELS / kind / "target_kernel.pt")
        sources = aggregate_history(diagonal_sources(load_sources(kind)))
        rows = []
        digest = hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest()
        for task in [
            t for t in development_tasks() if t["target_id"] == target_id and t["scenario"] == kind
        ]:
            path = run_path(task)
            saved = read_json(path)
            result = saved["result"]
            queries, ledger = result["queries"], saved["query_ledger"]
            initial = initial_indices(512, 8, task["algorithm_seed"])
            if (
                saved["specification"]
                != {"task": task, "method": "H0F1", "protocol_sha256": digest}
                or len(queries) != 128
                or len(ledger) != 128
                or result["actual_queries"] != 128
                or result["selected_indices"] != [q["index"] for q in queries]
                or len(set(result["selected_indices"])) != 128
                or result["selected_indices"][:8] != initial
            ):
                raise ValueError(
                    "A diagonal teacher differs from its frozen task and common budget"
                )
            for number, (query, record) in enumerate(zip(queries, ledger), 1):
                expected = executions[query["index"]].feedback.to_dict()
                if (
                    QueryFeedback.from_dict(query["feedback"]).to_dict() != expected
                    or record["feedback"] != query["feedback"]
                    or record["call"] != number
                    or record["index"] != query["index"]
                    or record["status"] != "complete"
                    or record["purpose"] != ("initial" if number <= 8 else "target")
                ):
                    raise ValueError(
                        "The canonical paid feedback or ledger differs from the physical trace"
                    )
            metrics = {
                **summarize(
                    bank.candidates, queries, [executions[q["index"]].trajectory for q in queries]
                ),
                **collision_recall(bank.candidates, archive.collision.tolist(), queries),
            }
            if json.loads(json.dumps(metrics)) != saved["metrics"]:
                raise ValueError(
                    "The diagonal-teacher metrics differ from the complete collision denominator"
                )
            for number in (9, 64, 128):
                row = queries[number - 1]
                index, probability, weights, components = reconstruct(
                    bank.candidates, base, sources, queries[: number - 1], "H0F1", initial
                )
                if index != row["index"]:
                    raise ValueError("The independent full-pool diagonal acquisition differs")
                np.testing.assert_allclose(
                    probability[index], row["prequery_probability"], atol=1e-7
                )
                np.testing.assert_allclose(
                    components[:, index], row["prequery_component_probabilities"], atol=1e-7
                )
                np.testing.assert_allclose(weights, row["prequery_component_weights"], atol=1e-12)
            rows.append(
                {
                    **task,
                    "method": "H0F1",
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    **saved["metrics"],
                    "independent_acquisition_checks": [9, 64, 128],
                }
            )
        return rows


def main():
    groups = sorted({(t["target_id"], t["scenario"]) for t in development_tasks()})
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(audit_group, target, kind) for target, kind in groups]
        for future in as_completed(futures):
            rows.extend(future.result())
            print("diagonal_audit", len(rows), "/ 40", flush=True)
    write_json(OUTPUT / "run_index.json", {"status": "passed", "runs": rows})
    write_json(
        OUTPUT / "audit.json",
        {
            "status": "passed",
            "runs": len(rows),
            "queries": 128 * len(rows),
            "independent_acquisition_checks": 3 * len(rows),
            "source_sha256": hashlib.sha256(
                (ROOT / "experiments/audit_controls.py").read_bytes()
            ).hexdigest(),
            "serialization_diagnosis": "the original post-save audit compared JSON lists to native tuples; canonical QueryFeedback restores types",
            "original_selector_protocol_unchanged": True,
            "reexecuted_queries": 0,
            "new_physical_calls": 0,
        },
    )
    return report()


if __name__ == "__main__":
    main()
