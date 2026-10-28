"""Read-only Stage A audit of the history/development banks for the new comparison."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from controllers.highway.catalog import BY_ID

from .collision_metrics import collision_recall
from .construction import load_archives
from .discovery_config import write_json
from .domain_risk import disclose_impact
from .executions import ExecutionBank, execution_contract
from .protocol import DEFAULT, ROOT, TEST_RECORDS_ROOT
from .source_registration import specification_matches

OUTPUT = ROOT / "experiment_outputs/benchmark/collision/discovery/domain_comparison"


def audit_archive(archive):
    kind = archive.scenarios[0].kind
    root = TEST_RECORDS_ROOT / "physical" / archive.role / kind / archive.source_id
    bank = ExecutionBank(root)
    if (
        bank.candidates != archive.scenarios
        or len(bank.candidates) != DEFAULT.candidate_count
        or bank.manifest["environment_seed"] != DEFAULT.environment_seeds[0]
        or not specification_matches(
            bank.manifest["contract"], execution_contract(BY_ID[archive.source_id])
        )
    ):
        raise ValueError(f"Bank contract differs from the allowed archive: {root}")
    labels, impacts = [], []
    censored = 0
    for index, scene in enumerate(bank.candidates):
        raw = bank(scene)
        feedback = disclose_impact(raw).feedback
        if feedback.collision != bool(archive.collision[index]) or not np.array_equal(
            [feedback.safety.min_clearance_m, feedback.safety.min_ttc_s], archive.safety[index]
        ):
            raise ValueError(f"Paid execution differs from construction at {root}, {index}")
        labels.append(feedback.collision)
        if feedback.collision:
            impacts.append(feedback.safety.impact_delta_v_mps)
        else:
            censored += feedback.safety.min_ttc_s > DEFAULT.episode_duration_s
    denominator = collision_recall(bank.candidates, labels, [])["pool_unique_collisions"]
    return {
        "source_id": archive.source_id,
        "role": archive.role,
        "scenario": kind,
        "lineage": archive.lineage,
        "status": "passed",
        "candidates": len(labels),
        "collision_executions": sum(labels),
        "pool_unique_collisions": denominator,
        "noncollision_ttc_censored": censored,
        "impact_count": len(impacts),
        "minimum_impact_delta_v_mps": min(impacts, default=None),
        "maximum_impact_delta_v_mps": max(impacts, default=None),
        "path": str(root.relative_to(ROOT)),
    }


def main():
    started = time.perf_counter()
    archives = []
    for kind in ("cut_in", "cut_out"):
        history, development = load_archives(kind)
        if len(history) != 8 or len(development) != 4:
            raise ValueError(
                "The new comparison requires eight history and four development objects"
            )
        archives.extend((*history, *development))
    with ThreadPoolExecutor(max_workers=4) as pool:
        banks = list(pool.map(audit_archive, archives))
    report = {
        "status": "passed",
        "banks": banks,
        "checked_executions": sum(row["candidates"] for row in banks),
        "new_physical_calls": 0,
        "supplement_needed": 0,
        "supplement_limit": 2048,
        "independent_targets_read": False,
        "full_comparison_frozen": False,
        "wall_s": time.perf_counter() - started,
    }
    write_json(OUTPUT / "data_audit.json", report)
    print({k: v for k, v in report.items() if k != "banks"}, flush=True)
    return report


if __name__ == "__main__":
    main()
