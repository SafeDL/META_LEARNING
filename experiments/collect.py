"""Qualified new response acquisition; final target banks require protocol freezing."""

import json

import numpy as np

from controllers.highway.catalog import SUTS
from controllers.highway.learning import CHECKPOINT_ROOT
from testing_methods.common.response import ResponseArchive

from .executions import collect_bank, require_qualified
from .protocol import DEFAULT, TEST_RECORDS_ROOT
from .scenarios import DOMAINS, sample
from .storage import load_archive, save_archive


def require_learning_freeze():
    protocol = json.loads((CHECKPOINT_ROOT / "training_protocol.json").read_text(encoding="utf-8"))
    for sut in SUTS:
        require_qualified(sut)
        if sut.family == "learning":
            report = json.loads(
                (CHECKPOINT_ROOT / sut.training_run / "training.json").read_text(encoding="utf-8")
            )
            if (
                report["configuration"] != protocol["configuration"]
                or report["source_contract"] != protocol["source_contract"]
                or report["actual_timesteps"] < protocol["configuration"]["total_timesteps"]
            ):
                raise ValueError(
                    "All five learning runs must follow the pre-target training protocol"
                )
    return protocol


def collect_history():
    require_learning_freeze()
    for role in ("history", "development"):
        for kind in DOMAINS:
            scenarios = sample(kind, DEFAULT.candidate_count, role)
            for sut in SUTS:
                if sut.role != role:
                    continue
                bank = collect_bank(
                    sut.sut_id,
                    scenarios,
                    TEST_RECORDS_ROOT / "physical" / role / kind / sut.sut_id,
                    environment_seed=DEFAULT.environment_seeds[0],
                )
                executions = [bank(s) for s in scenarios]
                provenance = {
                    "qualified": True,
                    "bank": str(bank.root.relative_to(TEST_RECORDS_ROOT)),
                    "contract": bank.manifest["contract"],
                    "environment_seed": DEFAULT.environment_seeds[0],
                    "physical_calls": bank.manifest["attempted_calls"],
                }
                archive = ResponseArchive(
                    sut.sut_id,
                    sut.family,
                    sut.lineage,
                    role,
                    scenarios,
                    np.asarray([e.feedback.collision for e in executions]),
                    np.array(
                        [
                            [e.feedback.safety.min_clearance_m, e.feedback.safety.min_ttc_s]
                            for e in executions
                        ]
                    ),
                    provenance=provenance,
                )
                destination = TEST_RECORDS_ROOT / role / kind / f"{sut.sut_id}.npz"
                if destination.exists():
                    existing = load_archive(destination)
                    if (
                        existing.provenance != provenance
                        or not np.array_equal(existing.collision, archive.collision)
                        or existing.scenarios != scenarios
                    ):
                        raise ValueError("Existing archive differs from its real execution bank")
                else:
                    save_archive(destination, archive, provenance=provenance)


if __name__ == "__main__":
    collect_history()
