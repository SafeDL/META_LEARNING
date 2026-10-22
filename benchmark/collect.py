"""Qualified new response acquisition; final target banks require protocol freezing."""

import argparse
import json

import numpy as np

from controllers.highway.catalog import SUTS
from controllers.highway.learning import CHECKPOINT_ROOT
from methods.common.response import ResponseArchive

from .executions import collect_bank, require_qualified
from .protocol import DATA_ROOT, DEFAULT, method_contract, validate_sut_freeze
from .scenarios import DOMAINS, sample
from .storage import load_archive, save_archive


def require_learning_freeze():
    protocol = json.loads((CHECKPOINT_ROOT / "training_protocol.json").read_text(encoding="utf-8"))
    for sut in SUTS:
        require_qualified(sut)
        if sut.family == "learning":
            report = json.loads((CHECKPOINT_ROOT / sut.training_run / "training.json").read_text(encoding="utf-8"))
            if (report["configuration"] != protocol["configuration"]
                    or report["source_contract"] != protocol["source_contract"]
                    or report["actual_timesteps"] < protocol["configuration"]["total_timesteps"]):
                raise ValueError("All five learning runs must follow the pre-target training protocol")
    return protocol


def collect_history():
    require_learning_freeze()
    for role in ("history", "development"):
        for kind in DOMAINS:
            scenarios = sample(kind, DEFAULT.candidate_count, role)
            for sut in SUTS:
                if sut.role != role:
                    continue
                bank = collect_bank(sut.sut_id, scenarios,
                                    DATA_ROOT / "physical" / role / kind / sut.sut_id,
                                    environment_seed=DEFAULT.environment_seeds[0])
                executions = [bank(s) for s in scenarios]
                provenance = {"qualified": True, "bank": str(bank.root.relative_to(DATA_ROOT)),
                              "contract": bank.manifest["contract"],
                              "environment_seed": DEFAULT.environment_seeds[0],
                              "physical_calls": bank.manifest["attempted_calls"]}
                archive = ResponseArchive(sut.sut_id, sut.family, sut.lineage, role, scenarios,
                                           np.asarray([e.feedback.collision for e in executions]),
                                           np.array([[e.feedback.safety.min_clearance_m,
                                                      e.feedback.safety.min_ttc_s] for e in executions]),
                                           provenance=provenance)
                destination = DATA_ROOT / role / kind / f"{sut.sut_id}.npz"
                if destination.exists():
                    existing = load_archive(destination)
                    if (existing.provenance != provenance
                            or not np.array_equal(existing.collision, archive.collision)
                            or existing.scenarios != scenarios):
                        raise ValueError("Existing archive differs from its real execution bank")
                else:
                    save_archive(destination, archive, provenance=provenance)


def collect_targets():
    require_learning_freeze()
    protocol = json.loads((DATA_ROOT / "frozen_protocol.json").read_text(encoding="utf-8"))
    validate_sut_freeze(protocol)
    if protocol["source_contract"] != method_contract():
        raise ValueError("Method or evaluation interface changed after protocol freezing")
    if protocol["protocol"] != json.loads(json.dumps(DEFAULT.__dict__)):
        raise ValueError("Runtime settings differ from the frozen final protocol")
    for kind in DOMAINS:
        for condition in DEFAULT.conditions:
            scenarios = sample(kind, DEFAULT.candidate_count, condition)
            for environment_seed in DEFAULT.environment_seeds:
                for sut in SUTS:
                    if sut.role != "target" or kind not in sut.supported_scenarios:
                        continue
                    collect_bank(sut.sut_id, scenarios,
                                 DATA_ROOT / "targets" / sut.sut_id / condition / kind / f"env_{environment_seed}",
                                 environment_seed=environment_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("history", "targets"))
    stage = parser.parse_args().stage
    if stage == "history":
        collect_history()
    else:
        collect_targets()


if __name__ == "__main__":
    main()
