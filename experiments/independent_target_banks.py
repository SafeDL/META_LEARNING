"""Construct independent raw pools without reading their discovery statistics."""

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict

from controllers.highway.catalog import INDEPENDENT_SUTS

from .discovery_config import read_json, write_json
from .executions import ExecutionBank, collect_bank, execution_contract
from .independent_preparation import OUTPUT, validate_preparation
from .independent_protocol import (
    construction_recipe,
    require_independent_qualification,
    target_bank_path,
)
from .protocol import ROOT
from .scenarios import DOMAINS, sample


def raw_bank_sha256(root):
    digest = hashlib.sha256()
    for path in sorted((root / "executions").glob("*.json.gz")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def design():
    recipe = construction_recipe()
    hashes = {
        p: h
        for p, h in recipe.pop("source_files").items()
        if p.startswith("testing_methods/")
        or p
        in (
            "experiments/independent_validation.py",
            "experiments/scenarios.py",
            "experiments/initialization.py",
            "experiments/collision_metrics.py",
            "experiments/query.py",
        )
    }
    return {
        "purpose": "Stage D independent pool construction; selector and model training recipe fixed before raw outcomes",
        "objects": [asdict(s) for s in INDEPENDENT_SUTS],
        "recipe": recipe,
        "model_and_selection_source_files": hashes,
        "contracts": {
            s.sut_id: {kind: execution_contract(s, kind=kind) for kind in DOMAINS}
            for s in INDEPENDENT_SUTS
        },
        "expected_physical_calls": 20480,
        "discovery_statistics_read": False,
        "model_checkpoint_selection": "allowed development data only while raw independent pools are constructed",
        "collector_sha256": hashlib.sha256(
            (ROOT / "experiments/independent_target_banks.py").read_bytes()
        ).hexdigest(),
    }


def freeze():
    validate_preparation()
    require_independent_qualification()
    if read_json(OUTPUT / "construction_data_audit.json")["status"] != "passed":
        raise ValueError(
            "Audit the new interaction and historical data before acquiring independent pools"
        )
    suites = list(ET.parse(OUTPUT / "regression.xml").getroot().iter("testsuite"))
    if not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ("failures", "errors")):
        raise ValueError(
            "Pass the final engineering checks before independent raw pool construction"
        )
    spec = json.loads(json.dumps(design()))
    path = OUTPUT / "target_bank_protocol.json"
    if path.exists() and read_json(path) != spec:
        raise ValueError("The independent raw-pool construction was already frozen differently")
    write_json(path, spec)
    return spec


def main():
    spec = freeze()
    rows, started = [], time.perf_counter()
    for sut in INDEPENDENT_SUTS:
        for kind in DOMAINS:
            task = {"target_id": sut.sut_id, "scenario": kind}
            root = target_bank_path(task)
            if (root / "manifest.json").exists() and read_json(root / "manifest.json")[
                "status"
            ] == "complete":
                bank = ExecutionBank(root)
            else:
                bank = collect_bank(sut.sut_id, sample(kind, 512, "new"), root, workers=16)
            if bank.manifest["contract"] != spec["contracts"][sut.sut_id][kind]:
                raise ValueError("A new target bank differs from its fixed qualified object")
            rows.append(
                {
                    "target_id": sut.sut_id,
                    "scenario": kind,
                    "path": str(root.relative_to(ROOT)).replace("\\", "/"),
                    "manifest_sha256": hashlib.sha256(
                        (root / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "raw_execution_files_sha256": raw_bank_sha256(root),
                    "attempted_calls": bank.manifest["attempted_calls"],
                    "physical_execution_s": bank.manifest["physical_execution_s"],
                    "failed_or_interrupted_calls": bank.manifest["failed_or_interrupted_calls"],
                }
            )
            write_json(
                OUTPUT / "target_bank_index.json",
                {
                    "status": "complete" if len(rows) == 40 else "running",
                    "banks": rows,
                    "attempted_calls": sum(r["attempted_calls"] for r in rows),
                    "discovery_statistics_read": False,
                    "last_collection_wall_s": time.perf_counter() - started,
                },
            )
    return rows


if __name__ == "__main__":
    main()
