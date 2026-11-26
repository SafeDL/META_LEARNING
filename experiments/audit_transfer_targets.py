"""Audit formal-target executions only after GP settings have been sealed.

This is evaluator-side data quality, never a source of model selection or query
feedback. Missing banks stop the entire audit before any target trace is read.
"""

import hashlib
from pathlib import Path

import numpy as np

from controllers.highway.catalog import FORMAL_TARGET_SUTS

from .audit_transfer_history import background_profile
from .executions import ExecutionBank, execution_contract, require_qualified
from .history_transfer_experiment import (
    OUTPUT,
    TARGET_COUNT,
    freeze,
    registered_populations,
    scenarios_for,
)
from .record_audit import BankTruth
from .rss import measurement_contract
from .storage import read_json, write_json
from .transfer_settings import REGISTRY, load_settings


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_banks(root):
    populations = registered_populations(root)
    banks = {}
    for kind in scenarios_for(root):
        for sut in FORMAL_TARGET_SUTS:
            require_qualified(sut)
            bank = ExecutionBank(Path(root) / "physical/target" / kind / sut.sut_id)
            if (
                bank.manifest["sut_id"] != sut.sut_id
                or bank.manifest["contract"] != execution_contract(sut, kind=kind)
                or bank.manifest["environment_seed"] != 81001
                or len(bank.candidates) != TARGET_COUNT
                or bank.candidates != populations["target", kind]
            ):
                raise ValueError("Target quality audit requires the complete registered population")
            banks[kind, sut.sut_id] = bank
    return banks


def audit_bank(bank):
    truth = BankTruth(bank)
    if not np.isfinite(truth.margins).all():
        raise ValueError("Target RSS measurements contain nonfinite values")
    maximum_stored_error, stored_rss = 0.0, 0
    for index, scene in enumerate(bank.candidates):
        feedback = bank(scene).feedback
        if feedback.safety.rss_margin is not None:
            if feedback.rss_contract != measurement_contract():
                raise ValueError("Stored target RSS uses a different measurement definition")
            error = abs(float(feedback.safety.rss_margin) - float(truth.margins[index]))
            if not np.isfinite(error) or error > 1e-12:
                raise ValueError("Stored target RSS disagrees with its measured native trajectory")
            maximum_stored_error = max(maximum_stored_error, error)
            stored_rss += 1
    return {
        "profile": {**truth.profile, **background_profile(bank, truth)},
        "rss": {
            "contract": measurement_contract(),
            "measured_records": len(truth.margins),
            "stored_rss_records": stored_rss,
            "maximum_stored_rss_error": maximum_stored_error,
            "minimum": float(truth.margins.min()),
            "maximum": float(truth.margins.max()),
            "mean": float(truth.margins.mean()),
        },
    }


def protocol(root, banks):
    root = Path(root)
    return {
        "scope": "Evaluator-side quality of all formal-target native records; no method fitting",
        "settings_sha256": sha(root / REGISTRY),
        "base_protocol_sha256": sha(root / "reports/protocol.json"),
        "banks": [
            {"scenario": kind, "sut_id": sut, "snapshot": bank.snapshot_fingerprint}
            for (kind, sut), bank in banks.items()
        ],
        "source_sha256": {
            name: sha(Path(__file__).with_name(name))
            for name in (
                "audit_transfer_targets.py",
                "audit_transfer_history.py",
                "record_audit.py",
                "rss.py",
                "executions.py",
            )
        },
        "collision_check": "Saved native ego collision flags; not a separate collision engine",
        "new_physical_executions": 0,
        "new_optimizer_steps": 0,
        "formal_target_labels_read": True,
        "use": "Quality and reporting only; sealed settings remain unchanged",
    }


def audit(root=OUTPUT):
    root = Path(root)
    load_settings(root)  # Must precede all target metadata and response access.
    banks = target_banks(root)  # All banks must be ready before decoding any target trace.
    directory = root / "reports/target_quality"
    declaration = freeze(directory / "protocol.json", protocol(root, banks))
    rows = []
    for (kind, sut), bank in banks.items():
        specification = {
            "protocol": declaration,
            "scenario": kind,
            "sut_id": sut,
            "physical_bank_snapshot": bank.snapshot_fingerprint,
        }
        path = directory / kind / f"{sut}.json"
        if path.exists():
            saved = read_json(path)
            if saved["specification"] != specification:
                raise ValueError(
                    "Target quality record belongs to another settings or execution version"
                )
        else:
            saved = {"specification": specification, **audit_bank(bank)}
            write_json(path, saved)
        if (
            saved["profile"]["records"] != TARGET_COUNT
            or saved["rss"]["measured_records"] != TARGET_COUNT
        ):
            raise ValueError("Target quality audit is incomplete")
        rows.append({"scenario": kind, "sut_id": sut, **saved["profile"], "rss": saved["rss"]})
        print({"scenario": kind, "sut_id": sut, "records": saved["profile"]["records"]}, flush=True)
    expected_banks = len(scenarios_for(root)) * len(FORMAL_TARGET_SUTS)
    if (
        len(rows) != expected_banks
        or sum(row["records"] for row in rows) != expected_banks * TARGET_COUNT
    ):
        raise ValueError("Target quality report omits registered tasks or responses")
    summary = {
        "protocol": declaration,
        "status": "complete",
        "banks": expected_banks,
        "records": sum(row["records"] for row in rows),
        "rows": rows,
        "new_physical_executions": 0,
        "new_optimizer_steps": 0,
        "formal_target_labels_read": True,
        "interpretation": "Audit completion; retain zero-collision, departure and background diagnostics without selecting ADSs or settings",
    }
    write_json(directory / "summary.json", summary)
    return summary


def verify(root=OUTPUT):
    """Check saved complete quality evidence against current banks, without refitting."""
    root = Path(root)
    load_settings(root)
    banks = target_banks(root)
    declaration = protocol(root, banks)
    directory = root / "reports/target_quality"
    summary = read_json(directory / "summary.json")
    if (
        summary["status"] != "complete"
        or summary["protocol"] != declaration
        or read_json(directory / "protocol.json") != declaration
    ):
        raise ValueError("Target quality proof belongs to another input or settings version")
    rows, files = [], [directory / "protocol.json", directory / "summary.json"]
    for (kind, sut), bank in banks.items():
        path = directory / kind / f"{sut}.json"
        saved = read_json(path)
        expected = {
            "protocol": declaration,
            "scenario": kind,
            "sut_id": sut,
            "physical_bank_snapshot": bank.snapshot_fingerprint,
        }
        profile, rss = saved["profile"], saved["rss"]
        if (
            saved["specification"] != expected
            or profile["records"] != TARGET_COUNT
            or type(profile["collisions"]) is not int
            or not 0 <= profile["collisions"] <= TARGET_COUNT
            or rss["contract"] != measurement_contract()
            or rss["measured_records"] != TARGET_COUNT
            or not np.isfinite(rss["maximum_stored_rss_error"])
            or not 0 <= rss["maximum_stored_rss_error"] <= 1e-12
            or not all(np.isfinite(rss[name]) for name in ("minimum", "maximum", "mean"))
            or not rss["minimum"] <= rss["mean"] <= rss["maximum"]
        ):
            raise ValueError("Target quality bank is incomplete or inconsistent")
        rows.append({"scenario": kind, "sut_id": sut, **profile, "rss": rss})
        files.extend(
            [
                path,
                bank.root / "manifest.json",
                *[bank.execution_path(i) for i in range(len(bank.candidates))],
            ]
        )
    if (
        summary["rows"] != rows
        or summary["banks"] != len(banks)
        or summary["records"] != len(banks) * TARGET_COUNT
        or summary["new_physical_executions"] != 0
        or summary["new_optimizer_steps"] != 0
    ):
        raise ValueError("Target quality summary omits registered tasks or responses")
    return {"summary": summary, "files": [str(path) for path in files]}


if __name__ == "__main__":
    audit()
