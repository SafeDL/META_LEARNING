"""Build reviewed ScenarioFuzz graph models from the two allowed historical tasks."""

from dataclasses import asdict

from experiments.construction import load_archives
from experiments.discovery_config import read_json, write_json
from experiments.protocol import ROOT
from testing_methods.common.response import archive_fingerprints

from .model import DEFAULT, load, train

OUTPUT = ROOT / "artifacts/runs/domain_comparison/methods/scenario_fuzz/stage_a"
MODELS = ROOT / "artifacts/runs/domain_comparison/models/scenario_fuzz"


def main():
    audit = read_json(
        ROOT / "artifacts/runs/domain_comparison/reports/scenario_fuzz_author_audit.json"
    )
    if audit["status"] != "passed":
        raise ValueError(
            "Review the pinned author architecture before starting historical graph training"
        )
    reports = []
    for kind in ("cut_in", "cut_out"):
        history, _ = load_archives(kind)
        path = MODELS / kind / "risk.pt"
        if path.exists():
            _, checkpoint = load(path)
            if checkpoint["config"] != asdict(DEFAULT) or checkpoint[
                "history_fingerprints"
            ] != archive_fingerprints(history):
                raise ValueError(
                    "Saved graph model belongs to a different allowed training construction"
                )
            report = {k: v for k, v in checkpoint.items() if k != "state_dict"}
        else:
            report = train(history, path)
        reports.append({"scenario": kind, "path": str(path.relative_to(ROOT)), **report})
        write_json(
            OUTPUT / "training.json",
            {
                "status": "complete" if len(reports) == 2 else "running",
                "tasks": reports,
                "new_physical_calls": 0,
            },
        )
    return reports


if __name__ == "__main__":
    main()
