"""Write an auditable Formal Stage1 acceptance decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from ..training.checkpoint import HierarchicalCheckpoint
from ..validation.stage1_acceptance import audit_stage1


def _command_passes(command: list[str]) -> bool:
    return subprocess.run(command, check=False).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--skip-engineering", action="store_true")
    args = parser.parse_args()
    run = args.run
    coverage = json.loads((run / "coverage.json").read_text(encoding="utf-8"))
    validation = json.loads((run / "validation.json").read_text(encoding="utf-8"))
    checkpoint = HierarchicalCheckpoint.load(run / "inner_pretrain.pt")
    pytest_passed = not args.skip_engineering and _command_passes(
        [sys.executable, "-m", "pytest", "mvr/tests", "-q"]
    )
    compileall_passed = not args.skip_engineering and _command_passes(
        [sys.executable, "-m", "compileall", "-q", "mvr", "archives/pearl_learning", "archives/sac_scenario_mining"]
    )
    result = audit_stage1(
        coverage,
        validation,
        checkpoint.state,
        pytest_passed=pytest_passed,
        compileall_passed=compileall_passed,
    )
    (run / "acceptance.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
