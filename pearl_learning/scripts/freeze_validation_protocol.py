"""Freeze validated PEARL choices before holdout evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pearl_learning.src.io import write_json
from pearl_learning.src.validation_freeze import freeze_validation_protocol


def _keyed_paths(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("artifacts must use policy=path")
        policy, path = value.split("=", 1)
        if not policy or not path or policy in result:
            raise ValueError("artifact policies must be unique non-empty policy=path pairs")
        result[policy] = path
    return result


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskbook-hash", required=True)
    parser.add_argument("--evaluation", nargs="+", required=True, help="policy=validation metrics JSON")
    parser.add_argument("--required-policies", nargs="+", default=["fixed", "random", "initial_condition_diversity", "posterior_action_disagreement"])
    parser.add_argument("--equal-budget", nargs="*", help="optional policy=validation equal-budget JSON")
    parser.add_argument("--representation-audit", nargs="*", default=[])
    parser.add_argument("--calibration")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluations = {policy: _read(path) for policy, path in _keyed_paths(args.evaluation).items()}
    budgets = None if args.equal_budget is None else {policy: _read(path) for policy, path in _keyed_paths(args.equal_budget).items()}
    result = freeze_validation_protocol(
        taskbook_hash=args.taskbook_hash, evaluations=evaluations, required_policies=list(args.required_policies),
        equal_budget=budgets, representation_audits=[_read(path) for path in args.representation_audit],
        calibration=None if args.calibration is None else _read(args.calibration),
    )
    write_json(args.output, result)
    print(f"validation protocol frozen: {args.output}")


if __name__ == "__main__":
    main()
