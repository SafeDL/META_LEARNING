"""Fit or explicitly reject a validation-only transferability threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pearl_learning.src.io import write_json
from pearl_learning.src.transferability_calibration import calibration_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor-report", required=True)
    parser.add_argument("--taskwise-summary", required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-independent-tasks", type=int, default=8)
    parser.add_argument("--posterior-audit", help="optional support-only posterior audit for an uncertainty threshold")
    args = parser.parse_args()
    descriptor = json.loads(Path(args.descriptor_report).read_text(encoding="utf-8"))
    taskwise = json.loads(Path(args.taskwise_summary).read_text(encoding="utf-8"))
    posterior = json.loads(Path(args.posterior_audit).read_text(encoding="utf-8")) if args.posterior_audit else None
    report = calibration_report(descriptor, taskwise, shot=args.shot, minimum_independent_tasks=args.minimum_independent_tasks, posterior_audit=posterior)
    write_json(args.output, report)
    print(f"transferability calibration: {args.output} ({report['status']})")


if __name__ == "__main__":
    main()
