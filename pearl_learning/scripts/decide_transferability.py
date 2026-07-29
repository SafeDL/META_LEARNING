"""Write a query-free runtime decision report from frozen descriptors/calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pearl_learning.src.io import write_json
from pearl_learning.src.transferability_decision import (
    DEFAULT_FALLBACK,
    transferability_decision_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor-report", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fallback", default=DEFAULT_FALLBACK)
    parser.add_argument("--minimum-leave-one-out-coverage", type=float, default=0.5)
    parser.add_argument("--minimum-leave-one-out-tasks", type=int, default=2)
    parser.add_argument("--posterior-audit", help="support-only posterior audit for a calibrated uncertainty threshold")
    args = parser.parse_args()
    descriptor = json.loads(Path(args.descriptor_report).read_text(encoding="utf-8"))
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    posterior = json.loads(Path(args.posterior_audit).read_text(encoding="utf-8")) if args.posterior_audit else None
    result = transferability_decision_report(
        descriptor, calibration, fallback=args.fallback,
        minimum_leave_one_out_coverage=args.minimum_leave_one_out_coverage,
        minimum_leave_one_out_tasks=args.minimum_leave_one_out_tasks,
        posterior_audit=posterior,
    )
    write_json(args.output, result)
    print(f"transferability decision: {args.output} ({result['status']})")


if __name__ == "__main__":
    main()
