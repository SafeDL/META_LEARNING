"""Run PR-DVM on the corrected E6 response bank without new simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from mvr.highway.config import ExperimentConfig
from mvr.highway.data.response_bank import ResponseBank
from mvr.highway.experiments.run_loso_differential_mining import (
    plot_curves,
    run_loso_differential_mining,
    summarize_gate,
    write_rows,
    write_summary,
)

DEFAULT_BANK = Path(
    "results/diva_highway/cutin_mvp_e6_action_fix/response_bank_highway_e6.npz"
)
DEFAULT_OUTPUT_DIR = Path("results/diva_highway/pr_dvm_mvp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.bank.is_file():
        raise FileNotFoundError(f"Corrected E6 response bank is required: {args.bank}")

    rows = run_loso_differential_mining(
        ResponseBank.load(args.bank), ExperimentConfig()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(rows, args.output_dir / "loso_differential_mining.csv")
    summary = summarize_gate(rows)
    summary["response_bank"] = str(args.bank)
    summary["new_highway_episodes"] = 0
    summary["new_model_training"] = 0
    write_summary(summary, args.output_dir / "summary.json")
    plot_curves(
        rows,
        "differential_curve",
        "Cumulative Differential Critical Score",
        args.output_dir / "differential_curve.png",
    )
    plot_curves(
        rows,
        "raw_curve",
        "Cumulative Raw Critical Score",
        args.output_dir / "raw_critical_curve.png",
    )
    plot_curves(
        rows,
        "target_specific_curve",
        "Cumulative Target-Specific Failures",
        args.output_dir / "target_specific_failure_curve.png",
    )
    print(f"Wrote PR-DVM results to {args.output_dir}")
    print(f"Gate PR-DVM-1: {summary['decision']} ({summary['reason']})")


if __name__ == "__main__":
    main()
