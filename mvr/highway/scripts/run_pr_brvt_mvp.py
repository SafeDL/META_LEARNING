"""Run PR-BRVT on the corrected E6 bank without new simulation or training."""

from __future__ import annotations

from pathlib import Path

from mvr.highway.config import RegressionExperimentConfig
from mvr.highway.data.response_bank import ResponseBank
from mvr.highway.experiments.run_loso_regression_mining import (
    plot_curves,
    run_loso_regression_mining,
    summarize_gate,
    write_rows,
    write_summary,
)

DEFAULT_BANK = Path(
    "results/diva_highway/cutin_mvp_e6_action_fix/response_bank_highway_e6.npz"
)
DEFAULT_OUTPUT_DIR = Path("results/diva_highway/pr_brvt_mvp")


def main() -> None:
    if not DEFAULT_BANK.is_file():
        raise FileNotFoundError(
            f"Corrected E6 response bank is required: {DEFAULT_BANK}"
        )

    rows = run_loso_regression_mining(
        ResponseBank.load(DEFAULT_BANK), RegressionExperimentConfig()
    )
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(rows, DEFAULT_OUTPUT_DIR / "loso_regression_mining.csv")
    summary = summarize_gate(rows)
    summary["response_bank"] = str(DEFAULT_BANK)
    summary["new_highway_episodes"] = 0
    summary["new_model_training"] = 0
    summary["new_response_bank"] = 0
    write_summary(summary, DEFAULT_OUTPUT_DIR / "summary.json")
    plot_curves(
        rows,
        "regression_curve",
        "Cumulative Regression Critical Score",
        DEFAULT_OUTPUT_DIR / "regression_critical_curve.png",
    )
    plot_curves(
        rows,
        "raw_curve",
        "Cumulative Raw Critical Score",
        DEFAULT_OUTPUT_DIR / "raw_critical_curve.png",
    )
    plot_curves(
        rows,
        "target_specific_curve",
        "Cumulative Target-Specific Failures",
        DEFAULT_OUTPUT_DIR / "target_specific_failure_curve.png",
    )
    print(f"Wrote PR-BRVT results to {DEFAULT_OUTPUT_DIR}")
    print(f"Gate BRVT-1: {summary['decision']} ({summary['reason']})")


if __name__ == "__main__":
    main()
