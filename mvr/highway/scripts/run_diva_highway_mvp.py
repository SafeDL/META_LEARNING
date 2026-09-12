"""Generate the response bank, run both LOSO evaluations, and create all MVP plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ExperimentConfig
from ..data.generate_anchor_bank import generate_anchor_bank
from ..data.response_bank import ResponseBank, build_response_bank
from ..experiments.run_loso_mining import run_loso_mining, write_rows as write_mining
from ..experiments.run_loso_ranking import run_loso_ranking, write_rows as write_ranking
from .render_diva_highway_mvp import render_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/diva_highway/cutin_mvp"))
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("results/diva_highway/cutin_mvp/response_bank_highway.npz"),
    )
    parser.add_argument("--rebuild-bank", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig()
    if args.rebuild_bank or not args.bank.exists():
        bank = build_response_bank(generate_anchor_bank(config.num_anchors, config.seed), config.seed)
        bank.save(args.bank)
    else:
        bank = ResponseBank.load(args.bank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.output_dir / "loso_ranking_highway.csv"
    mining_path = args.output_dir / "loso_mining_highway.csv"
    write_ranking(run_loso_ranking(bank, config), ranking_path)
    write_mining(run_loso_mining(bank, config), mining_path)
    print(f"Response bank: {args.bank}")
    print(f"Ranking results: {ranking_path}")
    print(f"Mining results: {mining_path}")
    render_results(bank, ranking_path, mining_path, args.output_dir)


if __name__ == "__main__":
    main()
