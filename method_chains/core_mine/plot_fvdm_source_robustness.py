"""Reuse the frozen physical-replication figures for the clean-source study."""

from method_chains.core_mine import plot_heterogeneous20_replication as figures
from method_chains.core_mine.fvdm_source_robustness import ROOT, SEEDS


def main() -> None:
    figures.ROOT = ROOT
    figures.SEEDS = SEEDS
    figures.main()


if __name__ == "__main__":
    main()
