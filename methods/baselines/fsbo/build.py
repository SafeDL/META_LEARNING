"""New-archive construction for complete-history and strict held-family FSBO."""

from benchmark.construction import load_construction
from benchmark.protocol import ROOT
from benchmark.scenarios import DOMAINS

from .method import train


MODEL_ROOT = ROOT / "results/baselines/fsbo/collision/models"


def main():
    for kind in DOMAINS:
        for family in (None, "rule", "optimization", "search", "learning"):
            history, development = load_construction(kind, family)
            train(history, development, MODEL_ROOT / (family or "all") / f"{kind}.pt",
                  held_family=family)


if __name__ == "__main__":
    main()
