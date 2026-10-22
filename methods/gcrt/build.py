"""Build only from new collision archives, including strict family holdouts."""

from benchmark.construction import load_construction
from benchmark.scenarios import DOMAINS

from .config import MODEL_ROOT
from .train import train


FAMILIES = ("rule", "optimization", "search", "learning")


def build(kind, held_family=None):
    history, development = load_construction(kind, held_family)
    destination = MODEL_ROOT / (held_family or "all") / f"{kind}.pt"
    return train(history, development, destination, held_family=held_family)


def main():
    for kind in DOMAINS:
        for held_family in (None, *FAMILIES):
            build(kind, held_family)


if __name__ == "__main__":
    main()
