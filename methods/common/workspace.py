"""Explicit, family-specific historical training inputs and artifact locations."""

from dataclasses import dataclass
from pathlib import Path

from benchmark.initialization import initial_indices


@dataclass
class HistoricalWorkspace:
    root: Path
    family: int
    history: dict

    def __post_init__(self):
        if self.family not in (0, 1) or any(
            not (source["x"][:, 4] == self.family).all()
            for source in self.history.values()
        ):
            raise ValueError("A workspace must contain only its functional scenario")

    @property
    def archives(self):
        return ("complete", "sparse", "brake_gap" if self.family else "cutin_gap")

    @property
    def kernel_root(self):
        return self.root / "kernel"

    @property
    def encoder_root(self):
        return self.root / "encoder"

    def initialization(self, size, count, seed):
        return initial_indices(size, count, seed)
