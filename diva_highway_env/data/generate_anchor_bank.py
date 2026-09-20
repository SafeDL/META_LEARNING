"""Generate the common Sobol anchor bank used by every highway-env SUT."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import qmc

PARAMETER_NAMES = ("initial_gap", "relative_speed")
LOWER_BOUNDS = np.array([5.0, -8.0])
UPPER_BOUNDS = np.array([40.0, 2.0])


def generate_anchor_bank(num_anchors: int, seed: int) -> np.ndarray:
    """Generate deterministic two-dimensional Sobol anchors in physical units."""
    if num_anchors < 1:
        raise ValueError("num_anchors must be positive")
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    unit_anchors = sampler.random(num_anchors)
    return qmc.scale(unit_anchors, LOWER_BOUNDS, UPPER_BOUNDS)


FUNCTIONAL_MODES = (
    "fast_intrusion",
    "cutin_braking",
    "lead_braking",
    "stop_and_go",
    "slow_lead_following",
)


def generate_multifunction_anchor_bank(
    num_anchors: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a balanced Cut-in and longitudinal-following scenario bank."""
    if num_anchors < len(FUNCTIONAL_MODES) or num_anchors % len(FUNCTIONAL_MODES):
        raise ValueError(
            "multifunction banks require a positive multiple of the mode count"
        )
    per_mode = num_anchors // len(FUNCTIONAL_MODES)
    anchors = np.vstack(
        [generate_anchor_bank(per_mode, seed + offset) for offset in range(len(FUNCTIONAL_MODES))]
    )
    return anchors, np.repeat(np.asarray(FUNCTIONAL_MODES, dtype="U32"), per_mode)


def save_anchor_bank(path: Path, anchors: np.ndarray) -> None:
    """Persist anchors with enough metadata to prevent coordinate ambiguity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        anchors=np.asarray(anchors, dtype=float),
        parameter_names=np.asarray(PARAMETER_NAMES),
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/diva_highway/anchor_bank.npz"),
    )
    parser.add_argument("--num-anchors", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    anchors = generate_anchor_bank(args.num_anchors, args.seed)
    save_anchor_bank(args.output, anchors)
    print(f"Wrote {len(anchors)} Sobol anchors to {args.output}")


if __name__ == "__main__":
    main()
