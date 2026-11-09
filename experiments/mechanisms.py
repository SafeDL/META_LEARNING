"""Mechanism controls on the same eight-ADS fixed-history tasks."""

from dataclasses import replace

from controllers.highway.catalog import CURRENT_INDEPENDENT_SUTS
from testing_methods.history_transfer.training import DEFAULT as TRAINING
from testing_methods.history_transfer.training import build_scope, load_model

from .joint_validation import OUTPUT
from .pilot import MECHANISMS, fixed_comparison


def historical_collision_models(kind):
    root = OUTPUT / "models/historical_collision"
    build_scope(kind, config=replace(TRAINING, historical_rss=False), model_root=root)
    return tuple(
        load_model(root / kind / "all_sources" / name)[0]
        for name in ("target_kernel.pt", "historical_encoder.pt")
    )


def main():
    return fixed_comparison(
        OUTPUT / "mechanisms",
        systems=CURRENT_INDEPENDENT_SUTS,
        suite="mechanisms",
        methods=("joint", *MECHANISMS),
        reuse_root=OUTPUT / "fixed",
        historical_collision_loader=historical_collision_models,
    )


if __name__ == "__main__":
    main()
