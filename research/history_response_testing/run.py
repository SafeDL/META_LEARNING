"""Run the retained experiment, diagnostics and report without CLI switches."""
import torch

from . import evaluate, experiment, group_validation, history_model
from . import original_pool, prediction_diagnostics, report
from .config import OUTPUT, SEEDS


def main():
    torch.set_num_threads(1)
    for seed in SEEDS:
        history_model.train(seed)
    experiment.main()
    summary = evaluate.main()
    if not summary["success"]:
        raise RuntimeError("The locked confirmation criteria were not met")
    original_pool.main()
    prediction_diagnostics.main()
    if not (OUTPUT / "group_validation.json").exists():
        group_validation.main()
    report.main()
    print("History response testing complete", flush=True)


if __name__ == "__main__":
    main()
