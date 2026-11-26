"""Hash the pre-existing source and result artifacts without loading their outcomes."""

import argparse
import hashlib

from experiments.protocol import ROOT
from experiments.storage import read_json

RECEIPT = ROOT / "results/lifelong_transfer/provenance/sealed_inputs.json"
NEW_SOURCES = {
    "controllers/highway/continuous_sac.py",
    "controllers/highway/interval_robust_planner.py",
    "experiments/accumulation/lifelong.py",
    "experiments/lifelong/__init__.py",
    "experiments/lifelong/__main__.py",
    "experiments/lifelong/report.py",
    "experiments/lifelong_seals.py",
    "experiments/lifelong_transfer.py",
    "experiments/plotting/lifelong.py",
    "experiments/public_ads/collect_continuous_sac.py",
    "experiments/public_ads/continuous_sac.py",
    "experiments/public_ads/evaluate_continuous_sac.py",
    "experiments/regression/lifelong_replay.py",
    "simulator/highway/continuous_env.py",
    "testing_methods/history_transfer/lifelong_archive.py",
    "testing_methods/history_transfer/lifelong_reliability.py",
    "testing_methods/history_transfer/lifelong_selection.py",
    "testing_methods/history_transfer/lifelong_training.py",
}
NEW_TARGETS = {"continuous_sac", "interval_robust_planner"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree(base):
    digest = hashlib.sha256()
    count = 0
    paths = sorted(
        path for path in base.rglob("*")
        if path.is_file() and not ({"lifelong_transfer", *NEW_TARGETS} & set(path.relative_to(base).parts))
    )
    for path in paths:
        relative = path.relative_to(base).as_posix()
        digest.update(f"{relative}\0{sha(path)}\n".encode())
        count += 1
    return {"files": count, "sha256": digest.hexdigest()}


def snapshot():
    sources = [
        path for name in ("controllers", "experiments", "simulator", "testing_methods")
        for path in (ROOT / name).rglob("*.py")
        if path.relative_to(ROOT).as_posix() not in NEW_SOURCES
    ]
    results = {
        path.relative_to(ROOT).as_posix(): _tree(path)
        for path in sorted((ROOT / "results").iterdir())
        if path.is_dir() and path.name != "lifelong_transfer"
    }
    for name in ("figures", "controllers/highway/checkpoints"):
        results[name] = _tree(ROOT / name)
    return {
        "source_files": {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(sources)},
        "artifact_trees": results,
    }


def verify():
    previous = read_json(RECEIPT)
    current = snapshot()
    if previous != current:
        changed = [
            key for category in current
            for key in sorted(set(previous[category]) | set(current[category]))
            if previous[category].get(key) != current[category].get(key)
        ]
        raise ValueError(f"Frozen sources or artifact trees changed: {', '.join(changed)}")
    return previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "verify"))
    args = parser.parse_args()
    if args.action == "record":
        from experiments.history_transfer_experiment import freeze

        result = freeze(RECEIPT, snapshot())
    else:
        result = verify()
    print(len(result["source_files"]), "source files;", sum(
        tree["files"] for tree in result["artifact_trees"].values()
    ), "sealed artifacts")


if __name__ == "__main__":
    main()
