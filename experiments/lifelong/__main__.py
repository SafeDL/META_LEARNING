"""Execute the authorized lifelong-transfer stages in dependency order."""

import argparse
import os
import subprocess
import sys

from experiments.lifelong_seals import RECEIPT, sha
from experiments.protocol import ROOT
from experiments.storage import read_json, write_json

STAGES = (
    ("release", ("experiments.public_ads.continuous_sac", "release", "--device", "cuda")),
    ("physical_bank", ("experiments.public_ads.collect_continuous_sac",)),
    ("baseline_matrix", ("experiments.public_ads.evaluate_continuous_sac",)),
    ("benchmark", ("experiments.lifelong_transfer",)),
    ("cold_start", ("experiments.accumulation.lifelong", "cold_start")),
    ("cross_architecture", ("experiments.accumulation.lifelong", "cross_architecture")),
    ("in_family", ("experiments.accumulation.lifelong", "in_family")),
    ("fixed_suite", ("experiments.regression.lifelong_replay",)),
    ("statistics", ("experiments.lifelong.report",)),
    ("figures", ("experiments.plotting.lifelong",)),
)
OUTPUT = ROOT / "results/lifelong_transfer"


def run(start_at):
    if not RECEIPT.exists():
        subprocess.run(
            [sys.executable, "-B", "-u", "-m", "experiments.lifelong_seals", "record"],
            cwd=ROOT, check=True,
        )
    first = next(index for index, (name, _) in enumerate(STAGES) if name == start_at)
    cache = ROOT / "cache/highway_agents"
    public_paths = [cache / "dependencies", *sorted((cache / "upstream").glob("*/")), cache / "highway_dt"]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([
        *(str(path) for path in public_paths), os.environ.get("PYTHONPATH", ""),
    ])}
    for name, module in STAGES[first:]:
        print(f"Starting {name}", flush=True)
        subprocess.run([sys.executable, "-B", "-u", "-m", *module], cwd=ROOT, check=True, env=environment)
        print(f"Completed {name}", flush=True)
    record_completion()


def record_completion():
    """Bind the summary, figure audits and published images to the completed stages."""
    report = OUTPUT / "reports/summary.json"
    audits = OUTPUT / "reports/figures"
    names = (
        "discovery_by_ads", "discovery_region_composition", "cross_architecture_regions",
        "prediction_gain_heatmaps", "calibration_and_weights", "cold_start_growth", "local_reliability_surface",
    )
    outputs = [report]
    for name in names:
        audit = audits / f"{name}.json"
        read_json(audit)
        outputs.extend((audit, ROOT / "figures/lifelong_transfer" / f"{name}.png"))
    write_json(OUTPUT / "completion.json", {
        "stages": [name for name, _ in STAGES],
        "artifacts": {
            path.relative_to(ROOT).as_posix(): sha(path)
            for path in outputs
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", choices=[name for name, _ in STAGES], default=STAGES[0][0])
    args = parser.parse_args()
    run(args.start_at)


if __name__ == "__main__":
    main()
