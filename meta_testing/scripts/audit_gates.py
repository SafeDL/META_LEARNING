"""CLI wrapper that writes a frozen Gate diagnostic from scalar measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=[f"G{index}" for index in range(1, 9)], required=True)
    parser.add_argument("--values-json", required=True, help="JSON mapping with the selected audit's scalar inputs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    values = json.loads(Path(args.values_json).read_text(encoding="utf-8"))
    functions = {
        "G1": audits.gate_map_representation, "G2": audits.gate_scenario_execution,
        "G3": audits.gate_heterogeneity, "G4": audits.gate_inner_controllability,
        "G5": audits.gate_identifiability, "G6": audits.gate_posterior_utility,
        "G7": audits.gate_outer_utility, "G8": audits.gate_all_in_budget,
    }
    report = functions[args.gate](**values)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
