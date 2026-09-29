"""Build a prospective variable-dimensional FBRT development manifest.

This module never changes or reuses a frozen research_v2 bank.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import yaml
from scipy.stats import qmc

CONFIG = Path(__file__).resolve().parent / "configs" / "scenario_parameter_space.yaml"
CATALOGUE = CONFIG.parent / "scenario_catalogue.yaml"
ROOT = Path("results/method_chains/failure_memory_regression/scenario_sampling_development")
IMPLEMENTED_FAMILIES = {"S01", "S02", "S03", "S04", "S05", "S06", "S08", "S09"}


def compile_manifest(config_path: Path = CONFIG) -> list[dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "fbrt_scenario_parameter_space_v3":
        raise ValueError("unsupported parameter-space schema")
    base = yaml.safe_load(CATALOGUE.read_text(encoding="utf-8"))
    cards = {card["id"]: card for card in base["scenarios"]}
    proposals = config["scenarios"]
    if set(cards) != {item["id"] for item in proposals} or len(proposals) != len(cards):
        raise ValueError("v3 must document each catalogue family exactly once")
    sampling = config["sampling"]
    if sampling["design"] != "scrambled_sobol" or sampling["target_outcome_filtering"]:
        raise ValueError("development sampling must be scrambled Sobol without outcome filtering")
    counts = {int(dimension): int(count) for dimension, count in
              sampling["samples_by_dimension"].items()}
    if any(count < 4 or count & (count - 1) for count in counts.values()):
        raise ValueError("Sobol sample counts must be powers of two >= 4")
    seed = int(sampling["simulator_seed"])
    rows = []
    for item in proposals:
        family = item["id"]
        axes = item["axes"]
        groups = item["groups"]
        if not axes or any(len(bounds) != 2 or bounds[0] >= bounds[1]
                           for bounds in axes.values()):
            raise ValueError(f"invalid bounds for {family}")
        if set(groups) != set(axes) or any(group not in {
                "geometry", "speed", "timing", "braking", "acceleration"}
                for group in groups.values()):
            raise ValueError(f"invalid semantic groups for {family}")
        if item["status"] != "executable":
            continue
        if family not in IMPLEMENTED_FAMILIES:
            raise ValueError(f"unsupported executable family: {family}")
        names = tuple(axes)
        count = counts[len(names)]
        unit = qmc.Sobol(d=len(names), scramble=True, seed=seed + int(family[1:])).random_base2(
            m=count.bit_length() - 1)
        for index, sample in enumerate(unit):
            values = {name: float(axes[name][0] + sample[k] *
                                  (axes[name][1] - axes[name][0]))
                      for k, name in enumerate(names)}
            fixed = dict(cards[family]["fixed_context"])
            # An active coordinate has one authoritative value in the concrete case.
            for name in names:
                fixed.pop(name, None)
            rows.append({
                "scenario_id": f"sobol:{family}:{index:04d}",
                "catalogue_id": family,
                "template_id": cards[family]["template_id"],
                "parameterization_version": "research_v3_mechanism",
                "context_id": f"{family}:research_v3:base",
                "sampling_kind": "scrambled_sobol",
                "sample_index": index,
                "active_parameters": values,
                "parameter_groups": groups,
                "research_bounds": axes,
                "fixed_context": fixed,
                "required_capability": cards[family].get("required_capability", []),
                "simulator_seed": seed,
            })
    return rows


def write_manifest(root: Path = ROOT, config_path: Path = CONFIG) -> dict:
    rows = compile_manifest(config_path)
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                 allow_nan=False) + "\n" for row in rows)
    candidate_counts = dict(sorted(Counter(row["catalogue_id"] for row in rows).items()))
    protocol = {
        "schema": "fbrt_scenario_sampling_development",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "catalogue_sha256": hashlib.sha256(CATALOGUE.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "families": list(candidate_counts),
        "candidate_count": len(rows),
        "candidate_counts": candidate_counts,
        "sampling": "scrambled_sobol_per_family",
        "simulator_seed": rows[0]["simulator_seed"],
        "status": "development_candidates_no_physical_results",
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "scenario_manifest.jsonl"
    protocol_path = root / "protocol.json"
    manifest.write_bytes(payload.encode("utf-8"))
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    return protocol


def main() -> None:
    print(json.dumps(write_manifest(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
