"""Conservative family-level paired summaries for frozen NL selector replays."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _area_by_family(queries: list[dict], family: str) -> float:
    local = sorted(queries, key=lambda row: row["rank"])
    k = len(local)
    if k == 0:
        return 0.0
    cumulative = 0
    total = 0
    for row in local:
        if row["context_id"].split(":", 1)[0] == family:
            cumulative += int(row["discovery"])
        total += cumulative
    return 2 * total / (k * (k + 1))


def _signflip_p(differences: list[float]) -> float:
    """Exact two-sided sign-flip resolution; not an independence guarantee."""
    observed = abs(sum(differences))
    extreme = sum(abs(sum(sign * value for sign, value in zip(signs, differences)))
                  >= observed - 1e-12 for signs in itertools.product((-1, 1),
                                                                       repeat=len(differences)))
    return extreme / (2 ** len(differences))


def summarize(root: Path, query_dir: Path | None = None,
              primary: str | None = None,
              baselines: tuple[str, ...] | None = None) -> dict:
    query_dir = query_dir or root
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    primary = primary or protocol.get("primary_method", "directed_residual")
    baselines = baselines or tuple(protocol.get(
        "primary_baselines", ("static_risk", "coordinate_residual")))
    queries = _read_jsonl(query_dir / "queries.jsonl")
    truth = _read_csv(query_dir / "transition_truth.csv")
    cases = _read_jsonl(root / "scenario_manifest.jsonl")
    families = sorted({case["catalogue_id"] for case in cases})
    pairs = sorted({(row["parent"], row["target"]) for row in queries})
    groups = defaultdict(list)
    for row in queries:
        if row["repeat"] == 0:
            groups[row["parent"], row["target"], row["direction"], row["method"]].append(row)
    results = []
    for parent, target in pairs:
        for direction, transition in (("R", "REGRESSION"), ("I", "IMPROVEMENT")):
            available = sum(row["parent_build_id"] == parent and
                            row["target_build_id"] == target and
                            row["transition"] == transition for row in truth)
            if not available:
                for baseline in baselines:
                    results.append({"parent": parent, "target": target,
                                    "direction": direction, "baseline": baseline,
                                    "available_changes": 0, "status": "NA_NO_TRUE_CHANGE"})
                continue
            for baseline in baselines:
                key = (parent, target, direction)
                left = groups[key + (primary,)]
                right = groups[key + (baseline,)]
                if not left or not right:
                    raise ValueError("frozen comparison method is missing")
                deltas = {family: _area_by_family(left, family) -
                          _area_by_family(right, family) for family in families}
                results.append({"parent": parent, "target": target,
                                "direction": direction, "baseline": baseline,
                                "available_changes": available,
                                "status": "DESCRIPTIVE_ONLY_SMALL_CLUSTER_COUNT",
                                "primary_area": sum(_area_by_family(left, family)
                                                    for family in families),
                                "baseline_area": sum(_area_by_family(right, family)
                                                     for family in families),
                                "delta_area": sum(deltas.values()),
                                "family_deltas": deltas,
                                "family_count": len(families),
                                "exact_signflip_p_unadjusted": _signflip_p(list(deltas.values()))})
    # Holm adjustment within each release pair, across every non-NA direction
    # and both predeclared baselines. Repeats are not independent replicates.
    for parent, target in pairs:
        local = sorted((row for row in results if row["parent"] == parent and
                        row["target"] == target and "exact_signflip_p_unadjusted" in row),
                       key=lambda row: row["exact_signflip_p_unadjusted"])
        running = 0.0
        for index, row in enumerate(local):
            running = max(running, min(1.0, (len(local) - index) *
                                       row["exact_signflip_p_unadjusted"]))
            row["holm_p_within_pair"] = running
    payload = {"primary": primary, "baselines": baselines,
               "unit": "scenario family within one related release chain",
               "family_count": len(families), "comparisons": results,
               "interpretation_limit": "Three families share one SUT chain; sign-flip p is coarse and does not establish population-level superiority."}
    path = query_dir / "paired_family_statistics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path)
    parser.add_argument("--primary")
    args = parser.parse_args()
    result = summarize(args.root, args.query_dir, args.primary)
    print(json.dumps({"comparisons": len(result["comparisons"]),
                      "family_count": result["family_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
