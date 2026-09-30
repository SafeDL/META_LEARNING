"""Directional and total-budget discovery prefixes for frozen replay banks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DIRECTION_PREFIXES = (1, 5, 10, 20)
TOTAL_PREFIXES = (10, 20, 40)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summarize(root: Path, query_dir: Path | None = None) -> list[dict]:
    """Report D@k separately by direction and by chronological total budget."""
    query_dir = query_dir or root
    queries = _jsonl(query_dir / "queries.jsonl")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in queries:
        groups[row["parent"], row["target"], row["method"], row["repeat"]].append(row)
    output = []
    for (parent, target, method, repeat), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["rank"])
        for direction in ("R", "I"):
            local = [row for row in rows if row["direction"] == direction]
            for prefix in DIRECTION_PREFIXES:
                used = min(prefix, len(local))
                output.append({"parent": parent, "target": target, "method": method,
                               "repeat": repeat, "scope": direction,
                               "prefix_budget": prefix, "actual_queries": used,
                               "short_pool": used < prefix,
                               "discoveries": sum(int(row["discovery"])
                                                  for row in local[:used])})
        for prefix in TOTAL_PREFIXES:
            used = min(prefix, len(rows))
            output.append({"parent": parent, "target": target, "method": method,
                           "repeat": repeat, "scope": "TOTAL",
                           "prefix_budget": prefix, "actual_queries": used,
                           "short_pool": used < prefix,
                           "discoveries": sum(int(row["discovery"])
                                              for row in rows[:used])})
    path = query_dir / "prefix_discoveries.csv"
    if output:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path)
    args = parser.parse_args()
    rows = summarize(args.root, args.query_dir)
    print(json.dumps({"prefix_rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
