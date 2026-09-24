"""Command line interface for Highway-env SUT qualification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import assets
from .runner import (
    ASSETS,
    DEFAULT_SUTS,
    ROOT,
    audit,
    audit_by_function,
    common_scenarios,
    evaluate,
    final_manifest,
    native_checkpoint_validation,
    qualification_scenarios,
    report,
    save_response_bank,
    snapshot,
)


def _selected_suts(value: list[str] | None) -> tuple[str, ...]:
    return tuple(value or DEFAULT_SUTS)


def _add_evaluation_commands(subparsers) -> None:
    native = subparsers.add_parser("native")
    native.add_argument("--episodes", type=int, default=20)

    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--suts", nargs="+")
    qualify.add_argument("--episodes", type=int, default=20)

    common = subparsers.add_parser("common")
    common.add_argument("--suts", nargs="+")
    common.add_argument("--scenarios", type=int, default=96)


def _read_episode_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    boolean_fields = (
        "ego_collision",
        "background_collision",
        "near_miss",
        "completed",
    )
    for record in records:
        for field in boolean_fields:
            record[field] = record[field].lower() == "true"
    return records


def _run_audit() -> None:
    common_path = ROOT / "common_validation/episodes.csv"
    if not common_path.exists():
        raise FileNotFoundError("Run the common evaluation before audit")

    records = _read_episode_records(common_path)
    retained = final_manifest(ROOT / "sut_manifest.json")
    retained_records = [record for record in records if record["sut"] in retained]
    audit(retained_records, ROOT / "risk_structure/pairwise.csv")
    audit_by_function(retained_records, ROOT / "risk_structure/by_function.csv")
    save_response_bank(
        retained_records,
        ROOT / "response_banks/retained_response_bank.npz",
    )

    qualification_path = ROOT / "common_validation/qualification.csv"
    qualifications = (
        _read_episode_records(qualification_path)
        if qualification_path.exists()
        else []
    )
    report(qualifications, retained_records, ROOT / "report.md")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("snapshot")
    subparsers.add_parser("fetch")
    subparsers.add_parser("inspect")
    subparsers.add_parser("audit")
    _add_evaluation_commands(subparsers)
    args = parser.parse_args()

    if args.command == "snapshot":
        snapshot()
    elif args.command == "fetch":
        records = assets.fetch(ASSETS)
        output = ROOT / "provenance/assets.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    elif args.command == "inspect":
        print(json.dumps(assets.inspect(ASSETS), indent=2))
    elif args.command == "native":
        records = native_checkpoint_validation(
            args.episodes,
            ROOT / "native_validation/episodes.csv",
            3100,
        )
        print(f"Wrote {len(records)} native validation records")
    elif args.command == "qualify":
        selected = _selected_suts(args.suts)
        records = evaluate(
            selected,
            qualification_scenarios(args.episodes, 4100),
            ROOT / "common_validation/qualification.csv",
            4100,
        )
        print(f"Wrote {len(records)} qualification records")
    elif args.command == "common":
        selected = _selected_suts(args.suts)
        records = evaluate(
            selected,
            common_scenarios(args.scenarios, 5100),
            ROOT / "common_validation/episodes.csv",
            5100,
        )
        save_response_bank(
            records,
            ROOT / "response_banks/retained_response_bank.npz",
        )
        print(f"Wrote {len(records)} common evaluation records")
    else:
        _run_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
