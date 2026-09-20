from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[1]
RESULTS = REPOSITORY / "results" / "highway_replications" / "adate"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_archived_results_use_semantic_names_and_have_complete_cross_target_cases():
    version_name = re.compile(r"(?:^|[_-])v\d+(?:$|[_.-])", re.IGNORECASE)
    managed_paths = [
        *PACKAGE.joinpath("configs").rglob("*"), *RESULTS.rglob("*")
    ]
    assert not [path for path in managed_paths if version_name.search(path.name)]

    response = RESULTS / "mixture"
    cross = RESULTS / "dense"
    assert (response / "method_summary.csv").is_file()
    assert (cross / "case_strategy_summary.csv").is_file()
    assert len(list(cross.glob("seed_*/target_*"))) == 6

    required = {
        "adaptation_transitions.jsonl",
        "adaptation_propensities.jsonl",
        "mixture_coefficients_trace.csv",
        "evaluation_draws.csv",
        "importance_sampling_summary.json",
        "diagnostic_replay.gif",
        "diagnostic_replay_trace.npz",
        "manifest.json",
    }
    for case in cross.glob("seed_*/target_*"):
        assert required <= {path.name for path in case.iterdir()}


def test_archived_counts_simplex_constraints_and_provenance_are_complete():
    response = RESULTS / "mixture"
    cross = RESULTS / "dense"

    assert len(_rows(response / "method_summary.csv")) == 60
    assert len(_rows(response / "target_query_trace.csv")) == 1200
    qp_rows = _rows(response / "qp_diagnostics.csv")
    assert len(qp_rows) == 276
    assert not any(row["fallback"].lower() == "true" for row in qp_rows)
    assert max(float(row["simplex_violation"]) for row in qp_rows) < 1e-12

    assert len(_rows(cross / "case_strategy_summary.csv")) == 42
    assert len(_rows(cross / "cross_seed_summary.csv")) == 21
    assert len(_rows(cross / "final_mixture_coefficients.csv")) == 24

    manifests = [response / "manifest.json", cross / "manifest.json"]
    for case in cross.glob("seed_*/target_*"):
        alpha_rows = _rows(case / "mixture_coefficients_trace.csv")
        assert len(alpha_rows) == 240
        alpha_keys = sorted(key for key in alpha_rows[0] if key.startswith("alpha_"))
        assert all(
            abs(sum(float(row[key]) for key in alpha_keys) - 1.0) < 1e-10 for row in alpha_rows)
        assert len(_rows(case / "evaluation_draws.csv")) == 1344
        replay = json.loads((case / "diagnostic_replay_metadata.json").read_text(encoding="utf-8"))
        assert replay["frame_count"] > 0
        manifests.append(case / "manifest.json")

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config_path = REPOSITORY / Path(manifest["config"])
        assert config_path.is_file()
