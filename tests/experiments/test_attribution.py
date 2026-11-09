import numpy as np
import pytest

from experiments import attribution_report
from experiments import rss_exploration_report as shared_private_report
from experiments.attribution import prediction_metrics
from experiments.protocol import ROOT
from experiments.storage import compact_result, write_json


def test_common_feedback_evaluation_excludes_observed_truth_and_handles_zero_failures():
    snapshot = {"observed_indices": [0, 2], "probabilities": [0.9, 0.8, 0.7, 0.2]}
    before = prediction_metrics(snapshot, [True, True, False, False])
    changed_observed = prediction_metrics(snapshot, [False, True, True, False])
    assert before == changed_observed
    assert before["unqueried_count"] == 2
    assert before["nll"] == pytest.approx(-np.log(0.8))
    assert before["brier"] == pytest.approx(0.04)
    assert before["top_200_failures"] == 1
    zero = prediction_metrics(snapshot, [True, False, True, False])
    assert zero["top_200_recall"] is None


def test_compact_result_references_shared_trace_without_changing_paid_feedback():
    source = {
        "queries": [
            {
                "index": 3,
                "feedback": {
                    "collision": True,
                    "safety": {"rss_margin": -0.4},
                    "trajectory": [{"collision": True}],
                },
            }
        ]
    }
    compact = compact_result(source, ROOT / "artifacts/data/example")
    query = compact["queries"][0]
    assert "trajectory" not in query["feedback"]
    assert source["queries"][0]["feedback"]["trajectory"] == [{"collision": True}]
    assert query["feedback"]["collision"] is True
    assert (
        query["execution_ref"].replace("\\", "/")
        == "artifacts/data/example/executions/000003.json.gz"
    )


@pytest.fixture
def attribution_records(tmp_path, monkeypatch):
    monkeypatch.setattr(attribution_report, "OUTPUT", tmp_path)
    monkeypatch.setattr(attribution_report, "CHECKPOINTS", (1, 2))
    monkeypatch.setattr(
        attribution_report,
        "registered_runs",
        lambda: {("ads", "car_following", "joint"): {}},
    )
    write_json(
        tmp_path / "reports/source_views.json",
        [{"scenario": "car_following", "affected": True}],
    )
    for variant in (*attribution_report.VARIANTS, "historical_collision_fixed"):
        folder = (
            "common_feedback" if variant in attribution_report.VARIANTS else "corrected_control"
        )
        write_json(
            tmp_path / folder / "ads/car_following" / f"{variant}.json.gz",
            {
                "specification": {
                    "sut_id": "ads",
                    "scenario": "car_following",
                    "variant": variant,
                },
                "metrics": {"selection_s": 1.0, "adaptation_s": 2.0, "replay_wall_s": 3.0},
                "result": {
                    "prediction_snapshots": [
                        {
                            "paid_queries": n,
                            "observed_indices": list(range(n)),
                            "probabilities": [0.5, 0.5, 0.5],
                            "component_weights": [0.75, 0.25],
                            "evaluation": {"nll": 0.7},
                        }
                        for n in (1, 2)
                    ]
                },
            },
        )
    return tmp_path


def test_report_reconstructs_all_records_and_costs_without_progress_log(attribution_records):
    records, corrected, costs = attribution_report.feedback_report()
    assert len(records) == 6
    assert set(corrected) == {("ads", "car_following")}
    assert costs == {"selection_s": 4.0, "adaptation_s": 8.0, "replay_wall_s": 12.0}
    assert not (attribution_records / "reports/progress.json").exists()


def test_report_rejects_missing_paid_comparison(attribution_records):
    path = attribution_records / "common_feedback/ads/car_following/collision.json.gz"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        attribution_report.feedback_report()


def test_shared_private_report_rejects_duplicate_registered_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_private_report, "OUTPUT", tmp_path)
    monkeypatch.setattr(
        shared_private_report,
        "registered_runs",
        lambda: {("first", "car_following", "joint"): {}, ("second", "car_following", "joint"): {}},
    )
    task = {"sut_id": "first", "scenario": "car_following"}
    write_json(tmp_path / "protocol.json", {"tasks": [task, task]})
    with pytest.raises(ValueError, match="every registered task exactly once"):
        shared_private_report.report_e2()
