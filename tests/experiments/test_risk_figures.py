from dataclasses import asdict

import numpy as np
import pytest
from PIL import Image

from experiments.risk_figures import (
    candidate_projection,
    projected_probability,
    risk_figures,
    snapshot_diagnostics,
)
from experiments.scenarios import candidate_pool, features
from experiments.storage import read_json, write_json


def test_projection_uses_only_public_coordinates_and_preserves_the_pca_transform():
    scenes = candidate_pool("front_brake", 8, replicate=92)
    result = candidate_projection(scenes)
    basis = np.asarray(result["components"])
    assert np.allclose(basis @ basis.T, np.eye(2), atol=1e-12)
    expected = (features(scenes) - np.asarray(result["mean"])) @ basis.T
    assert np.allclose(expected, result["coordinates"], atol=1e-12)
    assert 0 < sum(result["explained_variance_ratio"]) <= 1
    assert result["candidate_ids"] == [s.scenario_id for s in scenes]
    mapping = result["probability_map"]
    assert np.asarray(mapping["counts"]).sum() == len(scenes)
    assert mapping["axis_labels"] == ["Net gap (m)", "Closing speed (m/s)"]


def test_probability_map_aggregates_overlapping_candidates_without_order_bias():
    mapping = {
        "coordinates": [[0.2, 0.2], [0.2, 0.2], [0.8, 0.8], [2.0, 2.0]],
        "x_edges": [0, 1, 2],
        "y_edges": [0, 1, 2],
        "counts": [[3, 0], [0, 1]],
    }
    probabilities = [0.1, 0.5, 0.9, 0.8]
    expected = [[0.5, np.nan], [np.nan, 0.8]]
    assert np.allclose(projected_probability(probabilities, mapping), expected, equal_nan=True)
    reverse = {**mapping, "coordinates": list(reversed(mapping["coordinates"]))}
    assert np.allclose(
        projected_probability(list(reversed(probabilities)), reverse), expected, equal_nan=True
    )


def test_snapshot_scores_separate_paid_and_unqueried_actual_truth():
    run = {
        "truth": np.array([True, False, True, False]),
        "selected": [0, 1],
        "snapshots": [
            {"paid_queries": 2, "observed_indices": [0, 1], "probabilities": [0.9, 0.1, 0.2, 0.8]}
        ],
    }
    scores = snapshot_diagnostics(run)[0]
    assert scores["paid"]["brier"] == pytest.approx(0.01)
    assert scores["unqueried"]["brier"] == pytest.approx(0.64)
    assert scores["full_library"]["brier"] == pytest.approx(0.325)
    assert scores["paid"]["nll"] == pytest.approx(-np.log(0.9))
    run["snapshots"][0]["observed_indices"] = [0, 2]
    with pytest.raises(ValueError, match="paid query prefix"):
        snapshot_diagnostics(run)


def test_cut_out_physical_labels_fit_multimethod_and_snapshot_panels(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from experiments import risk_figures as figures
    from experiments.joint_figures import STYLE, STYLES

    scenes = candidate_pool("cut_out", 8, replicate=92)
    projection = candidate_projection(scenes)
    run = {
        "candidates": scenes,
        "truth": np.array([True, False] * 4),
        "selected": [0, 1, 2, 3],
        "weights": [],
        "snapshots": [
            {"paid_queries": n, "observed_indices": list(range(n)), "probabilities": [0.5] * 8}
            for n in (0, 1, 2, 4)
        ],
    }
    members = [{"method": name, "result_path": name} for name in tuple(STYLES)[:8]]
    checked = []

    def inspect(figure, path):
        if "physical" in path.name or "snapshots" in path.name:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            for axis in figure.axes:
                if "Obstacle" not in axis.get_ylabel():
                    continue
                label = axis.yaxis.label.get_window_extent(renderer)
                assert axis.bbox.y0 <= label.y0 < label.y1 <= axis.bbox.y1
            checked.append(path.name)
        plt.close(figure)

    monkeypatch.setattr(figures, "save_figure", inspect)
    with plt.rc_context(STYLE):
        figures.plot_space(members, {m["result_path"]: run for m in members}, projection, tmp_path)
        figures.plot_snapshots(run, projection, tmp_path)
    assert len(checked) == 2


def test_spatial_figures_and_saved_predictions_use_measured_truth_without_queries(tmp_path):
    scenes = candidate_pool("front_brake", 8, replicate=92)
    truth = [True, False] * 4
    rows = []
    for method, comparison in (("joint", "main"), ("random", "main"), ("collision", "control")):
        selected = [0, 1, 2, 3]
        probability = [0.6 if y else 0.4 for y in truth]
        snapshots = (
            [
                {
                    "paid_queries": n,
                    "observed_indices": selected[:n],
                    "probabilities": [0.5] * 8 if n == 0 else probability,
                }
                for n in (0, 2, 4)
            ]
            if method != "random"
            else []
        )
        queries = [
            {
                "query": i + 1,
                "feedback": {"collision": truth[index]},
                **(
                    {
                        "prequery_probability": probability[index],
                        "component_weights_after_query": [0.4, 0.6],
                    }
                    if method != "random"
                    else {}
                ),
            }
            for i, index in enumerate(selected)
        ]
        relative = f"methods/{method}/result.json"
        write_json(
            tmp_path / relative,
            {
                "specification": {"candidates": [asdict(s) for s in scenes]},
                "evaluation": {"candidate_collisions": truth},
                "result": {
                    "selected_indices": selected,
                    "prediction_snapshots": snapshots,
                    "queries": queries,
                    "component_names": ["history", "target"],
                },
            },
        )
        rows.append(
            {
                "method": method,
                "comparison": comparison,
                "scenario": "front_brake",
                "sut_id": "engineering_fixture",
                "result_path": relative,
                "metrics": {"discovered_ids": [scenes[i].scenario_id for i in (0, 2)]},
            }
        )
    for row in rows[:]:
        relative = row["result_path"].replace("methods/", "other_methods/")
        write_json(tmp_path / relative, read_json(tmp_path / row["result_path"]))
        rows.append({**row, "sut_id": "second_system", "result_path": relative})
    files, diagnosis = risk_figures(rows, tmp_path / "reports")
    assert len(files) == 10
    assert diagnosis["status"] == "complete"
    assert diagnosis["figure_systems"] == {"front_brake": "engineering_fixture"}
    assert len(diagnosis["snapshot_scores"]) == 6
    assert diagnosis["projections"]["front_brake"] == candidate_projection(scenes)
    assert len(diagnosis["snapshot_scores"]["methods/joint/result.json"]) == 3
    assert diagnosis["snapshot_scores"]["methods/random/result.json"] == []
    for relative in files:
        with Image.open(tmp_path / "reports" / relative) as figure:
            assert figure.info["dpi"] == pytest.approx((600, 600), abs=0.1)
    assert not list(tmp_path.rglob("*.pdf"))
    rows[0]["metrics"]["discovered_ids"] = [scenes[0].scenario_id]
    with pytest.raises(ValueError, match="candidate-ID metrics"):
        risk_figures(rows, tmp_path / "reports")
