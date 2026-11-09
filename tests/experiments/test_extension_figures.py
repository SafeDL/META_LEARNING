import pytest
from PIL import Image

from experiments import report_extensions
from experiments.joint_figures import plot_retention
from experiments.storage import write_json


def test_candidate_figures_preserve_truth_counts_and_undefined_ads(tmp_path, monkeypatch):
    child_reports = []
    monkeypatch.setattr(report_extensions, "report", lambda root: child_reports.append(root.name))
    rows = []
    for system in ("independent_safe", "independent_failures"):
        for size in (4, 8, 12):
            for method in ("joint", "random"):
                total = 0 if system.endswith("safe") else size // 2
                count = 0 if total == 0 else 2 if method == "joint" else 1
                rows.append(
                    {
                        "sut_id": system,
                        "candidate_count": size,
                        "method": method,
                        "metrics": {
                            "pool_unique_collisions": total,
                            "unique_collisions": count,
                            "discovery_curve": [0, count, count],
                            "collision_recall": count / total if total else None,
                            "pair_distance": 0.3 if count >= 2 else None,
                            "nearest_neighbor_distance": 0.3 if count >= 2 else None,
                        },
                    }
                )
    write_json(tmp_path / "reports/summary.json", {"status": "complete", "runs": rows})
    result = report_extensions.sensitivity(tmp_path)
    assert child_reports == ["4", "8", "12"]
    assert result["defined_ads"]["collision_recall"]["joint"] == [1, 1, 1]
    assert result["defined_ads"]["pair_distance"]["random"] == [0, 0, 0]
    assert [r["true_failures"] for r in result["pools"]] == [0, 0, 0, 2, 4, 6]
    for relative in result["figures"]:
        with Image.open(tmp_path / "reports" / relative) as image:
            assert image.info["dpi"] == pytest.approx((600, 600), abs=0.1)
    rows[-1]["metrics"]["pool_unique_collisions"] = 5
    write_json(tmp_path / "reports/summary.json", {"status": "complete", "runs": rows})
    with pytest.raises(ValueError, match="different candidate truth"):
        report_extensions.sensitivity(tmp_path)


def test_compute_figures_keep_cpu_rss_and_cuda_allocation_separate(tmp_path):
    rows = [
        {
            "axis": "candidates",
            "device": device,
            "settings": {"candidates": n},
            "source_construction_s": 0.01 * n,
            "adaptation_s": 0.1,
            "prediction_s": 0.001 * n,
            "resources": {
                "sampled_peak_cpu_bytes": 2**20,
                "peak_cuda_allocated_bytes": 0 if device == "cpu" else 2**19,
            },
        }
        for device in ("cpu", "cuda")
        for n in (4, 8)
    ]
    write_json(
        tmp_path / "reports/scaling.json",
        {"status": "complete", "runs": rows, "hardware": {"engineering_fixture": True}},
    )
    result = report_extensions.scaling(tmp_path)
    assert result["warmups_excluded"] and len(result["figures"]) == 5
    assert any("sampled_peak_cpu_bytes" in p for p in result["figures"])
    assert any("peak_cuda_allocated_bytes" in p for p in result["figures"])
    for relative in result["figures"]:
        with Image.open(tmp_path / "reports" / relative) as image:
            assert image.info["dpi"] == pytest.approx((600, 600), abs=0.1)


def test_completed_task_replay_exports_both_improvement_and_regression(tmp_path, monkeypatch):
    from experiments import joint_figures

    figures = []
    original = joint_figures.save_figure

    def capture(figure, path):
        figures.append(figure)
        original(figure, path)

    monkeypatch.setattr(joint_figures, "save_figure", capture)
    rows = [
        {"scenario": "car_following", "after_systems": 2, "nll_change": -0.2, "brier_change": -0.1},
        {"scenario": "car_following", "after_systems": 2, "nll_change": 0.4, "brier_change": 0.2},
    ]
    paths = plot_retention(rows, tmp_path)
    assert len(paths) == 1
    assert figures[0].axes[0].collections[0].get_offsets()[:, 1].tolist() == pytest.approx(
        [-0.2, 0.4]
    )
    assert figures[0].axes[0].lines[0].get_ydata() == pytest.approx([0.1])
    with Image.open(tmp_path / paths[0]) as image:
        assert image.info["dpi"] == pytest.approx((600, 600), abs=0.1)
