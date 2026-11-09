import pytest
from PIL import Image

from experiments.failure_metrics import METRIC_CONTRACT
from experiments.report_joint import METRICS, paired_difference, report, task_mean
from experiments.storage import write_json


def metrics(count):
    distance = 0.5 if count == 2 else None
    return {
        **dict.fromkeys(METRICS, 0),
        "metric_contract": METRIC_CONTRACT,
        "scenario": "car_following",
        "unique_collisions": count,
        "post_initial_collisions": max(0, count - 1),
        "pool_unique_collisions": 4,
        "discovery_auc": (1 + count) / 2,
        "first_collision_query": 1,
        "collision_recall": count / 4,
        "pair_distance": distance,
        "nearest_neighbor_distance": distance,
        "distances_by_discovery_count": {
            "pair_distance": [None] + ([distance] if count == 2 else []),
            "nearest_neighbor_distance": [None] + ([distance] if count == 2 else []),
        },
        "distance_curves": {
            "pair_distance": [None, distance],
            "nearest_neighbor_distance": [None, distance],
        },
        "initial_count": 1,
        "discovery_curve": [1, count],
        "recall_curve": [0.25, count / 4],
        "equivalent_testing_curve_s": [0.5, 1.2],
    }


def rows():
    return [
        {
            "sut_id": "independent_idm_mobil",
            "lineage": "rule",
            "scenario": "car_following",
            "seed": 11,
            "method": method,
            "held_family": held,
            "metrics": metrics(2 if method == "joint" else 1),
            "resources": {
                "wall_s": 2,
                "sampled_peak_cpu_bytes": 1024,
                "peak_cuda_allocated_bytes": 0,
            },
        }
        for held in (None, "rule")
        for method in ("joint", "collision")
    ]


def test_tables_weight_tasks_equally_and_keep_missing_values():
    tasks = [
        {"lineage": lineage, "metrics": {"value": value}}
        for lineage, value in (("rule", 0), ("rule", 0), ("search", 6), ("search", None))
    ]
    assert task_mean(tasks, "value") == 2
    assert task_mean(tasks[-1:], "value") is None


def test_report_exports_candidate_metrics_without_bootstrap_or_pdf(tmp_path):
    write_json(tmp_path / "reports/summary.json", {"status": "complete", "runs": rows()})
    summary = report(tmp_path)
    assert summary["means_by_task"]["all_sources/joint"]["unique_collisions"] == 2
    assert summary["means_by_task"]["all_sources/collision"]["pair_distance"] is None
    comparisons = summary["paired_differences"]
    assert set(comparisons) == {"all_sources/collision", "leave_family_out/collision"}
    for comparison in comparisons.values():
        assert comparison["unique_collisions"]["mean"] == -1
        assert comparison["unique_collisions"]["tasks"] == 1
        assert "interval" not in comparison["unique_collisions"]
    assert summary["task_coverage"]["all_sources/collision"]["distance_undefined_tasks"] == 1
    assert len(summary["figures"]) == 18
    assert not list(tmp_path.rglob("*.pdf"))
    for relative in summary["figures"]:
        with Image.open(tmp_path / "reports" / relative) as figure:
            assert figure.width == pytest.approx(4252, abs=2)
            assert figure.info["dpi"] == pytest.approx((600, 600), abs=0.1)
    report_text = (tmp_path / "reports/report.md").read_text(encoding="utf-8")
    assert "95%" not in report_text
    assert "真实失效数" in report_text
    assert "最近邻距离" in report_text


def test_comparison_groups_keep_all_baselines_and_compute_distances_separately():
    from experiments.joint_figures import comparison_sets, curve_data, defined_mean

    tasks = [
        {"method": name, "comparison": comparison, "metrics": metrics(count)}
        for name, comparison, count in (
            ("joint", "main", 2),
            ("random", "main", 2),
            ("collision", "control", 1),
            ("no_history", "control", 1),
        )
    ]
    comparisons = comparison_sets(tasks)
    assert [r["method"] for r in comparisons["main"]] == ["joint", "random"]
    assert [r["method"] for r in comparisons["controls"]] == ["joint", "collision", "no_history"]
    assert comparisons["controls"][0] is comparisons["main"][0]
    for name, expected in (("main", [2]), ("controls", [])):
        common = min(r["metrics"]["unique_collisions"] for r in comparisons[name])
        x, values = curve_data(tasks[0]["metrics"], "pair_distance", "discoveries", common)
        assert x.tolist() == expected and len(values) == len(expected)
    means, counts = defined_mean([[None, 0.2, 0.0], [None, None, 0.4]])
    import numpy as np

    assert np.isnan(means[0])
    assert means[1:].tolist() == [0.2, 0.2]
    assert counts.tolist() == [0, 1, 2]


def test_full_pilot_layout_keeps_eight_main_methods_and_three_control_curves(tmp_path, monkeypatch):
    from experiments import joint_figures
    from experiments.joint_figures import plot_curves
    from experiments.pilot import CONTROLS, MAIN_METHODS

    tasks = [
        {
            "sut_id": sut,
            "scenario": "car_following",
            "method": method,
            "comparison": "control" if method in CONTROLS else "main",
            "metrics": metrics(2 if method == "joint" and sut.endswith("ppo") else 1),
            "resources": {"wall_s": 2, "sampled_peak_cpu_bytes": 1024},
        }
        for sut in ("independent_idm_mobil", "independent_ppo")
        for method in (*MAIN_METHODS, *CONTROLS)
    ]
    save = joint_figures.save_figure

    def inspect_then_save(figure, path):
        if path.name.endswith("_discovery.png"):
            assert all(axis.get_ylim()[1] >= 2 for axis in figure.axes[:3])
        if "equal_count" in path.name:
            assert all(axis.get_xlim()[1] > axis.get_xlim()[0] for axis in figure.axes)
        save(figure, path)

    monkeypatch.setattr(joint_figures, "save_figure", inspect_then_save)
    files, diagnostic = plot_curves(tasks, tmp_path, "method")
    assert len(files) == 18
    assert sum("/main/" in name for name in files) == 9
    assert sum("/controls/" in name for name in files) == 9
    main = diagnostic["main/car_following/False"]
    assert set(main["discovery"]) == set(MAIN_METHODS)
    assert main["discovery"]["joint"]["defined_tasks"] == [2, 2]
    assert main["pair_distance"]["joint"]["defined_tasks"] == [0, 1]
    assert set(diagnostic["controls/car_following/False"]["discovery"]) == {"joint", *CONTROLS}


def test_incomplete_pairs_and_duplicate_runs_are_not_silently_dropped(tmp_path):
    tasks = rows()
    tasks[-1]["sut_id"] = "different_ads"
    with pytest.raises(ValueError, match="same completed tasks"):
        paired_difference(
            tasks, "leave_family_out/joint", "leave_family_out/collision", "unique_collisions"
        )
    tasks = rows()
    tasks.append(tasks[0])
    write_json(tmp_path / "reports/summary.json", {"status": "complete", "runs": tasks})
    with pytest.raises(ValueError, match="duplicate"):
        report(tmp_path)


@pytest.mark.parametrize("system_count", [2, 8])
@pytest.mark.parametrize("failure_count", [0, 1])
def test_empty_equal_count_panels_keep_labels_inside_canvas(
    tmp_path, monkeypatch, system_count, failure_count
):
    import matplotlib.pyplot as plt

    from experiments import joint_figures

    tasks = [
        {
            "sut_id": f"independent_ads_{system}",
            "scenario": "car_following",
            "method": method,
            "metrics": metrics(failure_count),
            "resources": {"wall_s": 2, "sampled_peak_cpu_bytes": 1024},
        }
        for system in range(system_count)
        for method in tuple(joint_figures.STYLES)[:8]
    ]
    checked = []

    def inspect(figure, path):
        if "equal_count" in path.name:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            data_axes = [axis for axis in figure.axes if axis.axison]
            assert len(data_axes) == system_count
            if system_count == 8:
                assert figure.get_size_inches()[1] * 25.4 < 230
            for axis in data_axes:
                assert axis.get_position().width * figure.get_size_inches()[0] * 25.4 > 30
                assert all(len(line.get_ydata()) == 0 for line in axis.lines)
                for text in axis.texts:
                    bounds = text.get_window_extent(renderer)
                    assert axis.bbox.x0 <= bounds.x0 < bounds.x1 <= axis.bbox.x1
                    assert axis.bbox.y0 <= bounds.y0 < bounds.y1 <= axis.bbox.y1
                bounds = axis.xaxis.label.get_window_extent(renderer)
                assert 0 <= bounds.x0 < bounds.x1 <= figure.bbox.x1
                assert 0 <= bounds.y0 < bounds.y1 <= figure.bbox.y1
            legends = [*figure.legends]
            legends.extend(axis.get_legend() for axis in figure.axes if axis.get_legend())
            assert len(legends) == 1
            bounds = legends[0].get_window_extent(renderer)
            assert 0 <= bounds.x0 < bounds.x1 <= figure.bbox.x1
            assert 0 <= bounds.y0 < bounds.y1 <= figure.bbox.y1
            checked.append(path.name)
        plt.close(figure)

    monkeypatch.setattr(joint_figures, "CURVES", joint_figures.CURVES[-2:])
    monkeypatch.setattr(joint_figures, "save_figure", inspect)
    joint_figures.plot_curves(tasks, tmp_path, "method")
    assert len(checked) == 2


def test_recorded_coarse_metrics_cannot_be_relabelled_as_new_results(tmp_path):
    tasks = rows()
    tasks[0]["metrics"].pop("metric_contract")
    write_json(tmp_path / "reports/summary.json", {"status": "complete", "runs": tasks})
    with pytest.raises(ValueError, match="candidate-ID"):
        report(tmp_path)


def test_online_cost_plot_uses_measured_wall_time_and_mib(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from experiments import joint_figures

    tasks = [
        {
            "sut_id": "independent_idm_mobil",
            "scenario": "car_following",
            "method": method,
            "resources": {"wall_s": wall_s, "sampled_peak_cpu_bytes": memory * 2**20},
            "metrics": {"equivalent_execution_s": 9000},
        }
        for method, wall_s, memory in (("joint", 20, 512), ("random", 1, 256))
    ]

    def inspect(figure, path):
        time_axis, memory_axis = figure.axes
        assert time_axis.lines[0].get_ydata().tolist() == [20, 1]
        assert time_axis.get_yscale() == "log"
        assert "log scale" in time_axis.get_ylabel()
        assert memory_axis.lines[0].get_ydata().tolist() == [512, 256]
        assert memory_axis.get_ylabel() == "Memory (MiB)"
        assert path.name == "car_following_online_costs.png"
        plt.close(figure)

    monkeypatch.setattr(joint_figures, "save_figure", inspect)
    joint_figures.plot_online_costs(
        tasks, tmp_path, "method", ["joint", "random"], ["independent_idm_mobil"]
    )


@pytest.mark.parametrize("pool_count", [0, 4])
def test_zero_discoveries_distinguish_undefined_recall_from_zero_recall(
    tmp_path, monkeypatch, pool_count
):
    import matplotlib.pyplot as plt
    import numpy as np

    from experiments import joint_figures

    measured = metrics(0)
    measured.update(
        pool_unique_collisions=pool_count,
        discovery_curve=[0, 0],
        discovery_auc=0,
        first_collision_query=None,
        collision_recall=None if pool_count == 0 else 0,
        recall_curve=[None, None] if pool_count == 0 else [0, 0],
        distances_by_discovery_count={"pair_distance": [], "nearest_neighbor_distance": []},
    )
    task = {
        "sut_id": "independent_idm_mobil",
        "scenario": "car_following",
        "method": "joint",
        "metrics": measured,
        "resources": {"wall_s": 2, "sampled_peak_cpu_bytes": 1024},
    }
    checked = set()

    def inspect(figure, path):
        axis = figure.axes[0]
        if path.stem.endswith("_recall"):
            note = " ".join(text.get_text() for text in axis.texts)
            assert ("no true failures" in note) == (pool_count == 0)
            values = np.asarray(axis.lines[0].get_ydata(), dtype=float)
            assert np.isnan(values).all() if pool_count == 0 else np.all(values == 0)
            checked.add("recall")
        elif path.stem.endswith(("_pair_distance", "_nearest_neighbor")):
            assert any("fewer than two" in text.get_text() for text in axis.texts)
            assert np.isnan(np.asarray(axis.lines[0].get_ydata(), dtype=float)).all()
            checked.add(path.stem)
        elif path.stem.endswith("_endpoints"):
            assert figure.axes[0].get_ylim() == (0, 1)
            assert figure.axes[2].get_ylim() == (0, 1)
            recall_axis = figure.axes[1]
            note = " ".join(text.get_text() for text in recall_axis.texts)
            assert ("no true failures" in note) == (pool_count == 0)
            assert len(recall_axis.collections) == (0 if pool_count == 0 else 1)
            checked.add("endpoints")
        plt.close(figure)

    monkeypatch.setattr(joint_figures, "save_figure", inspect)
    joint_figures.plot_curves([task], tmp_path, "method")
    assert len(checked) == 4
