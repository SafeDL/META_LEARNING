"""Synthetic layout fixtures stay in pytest temporary files, never research figures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.ticker import MaxNLocator
from PIL import Image

from controllers.highway.catalog import FORMAL_TARGET_SUTS
from experiments.plotting import k_shot as plot
from experiments.k_shot import METRICS
from experiments.storage import write_json


@pytest.fixture
def summary():
    generator = np.random.default_rng(7)
    scopes = {}
    for name in ("all", *plot.RUN_SCENARIOS):
        mean = generator.uniform(0.1, 0.8, (len(plot.METHODS), len(plot.CHECKPOINTS), len(METRICS)))
        mean[:, :, 0] += 0.5
        counts = np.full((len(plot.METHODS), len(plot.CHECKPOINTS), len(METRICS)), 8)
        counts[:, :, 2] = 7
        scopes[name] = {
            "design": {
                "methods": list(plot.METHODS),
                "registered_ads": 8,
                "ads_ids": [s.sut_id for s in FORMAL_TARGET_SUTS],
            },
            "unpaired": {
                "mean": mean.tolist(),
                "ci_low": (mean - 0.09).tolist(),
                "ci_high": (mean + 0.09).tolist(),
                "valid_ads": counts.tolist(),
            },
            "paired": {
                "ads_gain": generator.uniform(
                    -0.15, 0.15, (8, len(plot.METHODS), len(plot.CHECKPOINTS), len(METRICS))
                ).tolist()
            },
        }
    return {
        "experiment": "strict_k_shot",
        "metrics": list(METRICS),
        "checkpoints": list(plot.CHECKPOINTS),
        "scenes": list(plot.RUN_SCENARIOS),
        "seeds": list(plot.SEEDS),
        "scopes": scopes,
    }


def report_fixture(root, summary):
    destination = root / "reports/main_results/prediction"
    protocol = {"modes": ["prediction"], "scope": "synthetic unit fixture"}
    write_json(destination / "protocol.json", protocol)
    write_json(destination / "prediction_summary.json", summary)
    write_json(destination / "audited_records.json", {"scope": "unit fixture"})
    (destination / "metrics.csv").write_text("unit_fixture\n", encoding="utf-8")
    write_json(
        destination / "completion.json",
        {
            "status": "complete",
            "protocol": protocol,
            "cases": 8 * len(plot.RUN_SCENARIOS) * len(plot.SEEDS) * len(plot.METHODS),
            "metric_rows": 8 * len(plot.RUN_SCENARIOS) * len(plot.SEEDS) * len(plot.METHODS) * len(plot.CHECKPOINTS),
            "files": {
                name: plot._sha(destination / name)
                for name in ("prediction_summary.json", "audited_records.json", "metrics.csv")
            },
        },
    )
    return destination


def test_only_complete_unchanged_reports_can_enter_plotting(summary, tmp_path):
    directory = report_fixture(tmp_path, summary)
    assert plot.load_summary(tmp_path) == summary
    (directory / "metrics.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        plot.load_summary(tmp_path)


def test_no_results_produce_no_placeholder_images(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot.render(tmp_path, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()


def test_canvas_check_ignores_undrawn_ticks_but_rejects_overflow(tmp_path):
    fig, ax = plt.subplots(figsize=(2, 2))
    fig.subplots_adjust(left=0.25, right=0.9, bottom=0.25, top=0.95)
    ax.plot([0, 1], [0, 0.5])
    ax.set_ylim(0, 0.51)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    assert max(ax.get_yticks()) > ax.get_ylim()[1]
    assert plot._save(fig, [ax], tmp_path / "ticks.png").is_file()

    fig, ax = plt.subplots(figsize=(2, 2))
    fig.text(1.1, 0.5, "outside")
    with pytest.raises(ValueError, match="outside the canvas"):
        plot._save(fig, [ax], tmp_path / "overflow.png")
    plt.close(fig)


def test_five_pngs_keep_labels_intervals_and_paired_effect_direction(
    summary, tmp_path, monkeypatch
):
    report_fixture(tmp_path, summary)
    original = plot._save
    matrices = []

    def inspect(fig, axes, path):
        assert all(not ax.get_title() for ax in np.ravel(axes))
        if "paired" in path.name:
            matrices.extend(np.asarray(ax.images[0].get_array()) for ax in axes)
        else:
            assert [t.get_text() for t in fig.legends[0].get_texts()] == [
                v[0] for v in plot.METHODS.values()
            ]
            assert all(len(ax.lines) == len(plot.METHODS) and len(ax.collections) == len(plot.METHODS) for ax in np.ravel(axes))
            for ax in np.ravel(axes):
                assert len(ax.get_xticks()) == len(plot.CHECKPOINTS)
                for line in ax.lines:
                    assert len(line.get_xdata()) == 7
                    assert ax.get_xlim()[1] > line.get_xdata()[-1]
        return original(fig, axes, path)

    monkeypatch.setattr(plot, "_save", inspect)
    paths = plot.render(tmp_path, tmp_path / "figures")
    assert len(paths) == 5
    assert {p.suffix for p in (tmp_path / "figures").iterdir()} == {".png"}
    for path in paths:
        with Image.open(path) as image:
            assert image.width > 4000 and image.height > 2000
    assert len(matrices) == 1 and matrices[0].shape == (8, len(plot.RUN_SCENARIOS))
    for column, scene in enumerate(plot.RUN_SCENARIOS):
        np.testing.assert_array_equal(
            matrices[0][:, column],
            np.asarray(summary["scopes"][scene]["paired"]["ads_gain"])[:, list(plot.METHODS).index("full"), 2, 0],
        )
    assert not plt.get_fignums()
